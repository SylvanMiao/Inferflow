#include "llama/engine.h"

#ifdef INFERFLOW_ENABLE_CUDA
#include "llama/backend/cuda_backend.h"
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llama;

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::string model_path;
    std::string tokens_path;
    std::string backend = "both";
    int repeat = 5;
    int warmup = 1;
    int max_tokens = 0;
};

struct BenchResult {
    std::string backend;
    int tokens = 0;
    int repeat = 0;
    double total_ms = 0.0;
    double avg_run_ms = 0.0;
    double avg_token_ms = 0.0;
    double tokens_per_second = 0.0;
    double min_run_ms = 0.0;
    double max_run_ms = 0.0;
};

template<typename T>
T read_scalar(std::ifstream& f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

std::string default_test_path(const std::string& model_path) {
    std::string test_path = model_path;
    size_t dot = test_path.rfind(".bin");
    if (dot != std::string::npos) {
        test_path = test_path.substr(0, dot) + "_test.bin";
    } else {
        test_path += "_test.bin";
    }
    return test_path;
}

std::vector<int> read_token_ids(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open token data: " + path);
    }

    int num_tokens = read_scalar<int32_t>(f);
    std::vector<int> token_ids(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        token_ids[i] = read_scalar<int32_t>(f);
    }
    return token_ids;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <model.bin> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --tokens PATH       Token test file. Defaults to <model>_test.bin.\n"
        << "  --backend MODE      cpu, cuda, or both. Default: both.\n"
        << "  --max-tokens N      Use only the first N tokens. Default: all.\n"
        << "  --repeat N          Measured repetitions. Default: 5.\n"
        << "  --warmup N          Warmup repetitions. Default: 1.\n";
}

Args parse_args(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        throw std::runtime_error("missing model path");
    }

    Args args;
    args.model_path = argv[1];
    args.tokens_path = default_test_path(args.model_path);

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--tokens") {
            args.tokens_path = require_value(arg);
        } else if (arg == "--backend") {
            args.backend = require_value(arg);
        } else if (arg == "--repeat") {
            args.repeat = std::stoi(require_value(arg));
        } else if (arg == "--warmup") {
            args.warmup = std::stoi(require_value(arg));
        } else if (arg == "--max-tokens") {
            args.max_tokens = std::stoi(require_value(arg));
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (args.backend != "cpu" && args.backend != "cuda" && args.backend != "both") {
        throw std::runtime_error("--backend must be cpu, cuda, or both");
    }
    if (args.repeat <= 0) throw std::runtime_error("--repeat must be positive");
    if (args.warmup < 0) throw std::runtime_error("--warmup must be non-negative");
    if (args.max_tokens < 0) throw std::runtime_error("--max-tokens must be non-negative");

    return args;
}

double run_once(LlamaEngine& engine, const std::vector<int>& token_ids) {
    engine.reset_kv_cache();
    auto start = Clock::now();
    for (int pos = 0; pos < static_cast<int>(token_ids.size()); ++pos) {
        Eigen::VectorXf logits = engine.forward(token_ids[pos], pos);
        if (logits.size() == 0) {
            throw std::runtime_error("empty logits");
        }
    }
    return elapsed_ms(start, Clock::now());
}

BenchResult benchmark_backend(LlamaEngine& engine,
                              backend::BackendKind backend_kind,
                              const std::string& backend_name,
                              const std::vector<int>& token_ids,
                              int warmup,
                              int repeat) {
    if (!engine.set_backend(backend_kind)) {
        throw std::runtime_error("failed to enable backend: " + backend_name);
    }

    for (int i = 0; i < warmup; ++i) {
        std::cout << "[bench] backend=" << backend_name
                  << " warmup=" << (i + 1) << "/" << warmup
                  << std::endl;
        (void)run_once(engine, token_ids);
    }

    std::vector<double> runs;
    runs.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        std::cout << "[bench] backend=" << backend_name
                  << " run=" << (i + 1) << "/" << repeat
                  << std::endl;
        runs.push_back(run_once(engine, token_ids));
    }

    double total = std::accumulate(runs.begin(), runs.end(), 0.0);
    auto [min_it, max_it] = std::minmax_element(runs.begin(), runs.end());

    BenchResult result;
    result.backend = backend_name;
    result.tokens = static_cast<int>(token_ids.size());
    result.repeat = repeat;
    result.total_ms = total;
    result.avg_run_ms = total / static_cast<double>(repeat);
    result.avg_token_ms = result.avg_run_ms / static_cast<double>(result.tokens);
    result.tokens_per_second = static_cast<double>(result.tokens) * 1000.0 / result.avg_run_ms;
    result.min_run_ms = *min_it;
    result.max_run_ms = *max_it;
    return result;
}

void print_result(const BenchResult& r) {
    std::cout << std::fixed << std::setprecision(3)
              << "[bench] backend=" << r.backend
              << " tokens=" << r.tokens
              << " repeat=" << r.repeat
              << " avg_run_ms=" << r.avg_run_ms
              << " avg_token_ms=" << r.avg_token_ms
              << " tok_s=" << r.tokens_per_second
              << " min_run_ms=" << r.min_run_ms
              << " max_run_ms=" << r.max_run_ms
              << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Args args = parse_args(argc, argv);
        std::vector<int> token_ids = read_token_ids(args.tokens_path);
        if (token_ids.empty()) {
            throw std::runtime_error("token list is empty");
        }
        if (args.max_tokens > 0 && args.max_tokens < static_cast<int>(token_ids.size())) {
            token_ids.resize(args.max_tokens);
        }

        std::cout << "=== Forward Benchmark ===" << std::endl;
        std::cout << "model=" << args.model_path << std::endl;
        std::cout << "tokens=" << args.tokens_path << " count=" << token_ids.size() << std::endl;
        std::cout << "warmup=" << args.warmup << " repeat=" << args.repeat << std::endl;

        LlamaEngine engine;
        if (!engine.load(args.model_path)) {
            throw std::runtime_error("failed to load model");
        }

        if (args.backend == "cpu" || args.backend == "both") {
            print_result(benchmark_backend(engine, backend::BackendKind::CPU, "cpu",
                                           token_ids, args.warmup, args.repeat));
        }

        if (args.backend == "cuda" || args.backend == "both") {
#ifdef INFERFLOW_ENABLE_CUDA
            if (!backend::cuda_available()) {
                std::cout << "[bench] backend=cuda skipped=no_cuda_device" << std::endl;
            } else {
                print_result(benchmark_backend(engine, backend::BackendKind::CUDA, "cuda",
                                               token_ids, args.warmup, args.repeat));
            }
#else
            std::cout << "[bench] backend=cuda skipped=not_built_with_cuda" << std::endl;
#endif
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
