#include "llama/backend/cuda_backend.h"
#include "llama/engine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace llama;

namespace {

struct TestData {
    std::vector<int32_t> token_ids;
};

template<typename T>
T read_scalar(std::ifstream& f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

TestData read_token_ids(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open test data: " << path << std::endl;
        std::exit(1);
    }

    TestData td;
    int num_tokens = read_scalar<int32_t>(f);
    td.token_ids.resize(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        td.token_ids[i] = read_scalar<int32_t>(f);
    }
    return td;
}

std::string test_path_for_model(const std::string& model_path) {
    std::string test_path = model_path;
    size_t dot = test_path.rfind(".bin");
    if (dot != std::string::npos) {
        test_path = test_path.substr(0, dot) + "_test.bin";
    } else {
        test_path += "_test.bin";
    }
    return test_path;
}

std::vector<int> topk(const Eigen::VectorXf& logits, int k) {
    std::vector<std::pair<float, int>> sorted(logits.size());
    for (int i = 0; i < logits.size(); ++i) {
        sorted[i] = {logits(i), i};
    }
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                      std::greater<std::pair<float, int>>());

    std::vector<int> ids(k);
    for (int i = 0; i < k; ++i) ids[i] = sorted[i].second;
    return ids;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== CUDA Forward Parity Test ===" << std::endl;

    if (!backend::cuda_available()) {
        std::cout << "No CUDA device found; skipping." << std::endl;
        return 0;
    }

    if (argc < 2) {
        std::cerr << "Usage: test_cuda_forward <model.bin>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    TestData test = read_token_ids(test_path_for_model(model_path));

    LlamaEngine engine;
    if (!engine.load(model_path)) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    std::vector<Eigen::VectorXf> cpu_logits;
    cpu_logits.reserve(test.token_ids.size());
    engine.set_backend(backend::BackendKind::CPU);
    engine.reset_kv_cache();
    for (int pos = 0; pos < static_cast<int>(test.token_ids.size()); ++pos) {
        cpu_logits.push_back(engine.forward(test.token_ids[pos], pos));
    }

    if (!engine.set_backend(backend::BackendKind::CUDA)) {
        std::cerr << "Failed to enable CUDA backend." << std::endl;
        return 1;
    }

    engine.reset_kv_cache();
    int passed = 0;
    int failed = 0;
    float max_overall_diff = 0.0f;
    for (int pos = 0; pos < static_cast<int>(test.token_ids.size()); ++pos) {
        Eigen::VectorXf cuda_logits = engine.forward(test.token_ids[pos], pos);
        const Eigen::VectorXf& ref = cpu_logits[pos];

        float max_diff = (cuda_logits - ref).cwiseAbs().maxCoeff();
        max_overall_diff = std::max(max_overall_diff, max_diff);
        bool top5_match = topk(cuda_logits, 5) == topk(ref, 5);

        std::cout << "  pos=" << pos
                  << " max_diff=" << max_diff
                  << " top5_match=" << (top5_match ? "yes" : "no")
                  << std::endl;

        if (top5_match && max_diff < 5e-2f) {
            passed++;
        } else {
            failed++;
        }
    }

    std::cout << "Overall max diff: " << max_overall_diff << std::endl;
    std::cout << "CPU/CUDA parity: " << passed << "/" << test.token_ids.size()
              << " (" << failed << " failed)" << std::endl;
    return failed == 0 ? 0 : 1;
}
