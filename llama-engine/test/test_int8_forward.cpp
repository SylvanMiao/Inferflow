#include "llama/engine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

using namespace llama;

struct TestData {
    std::vector<int32_t> token_ids;
    int num_tokens = 0;
};

template<typename T>
T read_scalar(std::ifstream& f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

TestData read_test_data(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open test data: " << path << std::endl;
        std::exit(1);
    }

    TestData td;
    td.num_tokens = read_scalar<int32_t>(f);
    td.token_ids.resize(td.num_tokens);
    for (int i = 0; i < td.num_tokens; ++i) {
        td.token_ids[i] = read_scalar<int32_t>(f);
    }
    return td;
}

int main(int argc, char* argv[]) {
    std::cout << "=== TinyLlama Int8 Forward Test ===" << std::endl;
    if (argc < 2) {
        std::cerr << "Usage: test_int8_forward <model.bin>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string test_path = model_path;
    size_t dot = test_path.rfind(".bin");
    if (dot != std::string::npos) {
        test_path = test_path.substr(0, dot) + "_test.bin";
    } else {
        test_path += "_test.bin";
    }

    LlamaEngine engine;
    if (!engine.load(model_path)) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    TestData test = read_test_data(test_path);
    const auto& cfg = engine.config();
    if (test.num_tokens <= 0) {
        std::cerr << "No test tokens." << std::endl;
        return 1;
    }

    engine.reset_kv_cache();
    std::vector<Eigen::VectorXf> fp32_logits;
    fp32_logits.reserve(test.num_tokens);
    for (int pos = 0; pos < test.num_tokens; ++pos) {
        fp32_logits.push_back(engine.forward(test.token_ids[pos], pos));
    }

    if (!engine.quantize_int8(128)) {
        std::cerr << "Int8 quantization failed." << std::endl;
        return 1;
    }
    if (!engine.use_int8_weights(true)) {
        std::cerr << "Failed to enable Int8 weights." << std::endl;
        return 1;
    }

    engine.reset_kv_cache();
    int finite_ok = 0;
    float max_diff = 0.0f;
    int top5_overlap_sum = 0;
    int min_top5_overlap = 5;

    for (int pos = 0; pos < test.num_tokens; ++pos) {
        Eigen::VectorXf int8_logits = engine.forward(test.token_ids[pos], pos);
        const Eigen::VectorXf& ref = fp32_logits[pos];
        if (int8_logits.size() != ref.size() || int8_logits.size() != cfg.vocab_size) {
            std::cerr << "Logit size mismatch at pos=" << pos << std::endl;
            return 1;
        }

        bool finite = true;
        float local_max = 0.0f;
        std::vector<std::pair<float, int>> ref_sorted(cfg.vocab_size);
        std::vector<std::pair<float, int>> int8_sorted(cfg.vocab_size);
        for (int i = 0; i < cfg.vocab_size; ++i) {
            float a = ref(i);
            float b = int8_logits(i);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                finite = false;
            }
            local_max = std::max(local_max, std::abs(a - b));
            ref_sorted[i] = {a, i};
            int8_sorted[i] = {b, i};
        }

        std::partial_sort(ref_sorted.begin(), ref_sorted.begin() + 5, ref_sorted.end(),
                          std::greater<std::pair<float, int>>());
        std::partial_sort(int8_sorted.begin(), int8_sorted.begin() + 5, int8_sorted.end(),
                          std::greater<std::pair<float, int>>());

        int overlap = 0;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                if (ref_sorted[i].second == int8_sorted[j].second) {
                    ++overlap;
                    break;
                }
            }
        }

        finite_ok += finite ? 1 : 0;
        top5_overlap_sum += overlap;
        min_top5_overlap = std::min(min_top5_overlap, overlap);
        max_diff = std::max(max_diff, local_max);

        std::cout << "  pos=" << pos
                  << " max_diff=" << local_max
                  << " top5_overlap=" << overlap << "/5"
                  << " finite=" << (finite ? "yes" : "no")
                  << std::endl;
    }

    std::cout << "Overall max diff: " << max_diff << std::endl;
    std::cout << "Finite outputs: " << finite_ok << "/" << test.num_tokens << std::endl;
    std::cout << "Avg top5 overlap: " << (static_cast<float>(top5_overlap_sum) / test.num_tokens)
              << "/5" << std::endl;

    if (finite_ok != test.num_tokens) {
        std::cerr << "Non-finite Int8 logits found." << std::endl;
        return 1;
    }
    if (min_top5_overlap < 3) {
        std::cerr << "Int8 top-5 overlap too low." << std::endl;
        return 1;
    }
    if (static_cast<float>(top5_overlap_sum) / test.num_tokens < 4.0f) {
        std::cerr << "Average Int8 top-5 overlap too low." << std::endl;
        return 1;
    }

    std::cout << "Int8 forward smoke test passed." << std::endl;
    return 0;
}
