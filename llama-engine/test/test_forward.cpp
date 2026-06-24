/**
 * End-to-end forward pass verification against HuggingFace.
 *
 * Reads the exported model binary and test data, runs forward pass
 * token by token, and compares logits against HF's expected output.
 *
 * Usage:
 *   1. On Windows: python export_model.py --model .../TinyLlama --output tinyllama.bin
 *      (this also creates tinyllama_test.bin)
 *   2. Transfer tinyllama.bin and tinyllama_test.bin to this directory
 *   3. ./test_forward tinyllama.bin
 */

#include "llama/engine.h"
#include "llama/ops.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace llama;

// ---- Read test data ----

struct TestData {
    std::vector<int32_t> token_ids;
    std::vector<float> expected_logits;  // [num_tokens * vocab_size] column-major
    int num_tokens;
    int vocab_size;
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

    // Read rest of file as float32 (expected logits)
    // Current position is right after token_ids
    std::streamsize current = f.tellg();
    f.seekg(0, std::ios::end);
    std::streamsize total = f.tellg();
    f.seekg(current);
    std::streamsize float_bytes = total - current;
    size_t num_floats = static_cast<size_t>(float_bytes) / sizeof(float);
    td.expected_logits.resize(num_floats);
    f.read(reinterpret_cast<char*>(td.expected_logits.data()), float_bytes);

    td.vocab_size = static_cast<int>(num_floats) / td.num_tokens;

    std::cout << "Test data: " << td.num_tokens << " tokens, "
              << td.vocab_size << " vocab" << std::endl;
    return td;
}

// ---- Main comparison ----

int main(int argc, char* argv[]) {
    std::cout << "=== Phase 1: Forward Pass Verification ===" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: test_forward <model.bin>" << std::endl;
        std::cerr << "  Expects <model.bin>_test.bin in the same directory." << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string test_path = model_path;
    // Replace .bin with _test.bin
    size_t dot = test_path.rfind(".bin");
    if (dot != std::string::npos) {
        test_path = test_path.substr(0, dot) + "_test.bin";
    } else {
        test_path += "_test.bin";
    }

    // Load model
    LlamaEngine engine;
    if (!engine.load(model_path)) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    // Load test data
    TestData test = read_test_data(test_path);
    const auto& cfg = engine.config();

    // ---- Run forward pass for each token and compare ----
    int num_tokens = test.num_tokens;
    int vocab_size = cfg.vocab_size;

    if (test.vocab_size != vocab_size) {
        std::cerr << "Vocab size mismatch: model=" << vocab_size
                  << " test=" << test.vocab_size << std::endl;
        return 1;
    }

    // ---- Debug: dump weights from first layer ----
    {
        const auto& lw = engine.weights().layers[0];
        std::cout << "  attn_norm[0:5]: ";
        for (int i = 0; i < 5; ++i) std::cout << lw.attn_norm(i) << " ";
        std::cout << std::endl;
        std::cout << "  attn_q(0,0)=" << lw.attn_q(0,0)
                  << " attn_q(0,1)=" << lw.attn_q(0,1) << std::endl;
        std::cout << "  attn_k(0,0)=" << lw.attn_k(0,0)
                  << " attn_k(0,1)=" << lw.attn_k(0,1) << std::endl;
        // rmsnorm the embedding and print first 5
        auto emb = engine.weights().token_embd.row(test.token_ids[0]);
        Eigen::VectorXf normed(engine.config().dim);
        llama::ops::rmsnorm(emb, lw.attn_norm, engine.config().norm_eps, normed);
        std::cout << "  emb[0:3]: " << emb(0) << " " << emb(1) << " " << emb(2) << std::endl;
        std::cout << "  rmsnorm_out[0:5]: ";
        for (int i = 0; i < 5; ++i) std::cout << normed(i) << " ";
        std::cout << std::endl;
    }

    engine.reset_kv_cache();
    int passed = 0;
    int failed = 0;
    float max_overall_diff = 0.0f;

    for (int pos = 0; pos < num_tokens; ++pos) {
        int token = test.token_ids[pos];
        Eigen::VectorXf logits = engine.forward(token, pos);

        // Expected logits are row-major: [num_tokens, vocab_size]
        // Row `pos` contains logits for this position, contiguous: data[pos * vocab_size : (pos+1) * vocab_size]
        const float* expected = test.expected_logits.data() + static_cast<size_t>(pos) * vocab_size;

        // Compare
        float max_diff = 0.0f;
        // Collect top-5 for both
        std::vector<std::pair<float, int>> our_sorted(vocab_size);
        std::vector<std::pair<float, int>> hf_sorted(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            float diff = std::abs(logits(i) - expected[i]);
            max_diff = std::max(max_diff, diff);
            our_sorted[i] = {logits(i), i};
            hf_sorted[i] = {expected[i], i};
        }

        // Top-5 match check
        std::partial_sort(our_sorted.begin(), our_sorted.begin() + 5, our_sorted.end(),
                          std::greater<std::pair<float, int>>());
        std::partial_sort(hf_sorted.begin(), hf_sorted.begin() + 5, hf_sorted.end(),
                          std::greater<std::pair<float, int>>());

        bool top5_ok = true;
        for (int k = 0; k < 5; ++k) {
            if (our_sorted[k].second != hf_sorted[k].second) {
                top5_ok = false;
                break;
            }
        }

        max_overall_diff = std::max(max_overall_diff, max_diff);

        if (top5_ok || pos < 8) {
            std::cout << "  pos=" << pos << " max_diff=" << max_diff
                      << " top5_match=" << (top5_ok ? "✓" : "✗ MISMATCH")
                      << std::endl;
        }

        if (!top5_ok) {
            failed++;
            std::cout << "    Our top-5: ";
            for (int k = 0; k < 5; ++k) std::cout << our_sorted[k].second << " ";
            std::cout << std::endl;
            std::cout << "    HF top-5:  ";
            for (int k = 0; k < 5; ++k) std::cout << hf_sorted[k].second << " ";
            std::cout << std::endl;
        } else {
            passed++;
        }
    }

    std::cout << std::endl;
    std::cout << "Overall max diff: " << max_overall_diff << std::endl;
    std::cout << "Top-5 match: " << passed << "/" << num_tokens
              << " (" << failed << " failed)" << std::endl;

    if (failed == 0) {
        std::cout << "ALL PASSED ✓ — C++ matches HuggingFace!" << std::endl;
        return 0;
    } else {
        std::cout << "SOME FAILED ✗" << std::endl;
        return 1;
    }
}
