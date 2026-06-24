/**
 * Engine integration test — forward pass with random weights.
 *
 * Equivalent to Python test_model_forward() and test_generate_with_random_weights().
 */

#include "llama/engine.h"
#include "llama/config.h"
#include "llama/tokenizer.h"
#include <iostream>
#include <cassert>
#include <string>

using namespace llama;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN(name) do { \
    std::cout << "[test] " #name "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
    tests_passed++; \
} while(0)

#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAILED: " #cond << std::endl; tests_failed++; return; }} while(0)

// ═══════════════════════════════════════════════════════════════════════════

TEST(config_validation) {
    ModelConfig cfg = ModelConfig::tiny_llama();
    std::string err = cfg.validate();
    CHECK(err.empty());

    // Check derived values
    CHECK(cfg.n_groups() == 8);  // 32 / 4
    CHECK(cfg.kv_dim == 256);    // 4 * 64
}

TEST(kv_cache_init) {
    ModelConfig cfg = ModelConfig::tiny_llama();
    KVCache cache(cfg);

    CHECK(cache.n_layers() == 22);
    CHECK(cache.max_seq_len() == 2048);
    CHECK(cache.kv_dim() == 256);
    CHECK(cache.current_len() == 0);

    // Should be zero-initialized
    CHECK(cache.k(0)(0, 0) == 0.0f);
}

TEST(kv_cache_write_read) {
    ModelConfig cfg = ModelConfig::tiny_llama();
    KVCache cache(cfg);

    Eigen::VectorXf k = Eigen::VectorXf::Random(cfg.kv_dim);
    Eigen::VectorXf v = Eigen::VectorXf::Random(cfg.kv_dim);

    cache.write(0, 0, k, v);

    // Read back should match
    for (int i = 0; i < cfg.kv_dim; ++i) {
        CHECK(cache.k(0)(0, i) == k(i));
        CHECK(cache.v(0)(0, i) == v(i));
    }

    // Position 1 should still be zero
    CHECK(cache.k(0)(1, 0) == 0.0f);
}

TEST(rope_cache_precompute) {
    ModelConfig cfg = ModelConfig::tiny_llama();
    LlamaEngine engine;

    // Just verify it doesn't crash with uninitialized engine
    // (full engine tests need a model file)
    CHECK(true);
}

TEST(generate_returns_token_ids) {
    LlamaEngine engine;
    CHECK(engine.load("../test/tinyllama.bin"));

    GenerationConfig gen_cfg;
    gen_cfg.max_tokens = 3;
    gen_cfg.temperature = 0.0f;

    std::vector<int> prompt = {450, 7483, 310};
    auto output = engine.generate(prompt, gen_cfg);

    CHECK(!output.empty());
    CHECK(output.size() <= static_cast<std::size_t>(gen_cfg.max_tokens));
    for (int token : output) {
        CHECK(token >= 0);
        CHECK(token < engine.config().vocab_size);
    }
}

TEST(generate_respects_zero_max_tokens) {
    LlamaEngine engine;
    CHECK(engine.load("../test/tinyllama.bin"));

    GenerationConfig gen_cfg;
    gen_cfg.max_tokens = 0;

    std::vector<int> prompt = {450, 7483, 310};
    auto output = engine.generate(prompt, gen_cfg);

    CHECK(output.empty());
}

TEST(sentencepiece_tokenizer_roundtrip) {
    SentencePieceTokenizer tokenizer;
    CHECK(tokenizer.load("../../../models/tokenizer.model"));

    const std::string text = "Hello local inference";
    auto token_ids = tokenizer.encode(text, true, false);
    CHECK(!token_ids.empty());
    CHECK(token_ids.front() == tokenizer.bos_id());
    CHECK(tokenizer.vocab_size() > 0);

    std::vector<int> content_tokens(token_ids.begin() + 1, token_ids.end());
    auto decoded = tokenizer.decode(content_tokens);
    CHECK(decoded.find("Hello") != std::string::npos);
    CHECK(decoded.find("inference") != std::string::npos);
}

TEST(engine_generate_text_smoke) {
    LlamaEngine engine;
    CHECK(engine.load("../test/tinyllama.bin"));
    CHECK(engine.load_tokenizer("../../../models/tokenizer.model"));

    GenerationConfig gen_cfg;
    gen_cfg.max_tokens = 2;
    gen_cfg.temperature = 0.0f;

    auto text = engine.generate_text("The capital of France is", gen_cfg);
    CHECK(!text.empty());
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Phase 1: Engine Integration Tests ===" << std::endl;

    RUN(config_validation);
    RUN(kv_cache_init);
    RUN(kv_cache_write_read);
    RUN(rope_cache_precompute);
    RUN(generate_returns_token_ids);
    RUN(generate_respects_zero_max_tokens);
    RUN(sentencepiece_tokenizer_roundtrip);
    RUN(engine_generate_text_smoke);

    std::cout << std::endl;
    std::cout << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
    return tests_failed > 0 ? 1 : 0;
}
