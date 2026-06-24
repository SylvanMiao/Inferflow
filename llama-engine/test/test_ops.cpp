/**
 * Operator unit tests — verify shapes, invariants, and numerical properties.
 *
 * These are the C++ equivalent of the Python standalone tests in llama_forward.py.
 */

#include "llama/ops.h"
#include "llama/config.h"
#include "llama/weights.h"
#include <iostream>
#include <cassert>
#include <cmath>

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
    std::cerr << "FAILED: " #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    tests_failed++; return; }} while(0)

#define CHECK_CLOSE(a, b, tol) do { if (std::abs((a) - (b)) > (tol)) { \
    std::cerr << "FAILED: |" << (a) << " - " << (b) << "| > " << (tol) << std::endl; \
    tests_failed++; return; }} while(0)

// ---- Test helpers ----

static ModelConfig test_cfg() {
    return ModelConfig::tiny_llama();
}

static LayerWeights random_layer_weights(const ModelConfig& cfg) {
    LayerWeights lw;
    int dim = cfg.dim, kv_dim = cfg.kv_dim, hdim = cfg.hidden_dim;
    lw.attn_norm    = Eigen::VectorXf::Random(dim);
    lw.attn_q       = Eigen::MatrixXf::Random(dim, dim) * 0.02f;
    lw.attn_k       = Eigen::MatrixXf::Random(kv_dim, dim) * 0.02f;
    lw.attn_v       = Eigen::MatrixXf::Random(kv_dim, dim) * 0.02f;
    lw.attn_output  = Eigen::MatrixXf::Random(dim, dim) * 0.02f;
    lw.ffn_norm     = Eigen::VectorXf::Random(dim);
    lw.ffn_gate     = Eigen::MatrixXf::Random(hdim, dim) * 0.02f;
    lw.ffn_up       = Eigen::MatrixXf::Random(hdim, dim) * 0.02f;
    lw.ffn_down     = Eigen::MatrixXf::Random(dim, hdim) * 0.02f;
    return lw;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(rmsnorm_output_shape) {
    auto cfg = test_cfg();
    Eigen::VectorXf x = Eigen::VectorXf::Random(cfg.dim);
    Eigen::VectorXf weight = Eigen::VectorXf::Ones(cfg.dim);
    Eigen::VectorXf out(cfg.dim);

    ops::rmsnorm(x, weight, cfg.norm_eps, out);

    CHECK(out.size() == cfg.dim);
    // For unit weight, output = x / rms(x)
    float expected_rms = std::sqrt(x.squaredNorm() / cfg.dim + cfg.norm_eps);
    Eigen::VectorXf expected = x / expected_rms;
    for (int i = 0; i < cfg.dim; ++i) {
        CHECK_CLOSE(out(i), expected(i), 1e-5f);
    }
}

TEST(linear_output_size) {
    auto cfg = test_cfg();
    Eigen::VectorXf input = Eigen::VectorXf::Random(cfg.dim);
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(cfg.kv_dim, cfg.dim);
    Eigen::VectorXf out(cfg.kv_dim);

    ops::linear(input, weight, out);

    CHECK(out.size() == cfg.kv_dim);
}

TEST(rope_preserves_norm) {
    auto cfg = test_cfg();
    Eigen::MatrixXf cos_cache, sin_cache;
    ops::precompute_rope_cache(cfg, cos_cache, sin_cache);

    Eigen::MatrixXf q = Eigen::MatrixXf::Random(cfg.n_heads, cfg.head_dim);
    Eigen::MatrixXf k = Eigen::MatrixXf::Random(cfg.n_kv_heads, cfg.head_dim);

    float q_norm_before = q.norm();
    float k_norm_before = k.norm();

    ops::rope(q, k, 5, cos_cache, sin_cache);

    float q_norm_after = q.norm();
    float k_norm_after = k.norm();

    CHECK_CLOSE(q_norm_before, q_norm_after, 1e-3f);
    CHECK_CLOSE(k_norm_before, k_norm_after, 1e-3f);
}

TEST(rope_pos0_is_identity) {
    auto cfg = test_cfg();
    Eigen::MatrixXf cos_cache, sin_cache;
    ops::precompute_rope_cache(cfg, cos_cache, sin_cache);

    Eigen::MatrixXf q = Eigen::MatrixXf::Random(cfg.n_heads, cfg.head_dim);
    Eigen::MatrixXf k = Eigen::MatrixXf::Random(cfg.n_kv_heads, cfg.head_dim);
    Eigen::MatrixXf q_orig = q;
    Eigen::MatrixXf k_orig = k;

    ops::rope(q, k, 0, cos_cache, sin_cache);

    // At pos=0, cos(0)=1, sin(0)=0 for all frequencies → identity
    for (int r = 0; r < q.rows(); ++r)
        for (int c = 0; c < q.cols(); ++c)
            CHECK_CLOSE(q(r, c), q_orig(r, c), 1e-5f);
    for (int r = 0; r < k.rows(); ++r)
        for (int c = 0; c < k.cols(); ++c)
            CHECK_CLOSE(k(r, c), k_orig(r, c), 1e-5f);
}

TEST(softmax_sums_to_one) {
    Eigen::VectorXf x(10);
    x << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f;

    ops::softmax(x);

    CHECK_CLOSE(x.sum(), 1.0f, 1e-5f);
    // First < last (monotonic)
    CHECK(x(0) < x(9));
}

TEST(repeat_kv_correct_shape) {
    Eigen::MatrixXf k_all(4, 128);  // [4, 2*64] for seq_len=2, head_dim=64
    k_all.setRandom();
    Eigen::MatrixXf k_exp;

    ops::repeat_kv(k_all, 8, k_exp);  // n_groups=8 → 32 heads

    CHECK(k_exp.rows() == 32);
    CHECK(k_exp.cols() == 128);
    // First 8 rows should equal row 0
    for (int g = 0; g < 8; ++g) {
        CHECK(k_exp.row(g).isApprox(k_all.row(0)));
    }
}

TEST(attention_output_shape) {
    auto cfg = test_cfg();
    auto lw = random_layer_weights(cfg);

    Eigen::MatrixXf cos_cache, sin_cache;
    ops::precompute_rope_cache(cfg, cos_cache, sin_cache);

    KVCache kv_cache(cfg);
    Eigen::VectorXf hidden = Eigen::VectorXf::Random(cfg.dim);
    Eigen::VectorXf attn_out(cfg.dim);

    ops::attention_forward(hidden, lw, kv_cache, 0, 0, cfg,
                           cos_cache, sin_cache, attn_out);

    CHECK(attn_out.size() == cfg.dim);
}

TEST(swiglu_output_shape) {
    auto cfg = test_cfg();
    auto lw = random_layer_weights(cfg);

    Eigen::VectorXf hidden = Eigen::VectorXf::Random(cfg.dim);
    Eigen::VectorXf ffn_out(cfg.dim);

    ops::swiglu_ffn(hidden, lw, cfg, ffn_out);

    CHECK(ffn_out.size() == cfg.dim);
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Phase 1: Operator Unit Tests ===" << std::endl;

    RUN(rmsnorm_output_shape);
    RUN(linear_output_size);
    RUN(rope_preserves_norm);
    RUN(rope_pos0_is_identity);
    RUN(softmax_sums_to_one);
    RUN(repeat_kv_correct_shape);
    RUN(attention_output_shape);
    RUN(swiglu_output_shape);

    std::cout << std::endl;
    std::cout << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
    return tests_failed > 0 ? 1 : 0;
}
