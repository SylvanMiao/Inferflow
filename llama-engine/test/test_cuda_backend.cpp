#include "llama/backend/cuda_backend.h"
#include "llama/ops.h"
#include "llama/quantization.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

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

TEST(cuda_rmsnorm_matches_cpu) {
    Eigen::VectorXf input = Eigen::VectorXf::Random(128);
    Eigen::VectorXf weight = Eigen::VectorXf::Random(128);
    Eigen::VectorXf expected;
    Eigen::VectorXf actual;

    ops::rmsnorm(input, weight, 1e-5f, expected);
    backend::cuda_rmsnorm(input, weight, 1e-5f, actual);

    for (int i = 0; i < expected.size(); ++i) {
        CHECK_CLOSE(actual(i), expected(i), 1e-5f);
    }
}

TEST(cuda_matmul_matches_cpu) {
    Eigen::VectorXf input = Eigen::VectorXf::Random(64);
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(31, 64);
    Eigen::VectorXf expected = weight * input;
    Eigen::VectorXf actual;

    backend::cuda_matmul(input, weight, actual);

    for (int i = 0; i < expected.size(); ++i) {
        CHECK_CLOSE(actual(i), expected(i), 1e-4f);
    }
}

TEST(cuda_int8_matmul_matches_cpu_reference) {
    Eigen::VectorXf input = Eigen::VectorXf::Random(64);
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(19, 64);
    auto q = quant::quantize_groupwise(weight, 32);
    Eigen::VectorXf expected = quant::matmul_dequant(q, input);
    Eigen::VectorXf actual;

    backend::cuda_matmul_int8(input, q, actual);

    for (int i = 0; i < expected.size(); ++i) {
        CHECK_CLOSE(actual(i), expected(i), 1e-4f);
    }
}

TEST(cuda_rope_matches_cpu) {
    ModelConfig cfg = ModelConfig::tiny_llama();
    Eigen::MatrixXf cos_cache;
    Eigen::MatrixXf sin_cache;
    ops::precompute_rope_cache(cfg, cos_cache, sin_cache);

    Eigen::MatrixXf q_cpu = Eigen::MatrixXf::Random(cfg.n_heads, cfg.head_dim);
    Eigen::MatrixXf k_cpu = Eigen::MatrixXf::Random(cfg.n_kv_heads, cfg.head_dim);
    Eigen::MatrixXf q_cuda = q_cpu;
    Eigen::MatrixXf k_cuda = k_cpu;

    ops::rope(q_cpu, k_cpu, 7, cos_cache, sin_cache);
    backend::cuda_rope(q_cuda, k_cuda, 7, cos_cache, sin_cache);

    for (int r = 0; r < q_cpu.rows(); ++r)
        for (int c = 0; c < q_cpu.cols(); ++c)
            CHECK_CLOSE(q_cuda(r, c), q_cpu(r, c), 1e-5f);
    for (int r = 0; r < k_cpu.rows(); ++r)
        for (int c = 0; c < k_cpu.cols(); ++c)
            CHECK_CLOSE(k_cuda(r, c), k_cpu(r, c), 1e-5f);
}

int main() {
    std::cout << "=== CUDA Backend Tests ===" << std::endl;
    if (!backend::cuda_available()) {
        std::cout << "No CUDA device found; skipping." << std::endl;
        return 0;
    }
    RUN(cuda_rmsnorm_matches_cpu);
    RUN(cuda_matmul_matches_cpu);
    RUN(cuda_int8_matmul_matches_cpu_reference);
    RUN(cuda_rope_matches_cpu);
    std::cout << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
    return tests_failed > 0 ? 1 : 0;
}
