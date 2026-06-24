#include "llama/backend/backend.h"
#include "llama/backend/cpu_backend.h"
#include "llama/ops.h"

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

TEST(backend_names) {
    CHECK(std::string(backend::backend_name(backend::BackendKind::CPU)) == "cpu");
    CHECK(std::string(backend::backend_name(backend::BackendKind::CUDA)) == "cuda");
}

TEST(cpu_backend_matches_ops) {
    Eigen::VectorXf input = Eigen::VectorXf::Random(32);
    Eigen::VectorXf weight = Eigen::VectorXf::Random(32);
    Eigen::VectorXf expected;
    Eigen::VectorXf actual;

    ops::rmsnorm(input, weight, 1e-5f, expected);
    backend::cpu_rmsnorm(input, weight, 1e-5f, actual);

    for (int i = 0; i < expected.size(); ++i) {
        CHECK_CLOSE(actual(i), expected(i), 1e-6f);
    }
}

TEST(cpu_matmul_matches_ops) {
    Eigen::VectorXf input = Eigen::VectorXf::Random(16);
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(9, 16);
    Eigen::VectorXf expected;
    Eigen::VectorXf actual;

    ops::linear(input, weight, expected);
    backend::cpu_matmul(input, weight, actual);

    for (int i = 0; i < expected.size(); ++i) {
        CHECK_CLOSE(actual(i), expected(i), 1e-6f);
    }
}

int main() {
    std::cout << "=== Backend Tests ===" << std::endl;
    RUN(backend_names);
    RUN(cpu_backend_matches_ops);
    RUN(cpu_matmul_matches_ops);
    std::cout << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
    return tests_failed > 0 ? 1 : 0;
}
