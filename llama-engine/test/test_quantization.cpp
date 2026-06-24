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

TEST(groupwise_shapes) {
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(7, 13);
    auto q = quant::quantize_groupwise(weight, 16);

    CHECK(q.rows == 7);
    CHECK(q.cols == 13);
    CHECK(q.group_size == 16);
    CHECK(q.values.size() == static_cast<size_t>(7 * 13));
    CHECK(q.scales.size() == static_cast<size_t>((7 * 13 + 15) / 16));
    CHECK(q.validate().empty());
}

TEST(dequant_is_close_to_source) {
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(8, 16);
    auto q = quant::quantize_groupwise(weight, 32);
    Eigen::MatrixXf dq = quant::dequantize_groupwise(q);

    CHECK(dq.rows() == weight.rows());
    CHECK(dq.cols() == weight.cols());
    const float max_error = (weight - dq).cwiseAbs().maxCoeff();
    const float max_scale = *std::max_element(q.scales.begin(), q.scales.end());
    CHECK(max_error <= max_scale * 0.51f);
}

TEST(matmul_matches_dequantized_weight) {
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(11, 17);
    Eigen::VectorXf input = Eigen::VectorXf::Random(17);
    auto q = quant::quantize_groupwise(weight, 16);

    Eigen::VectorXf expected = quant::dequantize_groupwise(q) * input;
    Eigen::VectorXf actual = quant::matmul_dequant(q, input);

    CHECK(actual.size() == expected.size());
    for (int i = 0; i < actual.size(); ++i) {
        CHECK_CLOSE(actual(i), expected(i), 1e-5f);
    }
}

int main() {
    std::cout << "=== Int8 Quantization Tests ===" << std::endl;
    RUN(groupwise_shapes);
    RUN(dequant_is_close_to_source);
    RUN(matmul_matches_dequantized_weight);
    std::cout << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
    return tests_failed > 0 ? 1 : 0;
}
