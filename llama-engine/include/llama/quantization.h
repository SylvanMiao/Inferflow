#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace llama {
namespace quant {

struct QuantizedMatrixInt8 {
    int rows = 0;
    int cols = 0;
    int group_size = 0;
    std::vector<int8_t> values;
    std::vector<float> scales;

    bool empty() const { return rows == 0 || cols == 0 || values.empty(); }
    int num_groups() const;
    std::string validate() const;
};

QuantizedMatrixInt8 quantize_groupwise(const Eigen::MatrixXf& weight, int group_size);

Eigen::MatrixXf dequantize_groupwise(const QuantizedMatrixInt8& qweight);

void matmul_dequant(const QuantizedMatrixInt8& qweight,
                    const Eigen::VectorXf& input,
                    Eigen::VectorXf& output);

Eigen::VectorXf matmul_dequant(const QuantizedMatrixInt8& qweight,
                               const Eigen::VectorXf& input);

}  // namespace quant
}  // namespace llama
