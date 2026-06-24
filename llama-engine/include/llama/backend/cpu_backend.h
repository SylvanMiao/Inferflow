#pragma once

#include <Eigen/Dense>

namespace llama {
namespace backend {

void cpu_rmsnorm(const Eigen::VectorXf& input,
                 const Eigen::VectorXf& weight,
                 float eps,
                 Eigen::VectorXf& output);

void cpu_matmul(const Eigen::VectorXf& input,
                const Eigen::MatrixXf& weight,
                Eigen::VectorXf& output);

}  // namespace backend
}  // namespace llama
