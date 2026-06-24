#include "llama/backend/cpu_backend.h"

#include "llama/backend/backend.h"
#include "llama/ops.h"

namespace llama {
namespace backend {

void cpu_rmsnorm(const Eigen::VectorXf& input,
                 const Eigen::VectorXf& weight,
                 float eps,
                 Eigen::VectorXf& output) {
    ops::rmsnorm(input, weight, eps, output);
}

void cpu_matmul(const Eigen::VectorXf& input,
                const Eigen::MatrixXf& weight,
                Eigen::VectorXf& output) {
    ops::linear(input, weight, output);
}

}  // namespace backend
}  // namespace llama
