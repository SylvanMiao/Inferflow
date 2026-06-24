#pragma once

#include "llama/config.h"
#include "llama/quantization.h"

#include <Eigen/Dense>
#include <cstddef>

namespace llama {
namespace backend {

bool cuda_available();

void cuda_rmsnorm(const Eigen::VectorXf& input,
                  const Eigen::VectorXf& weight,
                  float eps,
                  Eigen::VectorXf& output);

void cuda_matmul(const Eigen::VectorXf& input,
                 const Eigen::MatrixXf& weight,
                 Eigen::VectorXf& output);

struct CudaMatrix;

CudaMatrix* cuda_matrix_create(const Eigen::MatrixXf& weight);
void cuda_matrix_destroy(CudaMatrix* matrix);
void cuda_matmul_device_weight(const Eigen::VectorXf& input,
                               const CudaMatrix* weight,
                               Eigen::VectorXf& output);
int cuda_matrix_rows(const CudaMatrix* matrix);
int cuda_matrix_cols(const CudaMatrix* matrix);

void cuda_matmul_int8(const Eigen::VectorXf& input,
                      const quant::QuantizedMatrixInt8& weight,
                      Eigen::VectorXf& output);

void cuda_rope(Eigen::MatrixXf& q,
               Eigen::MatrixXf& k,
               int pos,
               const Eigen::MatrixXf& cos_cache,
               const Eigen::MatrixXf& sin_cache);

void cuda_attention(const Eigen::VectorXf& q,
                    const Eigen::MatrixXf& k_cached,
                    const Eigen::MatrixXf& v_cached,
                    int seq_len,
                    int n_heads,
                    int n_kv_heads,
                    int head_dim,
                    Eigen::VectorXf& output);

// ═══════════════════════════════════════════════════════════════════════════
// GPU-resident forward context — holds KV cache + intermediates on device
// ═══════════════════════════════════════════════════════════════════════════

struct CudaForwardContext;

CudaForwardContext* cuda_context_create(const ModelConfig& cfg);
void cuda_context_destroy(CudaForwardContext* ctx);
void cuda_context_reset(CudaForwardContext* ctx);

void cuda_rmsnorm_cached(CudaForwardContext* ctx,
                         const Eigen::VectorXf& input,
                         const Eigen::VectorXf& weight,
                         float eps,
                         Eigen::VectorXf& output);

void cuda_matmul_device_weight_cached(CudaForwardContext* ctx,
                                      const Eigen::VectorXf& input,
                                      const CudaMatrix* weight,
                                      Eigen::VectorXf& output);

void cuda_rope_cached(CudaForwardContext* ctx,
                      Eigen::MatrixXf& q,
                      Eigen::MatrixXf& k,
                      int pos,
                      const Eigen::MatrixXf& cos_cache,
                      const Eigen::MatrixXf& sin_cache);

/** Write one token's K and V (CPU data) into the GPU KV cache at position pos. */
void cuda_write_kv_cache(CudaForwardContext* ctx,
                          int layer, int pos,
                          const float* k_data, const float* v_data);

/** Attention using GPU-resident KV cache. q_data is still on CPU (uploaded internally). */
void cuda_attention_cached(CudaForwardContext* ctx,
                            int layer, int seq_len,
                            const float* q_data,
                            int n_heads, int n_kv_heads, int head_dim,
                            Eigen::VectorXf& output);

}  // namespace backend
}  // namespace llama
