#pragma once
/**
 * Transformer operators.
 *
 * Each function maps 1:1 to a Python function in llama_forward.py.
 * All functions are pure (outputs written to caller-provided buffers)
 * to minimize allocations during inference.
 */

#include <Eigen/Dense>
#include "config.h"
#include "weights.h"
#include "kv_cache.h"

namespace llama {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// RMSNorm
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Root Mean Square Layer Normalization.
 *
 * Formula: output = x / sqrt(mean(x^2) + eps) * weight
 *
 * @param x       Input  [dim]
 * @param weight  Scale  [dim]
 * @param eps     Epsilon for numerical stability
 * @param output  Result [dim] (caller-allocated)
 */
void rmsnorm(const Eigen::VectorXf& x,
             const Eigen::VectorXf& weight,
             float eps,
             Eigen::VectorXf& output);

// ═══════════════════════════════════════════════════════════════════════════
// Linear (Matrix Multiply)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Linear projection: output = input * weight^T.
 *
 * Weight is stored as [out_features, in_features].
 * Input is [in_features].
 * Output is [out_features].
 */
void linear(const Eigen::VectorXf& input,
            const Eigen::MatrixXf& weight,
            Eigen::VectorXf& output);

// ═══════════════════════════════════════════════════════════════════════════
// RoPE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Precompute RoPE cos/sin caches.
 *
 * Stores cos and sin in two separate [max_seq_len, head_dim/2] matrices.
 * This replaces the complex-valued freqs_cis from the NumPy prototype.
 *
 * @param config      Model config
 * @param cos_cache   [out] [max_seq_len, head_dim/2]  column-major
 * @param sin_cache   [out] [max_seq_len, head_dim/2]
 */
void precompute_rope_cache(const ModelConfig& config,
                           Eigen::MatrixXf& cos_cache,
                           Eigen::MatrixXf& sin_cache);

/**
 * Apply Rotary Position Embedding (in-place style, HF convention).
 *
 * Pairs dimension i with i + head_dim/2 for rotation,
 * matching HuggingFace's rotate_half implementation.
 *
 * @param q          [n_heads, head_dim]  Query (mutated in place)
 * @param k          [n_kv_heads, head_dim]  Key (mutated in place)
 * @param pos        Current token position
 * @param cos_cache  [max_seq_len, head_dim/2]
 * @param sin_cache  [max_seq_len, head_dim/2]
 */
void rope(Eigen::MatrixXf& q,
          Eigen::MatrixXf& k,
          int pos,
          const Eigen::MatrixXf& cos_cache,
          const Eigen::MatrixXf& sin_cache);

// ═══════════════════════════════════════════════════════════════════════════
// GQA repeat_kv
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Expand KV heads for Grouped Query Attention.
 *
 * Repeated interleaved: kv0,kv0,kv0,... kv1,kv1,kv1,...
 *
 * @param k_all     [n_kv_heads, seq_len, head_dim]   row-major flattened
 *                   stored as [n_kv_heads * seq_len, head_dim]
 * @param n_groups  Number of Q heads per KV head
 * @param k_expanded [out] [n_heads, seq_len, head_dim]
 */
void repeat_kv(const Eigen::MatrixXf& k_all,
               int n_groups,
               Eigen::MatrixXf& k_expanded);

// ═══════════════════════════════════════════════════════════════════════════
// Softmax (stable)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Numerically stable softmax along the last axis.
 * output = exp(x - max(x)) / sum(exp(x - max(x)))
 */
void softmax(Eigen::VectorXf& x);

// ═══════════════════════════════════════════════════════════════════════════
// Attention
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Multi-Head Attention with KV Cache (GQA variant).
 *
 * Steps:
 *   1. RMSNorm on hidden state
 *   2. Q = normed @ Wq^T,  K = normed @ Wk^T,  V = normed @ Wv^T
 *   3. Apply RoPE to Q and K
 *   4. Write K and V into cache at position `pos`
 *   5. Read all cached K, V (positions 0..pos)
 *   6. Scaled dot-product attention with GQA
 *   7. Output projection: attn @ Wo^T
 *
 * @param hidden      Input hidden state [1, dim] or [dim]
 * @param lw          Layer weights
 * @param kv_cache    Persistent KV cache
 * @param layer_idx   Which layer
 * @param pos         Current position
 * @param config      Model config
 * @param cos_cache   RoPE cos cache
 * @param sin_cache   RoPE sin cache
 * @param attn_out    [out] Attention output [dim]
 */
void attention_forward(const Eigen::VectorXf& hidden,
                       const LayerWeights& lw,
                       KVCache& kv_cache,
                       int layer_idx,
                       int pos,
                       const ModelConfig& config,
                       const Eigen::MatrixXf& cos_cache,
                       const Eigen::MatrixXf& sin_cache,
                       Eigen::VectorXf& attn_out);

// ═══════════════════════════════════════════════════════════════════════════
// SwiGLU FFN
// ═══════════════════════════════════════════════════════════════════════════

/**
 * SwiGLU Feed-Forward Network.
 *
 *   1. RMSNorm → normed
 *   2. gate = SiLU(normed @ W_gate^T)
 *   3. up   = normed @ W_up^T
 *   4. merged = gate * up
 *   5. output = merged @ W_down^T
 *
 * @param hidden    Input [dim]
 * @param lw        Layer weights
 * @param config    Model config
 * @param ffn_out  [out] FFN output [dim]
 */
void swiglu_ffn(const Eigen::VectorXf& hidden,
                const LayerWeights& lw,
                const ModelConfig& config,
                Eigen::VectorXf& ffn_out);

}  // namespace ops
}  // namespace llama
