/**
 * Operator implementations — 1:1 port from llama_forward.py.
 *
 * Memory layout conventions:
 *   - Multi-head K/V: [n_heads, seq_len * head_dim] where each row stores
 *     one head's data for all positions, flattened as [t0_d0, t0_d1, ..., t1_d0, ...]
 *   - This matches the einsum 'hd,hld->hl' pattern:
 *       h indexes the row, (l * head_dim + d) indexes the column
 */

#include "llama/ops.h"
#include <cmath>

namespace llama {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// RMSNorm
// ═══════════════════════════════════════════════════════════════════════════

void rmsnorm(const Eigen::VectorXf& x,
             const Eigen::VectorXf& weight,
             float eps,
             Eigen::VectorXf& output) {
    float rms = std::sqrt(x.squaredNorm() / static_cast<float>(x.size()) + eps);
    output = (x.array() / rms) * weight.array();
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear
// ═══════════════════════════════════════════════════════════════════════════

void linear(const Eigen::VectorXf& input,
            const Eigen::MatrixXf& weight,
            Eigen::VectorXf& output) {
    output.noalias() = weight * input;
}

// ═══════════════════════════════════════════════════════════════════════════
// RoPE
// ═══════════════════════════════════════════════════════════════════════════

void precompute_rope_cache(const ModelConfig& config,
                           Eigen::MatrixXf& cos_cache,
                           Eigen::MatrixXf& sin_cache) {
    int head_dim = config.head_dim;
    int max_seq_len = config.max_seq_len;
    int half_dim = head_dim / 2;

    cos_cache.resize(max_seq_len, half_dim);
    sin_cache.resize(max_seq_len, half_dim);

    for (int i = 0; i < half_dim; ++i) {
        float freq = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        float theta = 1.0f / std::pow(config.rope_theta, freq);
        for (int pos = 0; pos < max_seq_len; ++pos) {
            float angle = static_cast<float>(pos) * theta;
            cos_cache(pos, i) = std::cos(angle);
            sin_cache(pos, i) = std::sin(angle);
        }
    }
}

void rope(Eigen::MatrixXf& q,
          Eigen::MatrixXf& k,
          int pos,
          const Eigen::MatrixXf& cos_cache,
          const Eigen::MatrixXf& sin_cache) {
    int head_dim = static_cast<int>(q.cols());
    int half_dim = head_dim / 2;

    int q_rows = static_cast<int>(q.rows());
    int k_rows = static_cast<int>(k.rows());

    // Build full cos/sin: [cos[0..d/2-1], cos[0..d/2-1]]  — HF convention
    Eigen::VectorXf cos_full(head_dim);
    Eigen::VectorXf sin_full(head_dim);

    // cos_cache/sin_cache are column-major [max_seq_len, half_dim]
    for (int i = 0; i < half_dim; ++i) {
        float c = cos_cache(pos, i);
        float s = sin_cache(pos, i);
        cos_full(i) = c;
        cos_full(i + half_dim) = c;
        sin_full(i) = s;
        sin_full(i + half_dim) = s;
    }

    // rotate_half as in HF: x → cat(-x[half:], x[:half])
    // result = x * cos_full + rotate_half(x) * sin_full
    auto rotate = [&](Eigen::MatrixXf& x, int rows) {
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < half_dim; ++i) {
                float x0 = x(r, i);
                float x1 = x(r, i + half_dim);
                x(r, i)           = x0 * cos_full(i)        - x1 * sin_full(i);
                x(r, i + half_dim) = x0 * sin_full(i + half_dim) + x1 * cos_full(i + half_dim);
            }
        }
    };

    rotate(q, q_rows);
    rotate(k, k_rows);
}

// ═══════════════════════════════════════════════════════════════════════════
// GQA repeat_kv
// ═══════════════════════════════════════════════════════════════════════════

void repeat_kv(const Eigen::MatrixXf& k_all,
               int n_groups,
               Eigen::MatrixXf& k_expanded) {
    // k_all: [n_kv_heads, seq_len * head_dim]
    // Each row is one KV head's data for all positions, flattened.
    // We interleave: h0 × n_groups, h1 × n_groups, ...
    int n_kv_heads = static_cast<int>(k_all.rows());
    int stride = static_cast<int>(k_all.cols());
    int n_heads = n_kv_heads * n_groups;

    k_expanded.resize(n_heads, stride);
    for (int h = 0; h < n_kv_heads; ++h) {
        for (int g = 0; g < n_groups; ++g) {
            k_expanded.row(h * n_groups + g) = k_all.row(h);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax
// ═══════════════════════════════════════════════════════════════════════════

void softmax(Eigen::VectorXf& x) {
    float x_max = x.maxCoeff();
    x = (x.array() - x_max).exp();
    float sum = x.sum();
    if (sum > 0.0f) x /= sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Attention
// ═══════════════════════════════════════════════════════════════════════════

void attention_forward(const Eigen::VectorXf& hidden,
                       const LayerWeights& lw,
                       KVCache& kv_cache,
                       int layer_idx,
                       int pos,
                       const ModelConfig& config,
                       const Eigen::MatrixXf& cos_cache,
                       const Eigen::MatrixXf& sin_cache,
                       Eigen::VectorXf& attn_out) {
    int dim       = config.dim;
    int n_heads   = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_dim  = config.head_dim;
    int kv_dim    = config.kv_dim;
    int n_groups  = config.n_groups();

    // ---- 1. RMSNorm ----
    Eigen::VectorXf normed(dim);
    rmsnorm(hidden, lw.attn_norm, config.norm_eps, normed);

    // ---- 2. QKV Projections ----
    Eigen::VectorXf q_vec(dim);
    Eigen::VectorXf k_vec(kv_dim);
    Eigen::VectorXf v_vec(kv_dim);
    linear(normed, lw.attn_q, q_vec);
    linear(normed, lw.attn_k, k_vec);
    linear(normed, lw.attn_v, v_vec);

    // Reshape into heads using ROW-MAJOR (matching NumPy convention)
    // NumPy: k[h, d] = k_flat[h * head_dim + d]
    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RowMat> q_heads(q_vec.data(), n_heads, head_dim);
    Eigen::Map<RowMat> k_heads(k_vec.data(), n_kv_heads, head_dim);
    Eigen::Map<RowMat> v_heads(v_vec.data(), n_kv_heads, head_dim);

    // ---- 3. RoPE ----
    // rope() takes MatrixXf (column-major), so copy into column-major for the call
    Eigen::MatrixXf q_cm = q_heads;
    Eigen::MatrixXf k_cm = k_heads;
    rope(q_cm, k_cm, pos, cos_cache, sin_cache);
    // Copy back
    Eigen::Map<RowMat>(q_vec.data(), n_heads, head_dim) = q_cm;
    Eigen::Map<RowMat>(k_vec.data(), n_kv_heads, head_dim) = k_cm;

    // ---- 4. Write to KV Cache ----
    kv_cache.write(layer_idx, pos, k_vec, v_vec);

    // ---- 5. Scaled Dot-Product Attention ----
    // ALL head matrices use RowMajor to match NumPy indexing convention:
    //   mat(h, d) = data[h * cols + d]  (same as NumPy)
    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    int seq_len = pos + 1;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const Eigen::MatrixXf& k_cached = kv_cache.k(layer_idx);  // [max_seq_len, kv_dim] col-major
    const Eigen::MatrixXf& v_cached = kv_cache.v(layer_idx);

    // Build k_all: [n_kv_heads, seq_len * head_dim]  ROW-MAJOR
    // Row h contains K for this head, flattened as [t0_d0, t0_d1, ..., t1_d0, ...]
    RowMat k_all(n_kv_heads, seq_len * head_dim);
    RowMat v_all(n_kv_heads, seq_len * head_dim);

    for (int h = 0; h < n_kv_heads; ++h) {
        for (int t = 0; t < seq_len; ++t) {
            int kv_offset = h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                k_all(h, t * head_dim + d) = k_cached(t, kv_offset + d);
                v_all(h, t * head_dim + d) = v_cached(t, kv_offset + d);
            }
        }
    }

    // Expand for GQA: interleave rows (kv0×8, kv1×8, ...)
    RowMat k_exp(n_heads, seq_len * head_dim);
    RowMat v_exp(n_heads, seq_len * head_dim);
    for (int h = 0; h < n_kv_heads; ++h) {
        for (int g = 0; g < n_groups; ++g) {
            k_exp.row(h * n_groups + g) = k_all.row(h);
            v_exp.row(h * n_groups + g) = v_all.row(h);
        }
    }

    // Attention per head — RowMajor matching NumPy
    RowMat attn_heads(n_heads, head_dim);
    attn_heads.setZero();

    for (int h = 0; h < n_heads; ++h) {
        Eigen::VectorXf scores(seq_len);
        for (int t = 0; t < seq_len; ++t) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_heads(h, d) * k_exp(h, t * head_dim + d);
            }
            scores(t) = dot * scale;
        }

        softmax(scores);

        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                attn_heads(h, d) += scores(t) * v_exp(h, t * head_dim + d);
            }
        }
    }

    // Flatten RowMajor [n_heads, head_dim] → [dim]
    // RowMajor: row 0 then row 1 then ... → matches NumPy flatten
    Eigen::Map<Eigen::VectorXf> attn_flat(attn_heads.data(), dim);

    // ---- 6. Output Projection ----
    linear(attn_flat, lw.attn_output, attn_out);
}

// ═══════════════════════════════════════════════════════════════════════════
// SwiGLU FFN
// ═══════════════════════════════════════════════════════════════════════════

void swiglu_ffn(const Eigen::VectorXf& hidden,
                const LayerWeights& lw,
                const ModelConfig& config,
                Eigen::VectorXf& ffn_out) {
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;

    // 1. RMSNorm
    Eigen::VectorXf normed(dim);
    rmsnorm(hidden, lw.ffn_norm, config.norm_eps, normed);

    // 2. Gate: SiLU(x @ W_gate^T) = x * sigmoid(x)
    Eigen::VectorXf gate(hidden_dim);
    linear(normed, lw.ffn_gate, gate);
    for (int i = 0; i < hidden_dim; ++i) {
        gate(i) = gate(i) / (1.0f + std::exp(-gate(i)));
    }

    // 3. Up: x @ W_up^T
    Eigen::VectorXf up(hidden_dim);
    linear(normed, lw.ffn_up, up);

    // 4. Element-wise multiply
    gate = gate.array() * up.array();

    // 5. Down projection
    linear(gate, lw.ffn_down, ffn_out);
}

}  // namespace ops
}  // namespace llama
