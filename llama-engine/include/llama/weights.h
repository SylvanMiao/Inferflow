#pragma once
/**
 * Weight structures — maps 1:1 to Python LayerWeights / LlamaWeights.
 *
 * All matrices are row-major (Eigen default: RowMajor not needed since
 * Eigen defaults to column-major, but we use MatrixXf which is column-major
 * by default. The key point is that weight[i, j] conceptually maps to
 * weight.data()[i * cols + j] in our loading code.)
 *
 * For the MVP, we store weights as Eigen::MatrixXf (column-major by default).
 * The linear() operator handles the transpose correctly.
 */

#include <Eigen/Dense>
#include <vector>

namespace llama {

struct LayerWeights {
    // Attention
    Eigen::VectorXf  attn_norm;     // [dim]
    Eigen::MatrixXf  attn_q;        // [dim, dim]           W^Q
    Eigen::MatrixXf  attn_k;        // [kv_dim, dim]        W^K
    Eigen::MatrixXf  attn_v;        // [kv_dim, dim]        W^V
    Eigen::MatrixXf  attn_output;   // [dim, dim]           W^O

    // FFN (SwiGLU)
    Eigen::VectorXf  ffn_norm;      // [dim]
    Eigen::MatrixXf  ffn_gate;      // [hidden_dim, dim]    W1
    Eigen::MatrixXf  ffn_up;        // [hidden_dim, dim]    W3
    Eigen::MatrixXf  ffn_down;      // [dim, hidden_dim]    W2
};

struct LlamaWeights {
    Eigen::MatrixXf  token_embd;    // [vocab_size, dim]
    Eigen::VectorXf  output_norm;   // [dim]
    Eigen::MatrixXf  lm_head;       // [vocab_size, dim]  (often = token_embd)
    std::vector<LayerWeights> layers;  // [n_layers]
};

}  // namespace llama
