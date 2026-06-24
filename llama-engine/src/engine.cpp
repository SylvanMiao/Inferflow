/**
 * LlamaEngine implementation — the main inference loop.
 *
 * Maps 1:1 to Python LlamaModel.forward() + LlamaForCausalLM.generate().
 */

#include "llama/engine.h"
#include "llama/backend/cpu_backend.h"
#include "llama/ops.h"
#ifdef INFERFLOW_ENABLE_CUDA
#include "llama/backend/cuda_backend.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <exception>
#include <sstream>

namespace llama {

// ═══════════════════════════════════════════════════════════════════════════
// Simple binary model loader (MVP format)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/** Read a scalar value from file. */
template<typename T>
T read_scalar(std::ifstream& f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

/** Read a vector from file. */
template<typename T>
void read_vector(std::ifstream& f, std::vector<T>& vec, size_t n) {
    vec.resize(n);
    f.read(reinterpret_cast<char*>(vec.data()), n * sizeof(T));
}

std::string apply_stop_strings(const std::string& text,
                               const std::vector<std::string>& stop_strings,
                               bool& stopped) {
    size_t stop_pos = std::string::npos;
    for (const auto& stop : stop_strings) {
        if (stop.empty()) continue;
        size_t pos = text.find(stop);
        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            stop_pos = pos;
        }
    }

    stopped = stop_pos != std::string::npos;
    if (!stopped) return text;
    return text.substr(0, stop_pos);
}

std::string hold_stop_prefix_suffix(const std::string& text,
                                    const std::vector<std::string>& stop_strings) {
    size_t hold_len = 0;
    for (const auto& stop : stop_strings) {
        if (stop.empty()) continue;
        size_t max_len = std::min(text.size(), stop.size() - 1);
        for (size_t len = 1; len <= max_len; ++len) {
            if (text.compare(text.size() - len, len, stop, 0, len) == 0) {
                hold_len = std::max(hold_len, len);
            }
        }
    }

    return text.substr(0, text.size() - hold_len);
}

}  // anonymous namespace

LlamaEngine::LlamaEngine()  = default;
LlamaEngine::~LlamaEngine() {
#ifdef INFERFLOW_ENABLE_CUDA
    destroy_cuda_weights();
    if (cuda_ctx_) {
        backend::cuda_context_destroy(cuda_ctx_);
        cuda_ctx_ = nullptr;
    }
#endif
}

bool LlamaEngine::set_backend(backend::BackendKind backend) {
    if (backend == backend::BackendKind::CPU) {
        backend_kind_ = backend;
        return true;
    }

#ifdef INFERFLOW_ENABLE_CUDA
    if (backend == backend::BackendKind::CUDA && backend::cuda_available()) {
        if (loaded_ && !cuda_ctx_) {
            cuda_ctx_ = backend::cuda_context_create(config_);
            backend::cuda_context_reset(cuda_ctx_);
        }
        if (loaded_) {
            ensure_cuda_weights();
        }
        backend_kind_ = backend;
        return true;
    }
#else
    (void)backend;
#endif

    return false;
}

bool LlamaEngine::load(const std::string& model_path) {
    // For MVP: load from a simple binary format.
    // Format:
    //   [int32_t] magic = 0x4C4C414D  ("LLAM")
    //   [int32_t] version
    //   [ModelConfig] config (flat struct)
    //   [weights]   raw float data in layer order
    //
    // Phase 2 will add GGUF support.

    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return false;
    }

    // Read magic and version
    int32_t magic = read_scalar<int32_t>(file);
    if (magic != 0x4C4C414D) {
        std::cerr << "Invalid model magic: " << std::hex << magic << std::endl;
        return false;
    }

    int32_t version = read_scalar<int32_t>(file);
    std::cout << "Loading model v" << version << std::endl;

    // Read config
    config_.dim         = read_scalar<int32_t>(file);
    config_.n_layers    = read_scalar<int32_t>(file);
    config_.n_heads     = read_scalar<int32_t>(file);
    config_.n_kv_heads  = read_scalar<int32_t>(file);
    config_.head_dim    = read_scalar<int32_t>(file);
    config_.hidden_dim  = read_scalar<int32_t>(file);
    config_.vocab_size  = read_scalar<int32_t>(file);
    config_.max_seq_len = read_scalar<int32_t>(file);
    config_.kv_dim      = read_scalar<int32_t>(file);
    config_.rope_theta  = read_scalar<float>(file);
    config_.norm_eps    = read_scalar<float>(file);
    config_.bos_token_id = read_scalar<int32_t>(file);
    config_.eos_token_id = read_scalar<int32_t>(file);

    std::string err = config_.validate();
    if (!err.empty()) {
        std::cerr << "Invalid config: " << err << std::endl;
        return false;
    }

    std::cout << "  dim=" << config_.dim << " layers=" << config_.n_layers
              << " vocab=" << config_.vocab_size << std::endl;

    int dim       = config_.dim;
    int hidden_dim = config_.hidden_dim;
    int vocab_size = config_.vocab_size;
    int kv_dim    = config_.kv_dim;
    int n_layers  = config_.n_layers;

    // Helper: read a float vector (1D, no layout ambiguity)
    auto read_vec = [&](int n) {
        Eigen::VectorXf v(n);
        file.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
        return v;
    };

    // Helper: read a float matrix — file is row-major (NumPy default),
    // but Eigen stores column-major. Read into temp buffer then copy.
    auto read_mat = [&](int rows, int cols) {
        Eigen::MatrixXf m(rows, cols);
        std::vector<float> buf(rows * cols);
        file.read(reinterpret_cast<char*>(buf.data()), rows * cols * sizeof(float));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                m(r, c) = buf[static_cast<size_t>(r) * cols + c];
        return m;
    };

    // ---- Global weights ----
    weights_.token_embd  = read_mat(vocab_size, dim);
    weights_.output_norm = read_vec(dim);
    weights_.lm_head     = read_mat(vocab_size, dim);

    // ---- Per-layer weights ----
    weights_.layers.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        auto& lw = weights_.layers[i];
        lw.attn_norm    = read_vec(dim);
        lw.attn_q       = read_mat(dim, dim);
        lw.attn_k       = read_mat(kv_dim, dim);
        lw.attn_v       = read_mat(kv_dim, dim);
        lw.attn_output  = read_mat(dim, dim);
        lw.ffn_norm     = read_vec(dim);
        lw.ffn_gate     = read_mat(hidden_dim, dim);
        lw.ffn_up       = read_mat(hidden_dim, dim);
        lw.ffn_down     = read_mat(dim, hidden_dim);
    }

    if (!file) {
        std::cerr << "Error reading model weights." << std::endl;
        return false;
    }

    // ---- Initialize ----
    init_buffers();
    precompute_rope();
    kv_cache_ = std::make_unique<KVCache>(config_);
    sampler_  = std::make_unique<Sampler>(GenerationConfig{});

#ifdef INFERFLOW_ENABLE_CUDA
    destroy_cuda_weights();
#endif
    int8_weights_ = Int8Weights{};
    int8_ready_ = false;
    int8_enabled_ = false;

    loaded_ = true;
#ifdef INFERFLOW_ENABLE_CUDA
    if (backend_kind_ == backend::BackendKind::CUDA && backend::cuda_available()) {
        if (!cuda_ctx_) {
            cuda_ctx_ = backend::cuda_context_create(config_);
            backend::cuda_context_reset(cuda_ctx_);
        }
        ensure_cuda_weights();
    }
#endif
    std::cout << "Model loaded successfully." << std::endl;
    return true;
}

bool LlamaEngine::quantize_int8(int group_size) {
    if (!loaded_) {
        std::cerr << "Cannot quantize Int8 before model is loaded." << std::endl;
        return false;
    }
    if (group_size <= 0) {
        std::cerr << "Invalid Int8 group size: " << group_size << std::endl;
        return false;
    }

    try {
        Int8Weights qweights;
        qweights.layers.resize(weights_.layers.size());

        std::cout << "Quantizing model weights to Int8 group-wise, group_size="
                  << group_size << "..." << std::endl;

        qweights.lm_head = quant::quantize_groupwise(weights_.lm_head, group_size);
        for (size_t i = 0; i < weights_.layers.size(); ++i) {
            const auto& src = weights_.layers[i];
            auto& dst = qweights.layers[i];
            dst.attn_q = quant::quantize_groupwise(src.attn_q, group_size);
            dst.attn_k = quant::quantize_groupwise(src.attn_k, group_size);
            dst.attn_v = quant::quantize_groupwise(src.attn_v, group_size);
            dst.attn_output = quant::quantize_groupwise(src.attn_output, group_size);
            dst.ffn_gate = quant::quantize_groupwise(src.ffn_gate, group_size);
            dst.ffn_up = quant::quantize_groupwise(src.ffn_up, group_size);
            dst.ffn_down = quant::quantize_groupwise(src.ffn_down, group_size);
        }

        int8_weights_ = std::move(qweights);
        int8_group_size_ = group_size;
        int8_ready_ = true;
        std::cout << "Int8 model weights ready." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Int8 quantization failed: " << e.what() << std::endl;
        int8_weights_ = Int8Weights{};
        int8_ready_ = false;
        int8_enabled_ = false;
        return false;
    }
}

bool LlamaEngine::use_int8_weights(bool enabled) {
    if (!enabled) {
        int8_enabled_ = false;
        return true;
    }
    if (!int8_ready_ && !quantize_int8(int8_group_size_)) {
        return false;
    }
    int8_enabled_ = true;
    return true;
}

bool LlamaEngine::load_tokenizer(const std::string& tokenizer_path) {
    auto tokenizer = std::make_unique<SentencePieceTokenizer>();
    if (!tokenizer->load(tokenizer_path)) {
        return false;
    }

    if (loaded_ && tokenizer->vocab_size() != config_.vocab_size) {
        std::cerr << "Tokenizer vocab size mismatch: tokenizer=" << tokenizer->vocab_size()
                  << " model=" << config_.vocab_size << std::endl;
        return false;
    }

    tokenizer_ = std::move(tokenizer);
    return true;
}

void LlamaEngine::init_buffers() {
    int dim       = config_.dim;
    int kv_dim    = config_.kv_dim;

    hidden_.resize(dim);
    q_buffer_.resize(dim);
    k_buffer_.resize(kv_dim);
    v_buffer_.resize(kv_dim);
    attn_buffer_.resize(dim);
    ffn_buffer_.resize(dim);
}

void LlamaEngine::precompute_rope() {
    ops::precompute_rope_cache(config_, cos_cache_, sin_cache_);
}

#ifdef INFERFLOW_ENABLE_CUDA
void LlamaEngine::destroy_cuda_weights() {
    if (!cuda_weights_ready_) return;
    for (auto& layer : cuda_weights_.layers) {
        backend::cuda_matrix_destroy(layer.attn_q);
        backend::cuda_matrix_destroy(layer.attn_k);
        backend::cuda_matrix_destroy(layer.attn_v);
        backend::cuda_matrix_destroy(layer.attn_output);
        backend::cuda_matrix_destroy(layer.ffn_gate);
        backend::cuda_matrix_destroy(layer.ffn_up);
        backend::cuda_matrix_destroy(layer.ffn_down);
        layer = {};
    }
    backend::cuda_matrix_destroy(cuda_weights_.lm_head);
    cuda_weights_.lm_head = nullptr;
    cuda_weights_.layers.clear();
    cuda_weights_ready_ = false;
}

void LlamaEngine::ensure_cuda_weights() {
    if (cuda_weights_ready_) return;

    std::cout << "Uploading CUDA matrix weights..." << std::endl;
    cuda_weights_.layers.resize(weights_.layers.size());
    for (size_t i = 0; i < weights_.layers.size(); ++i) {
        const auto& lw = weights_.layers[i];
        auto& dw = cuda_weights_.layers[i];
        dw.attn_q = backend::cuda_matrix_create(lw.attn_q);
        dw.attn_k = backend::cuda_matrix_create(lw.attn_k);
        dw.attn_v = backend::cuda_matrix_create(lw.attn_v);
        dw.attn_output = backend::cuda_matrix_create(lw.attn_output);
        dw.ffn_gate = backend::cuda_matrix_create(lw.ffn_gate);
        dw.ffn_up = backend::cuda_matrix_create(lw.ffn_up);
        dw.ffn_down = backend::cuda_matrix_create(lw.ffn_down);
    }
    cuda_weights_.lm_head = backend::cuda_matrix_create(weights_.lm_head);
    cuda_weights_ready_ = true;
    std::cout << "CUDA matrix weights ready." << std::endl;
}
#endif

void LlamaEngine::rmsnorm_backend(const Eigen::VectorXf& input,
                                  const Eigen::VectorXf& weight,
                                  Eigen::VectorXf& output) {
#ifdef INFERFLOW_ENABLE_CUDA
    if (backend_kind_ == backend::BackendKind::CUDA) {
        if (cuda_ctx_) {
            backend::cuda_rmsnorm_cached(cuda_ctx_, input, weight, config_.norm_eps, output);
        } else {
            backend::cuda_rmsnorm(input, weight, config_.norm_eps, output);
        }
        return;
    }
#endif
    backend::cpu_rmsnorm(input, weight, config_.norm_eps, output);
}

void LlamaEngine::linear_backend(const Eigen::VectorXf& input,
                                 const Eigen::MatrixXf& weight,
                                 Eigen::VectorXf& output,
                                 const quant::QuantizedMatrixInt8* int8_weight
#ifdef INFERFLOW_ENABLE_CUDA
                                 , const backend::CudaMatrix* cuda_weight
#endif
                                 ) {
    if (backend_kind_ == backend::BackendKind::CPU &&
        int8_enabled_ && int8_weight && !int8_weight->empty()) {
        quant::matmul_dequant(*int8_weight, input, output);
        return;
    }

#ifdef INFERFLOW_ENABLE_CUDA
    if (backend_kind_ == backend::BackendKind::CUDA && cuda_weight) {
        if (cuda_ctx_) {
            backend::cuda_matmul_device_weight_cached(cuda_ctx_, input, cuda_weight, output);
        } else {
            backend::cuda_matmul_device_weight(input, cuda_weight, output);
        }
        return;
    }
#endif
    backend::cpu_matmul(input, weight, output);
}

void LlamaEngine::attention_forward_backend(const Eigen::VectorXf& hidden,
                                            const LayerWeights& lw,
                                            int layer_idx,
                                            int pos,
                                            Eigen::VectorXf& attn_out
#ifdef INFERFLOW_ENABLE_CUDA
                                            , const CudaLayerDeviceWeights* cuda_lw
#endif
                                            , const Int8LayerWeights* int8_lw
                                            ) {
    int dim = config_.dim;
    int n_heads = config_.n_heads;
    int n_kv_heads = config_.n_kv_heads;
    int head_dim = config_.head_dim;
    int kv_dim = config_.kv_dim;
    int n_groups = config_.n_groups();

    Eigen::VectorXf normed(dim);
    rmsnorm_backend(hidden, lw.attn_norm, normed);

    Eigen::VectorXf q_vec(dim);
    Eigen::VectorXf k_vec(kv_dim);
    Eigen::VectorXf v_vec(kv_dim);
    linear_backend(normed, lw.attn_q, q_vec
                   , int8_lw ? &int8_lw->attn_q : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , cuda_lw ? cuda_lw->attn_q : nullptr
#endif
                   );
    linear_backend(normed, lw.attn_k, k_vec
                   , int8_lw ? &int8_lw->attn_k : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , cuda_lw ? cuda_lw->attn_k : nullptr
#endif
                   );
    linear_backend(normed, lw.attn_v, v_vec
                   , int8_lw ? &int8_lw->attn_v : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , cuda_lw ? cuda_lw->attn_v : nullptr
#endif
                   );

    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RowMat> q_heads(q_vec.data(), n_heads, head_dim);
    Eigen::Map<RowMat> k_heads(k_vec.data(), n_kv_heads, head_dim);

    Eigen::MatrixXf q_cm = q_heads;
    Eigen::MatrixXf k_cm = k_heads;
#ifdef INFERFLOW_ENABLE_CUDA
    if (backend_kind_ == backend::BackendKind::CUDA) {
        if (cuda_ctx_) {
            backend::cuda_rope_cached(cuda_ctx_, q_cm, k_cm, pos, cos_cache_, sin_cache_);
        } else {
            backend::cuda_rope(q_cm, k_cm, pos, cos_cache_, sin_cache_);
        }
    } else
#endif
    {
        ops::rope(q_cm, k_cm, pos, cos_cache_, sin_cache_);
    }
    Eigen::Map<RowMat>(q_vec.data(), n_heads, head_dim) = q_cm;
    Eigen::Map<RowMat>(k_vec.data(), n_kv_heads, head_dim) = k_cm;

    kv_cache_->write(layer_idx, pos, k_vec, v_vec);

#ifdef INFERFLOW_ENABLE_CUDA
    // Mirror K,V into GPU-resident KV cache so attention reads directly from device.
    if (cuda_ctx_) {
        backend::cuda_write_kv_cache(cuda_ctx_, layer_idx, pos,
                                      k_vec.data(), v_vec.data());
    }
#endif

    int seq_len = pos + 1;
    const Eigen::MatrixXf& k_cached = kv_cache_->k(layer_idx);
    const Eigen::MatrixXf& v_cached = kv_cache_->v(layer_idx);

#ifdef INFERFLOW_ENABLE_CUDA
    if (backend_kind_ == backend::BackendKind::CUDA && cuda_ctx_) {
        Eigen::VectorXf attn_flat(dim);
        backend::cuda_attention_cached(cuda_ctx_, layer_idx, seq_len,
                                        q_vec.data(),
                                        n_heads, n_kv_heads, head_dim,
                                        attn_flat);
        linear_backend(attn_flat, lw.attn_output, attn_out
                       , int8_lw ? &int8_lw->attn_output : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                       , cuda_lw ? cuda_lw->attn_output : nullptr
#endif
                       );
        return;
    }
#endif

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

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

    RowMat k_exp(n_heads, seq_len * head_dim);
    RowMat v_exp(n_heads, seq_len * head_dim);
    for (int h = 0; h < n_kv_heads; ++h) {
        for (int g = 0; g < n_groups; ++g) {
            k_exp.row(h * n_groups + g) = k_all.row(h);
            v_exp.row(h * n_groups + g) = v_all.row(h);
        }
    }

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

        ops::softmax(scores);

        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                attn_heads(h, d) += scores(t) * v_exp(h, t * head_dim + d);
            }
        }
    }

    Eigen::Map<Eigen::VectorXf> attn_flat(attn_heads.data(), dim);
    linear_backend(attn_flat, lw.attn_output, attn_out
                   , int8_lw ? &int8_lw->attn_output : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , cuda_lw ? cuda_lw->attn_output : nullptr
#endif
                   );
}

void LlamaEngine::swiglu_ffn_backend(const Eigen::VectorXf& hidden,
                                     const LayerWeights& lw,
                                     Eigen::VectorXf& ffn_out
#ifdef INFERFLOW_ENABLE_CUDA
                                     , const CudaLayerDeviceWeights* cuda_lw
#endif
                                     , const Int8LayerWeights* int8_lw
                                     ) {
    Eigen::VectorXf normed(config_.dim);
    rmsnorm_backend(hidden, lw.ffn_norm, normed);

    Eigen::VectorXf gate(config_.hidden_dim);
    Eigen::VectorXf up(config_.hidden_dim);
    linear_backend(normed, lw.ffn_gate, gate
                   , int8_lw ? &int8_lw->ffn_gate : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , cuda_lw ? cuda_lw->ffn_gate : nullptr
#endif
                   );
    linear_backend(normed, lw.ffn_up, up
                   , int8_lw ? &int8_lw->ffn_up : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , cuda_lw ? cuda_lw->ffn_up : nullptr
#endif
                   );

    for (int i = 0; i < config_.hidden_dim; ++i) {
        gate(i) = gate(i) / (1.0f + std::exp(-gate(i)));
    }
    gate = gate.array() * up.array();
    linear_backend(gate, lw.ffn_down, ffn_out
                   , int8_lw ? &int8_lw->ffn_down : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , cuda_lw ? cuda_lw->ffn_down : nullptr
#endif
                   );
}

Eigen::VectorXf LlamaEngine::forward(int token_id, int pos) {
    // Embedding lookup
    hidden_ = weights_.token_embd.row(token_id);

    // ---- Transformer Layers ----
    for (int i = 0; i < config_.n_layers; ++i) {
        // Attention sub-layer
        attention_forward_backend(hidden_, weights_.layers[i], i, pos, attn_buffer_
#ifdef INFERFLOW_ENABLE_CUDA
                                   , (backend_kind_ == backend::BackendKind::CUDA && cuda_weights_ready_)
                                         ? &cuda_weights_.layers[i]
                                         : nullptr
#endif
                                   , (int8_ready_ && i < static_cast<int>(int8_weights_.layers.size()))
                                         ? &int8_weights_.layers[i]
                                         : nullptr
                                   );
        // Residual
        hidden_ += attn_buffer_;

        // FFN sub-layer
        swiglu_ffn_backend(hidden_, weights_.layers[i], ffn_buffer_
#ifdef INFERFLOW_ENABLE_CUDA
                           , (backend_kind_ == backend::BackendKind::CUDA && cuda_weights_ready_)
                                 ? &cuda_weights_.layers[i]
                                 : nullptr
#endif
                           , (int8_ready_ && i < static_cast<int>(int8_weights_.layers.size()))
                                 ? &int8_weights_.layers[i]
                                 : nullptr
                           );
        // Residual
        hidden_ += ffn_buffer_;
    }

    // Final RMSNorm
    Eigen::VectorXf final_hidden(config_.dim);
    rmsnorm_backend(hidden_, weights_.output_norm, final_hidden);

    // LM Head: logits = final_hidden @ lm_head^T
    Eigen::VectorXf logits(config_.vocab_size);
    linear_backend(final_hidden, weights_.lm_head, logits
                   , int8_ready_ ? &int8_weights_.lm_head : nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                   , (backend_kind_ == backend::BackendKind::CUDA && cuda_weights_ready_)
                         ? cuda_weights_.lm_head
                         : nullptr
#endif
                   );

    return logits;
}

void LlamaEngine::generate_stream(const std::vector<int>& prompt_tokens,
                                   TokenCallback callback,
                                   const GenerationConfig& gen_cfg) {
    if (!loaded_) return;
    if (gen_cfg.max_tokens <= 0) return;

    sampler_->set_config(gen_cfg);
    reset_kv_cache();

    int max_tokens = gen_cfg.max_tokens;
    int prompt_len = static_cast<int>(prompt_tokens.size());
    int eos = config_.eos_token_id;

    // Run prompt tokens through the model (don't sample)
    for (int pos = 0; pos < prompt_len - 1; ++pos) {
        forward(prompt_tokens[pos], pos);
    }

    // Last prompt token → first sampling step
    int next_token;
    if (prompt_len > 0) {
        Eigen::VectorXf logits = forward(prompt_tokens.back(), prompt_len - 1);
        next_token = sampler_->sample(logits);
    } else {
        // Empty prompt: just start from BOS
        Eigen::VectorXf logits = forward(config_.bos_token_id, 0);
        next_token = sampler_->sample(logits);
    }

    int decode_pos = prompt_len > 0 ? prompt_len : 1;

    // Generation loop
    for (int step = 0; step < max_tokens; ++step) {
        // Callback with decoded token
        std::string token_str = std::to_string(next_token);  // placeholder
        callback(token_str, (next_token == eos));

        if (next_token == eos) break;

        Eigen::VectorXf logits = forward(next_token, decode_pos + step);
        next_token = sampler_->sample(logits);
    }
}

std::vector<int> LlamaEngine::generate(const std::vector<int>& prompt_tokens,
                                        const GenerationConfig& gen_cfg) {
    std::vector<int> result;
    if (!loaded_ || gen_cfg.max_tokens <= 0) return result;

    sampler_->set_config(gen_cfg);
    reset_kv_cache();

    int max_tokens = gen_cfg.max_tokens;
    int prompt_len = static_cast<int>(prompt_tokens.size());
    int eos = config_.eos_token_id;

    for (int pos = 0; pos < prompt_len - 1; ++pos) {
        forward(prompt_tokens[pos], pos);
    }

    int next_token;
    if (prompt_len > 0) {
        Eigen::VectorXf logits = forward(prompt_tokens.back(), prompt_len - 1);
        next_token = sampler_->sample(logits);
    } else {
        Eigen::VectorXf logits = forward(config_.bos_token_id, 0);
        next_token = sampler_->sample(logits);
    }

    int decode_pos = prompt_len > 0 ? prompt_len : 1;

    for (int step = 0; step < max_tokens; ++step) {
        result.push_back(next_token);
        if (next_token == eos) break;

        Eigen::VectorXf logits = forward(next_token, decode_pos + step);
        next_token = sampler_->sample(logits);
    }

    return result;
}

std::string LlamaEngine::generate_text(const std::string& prompt,
                                       const GenerationConfig& gen_cfg) {
    std::string result;
    generate_text_stream(prompt,
        [&result](const std::string& token, bool) {
            result += token;
        },
        gen_cfg);
    return result;
}

void LlamaEngine::generate_text_stream(const std::string& prompt,
                                       TokenCallback callback,
                                       const GenerationConfig& gen_cfg) {
    if (!loaded_ || !tokenizer_ || gen_cfg.max_tokens <= 0) return;

    last_metrics_ = InferenceMetrics{};
    const auto total_start = Clock::now();

    sampler_->set_config(gen_cfg);
    reset_kv_cache();

    auto prompt_tokens = tokenizer_->encode(prompt, true, false);
    std::vector<int> output_tokens;
    std::string emitted_text;

    int prompt_len = static_cast<int>(prompt_tokens.size());
    int eos = config_.eos_token_id;
    last_metrics_.prompt_tokens = prompt_len;

    const auto prefill_start = Clock::now();
    for (int pos = 0; pos < prompt_len - 1; ++pos) {
        forward(prompt_tokens[pos], pos);
    }

    int next_token;
    if (prompt_len > 0) {
        Eigen::VectorXf logits = forward(prompt_tokens.back(), prompt_len - 1);
        next_token = sampler_->sample(logits);
    } else {
        Eigen::VectorXf logits = forward(config_.bos_token_id, 0);
        next_token = sampler_->sample(logits);
    }
    const auto prefill_end = Clock::now();
    last_metrics_.prefill_ms = elapsed_ms(prefill_start, prefill_end);
    last_metrics_.first_token_ms = elapsed_ms(total_start, prefill_end);

    int decode_pos = prompt_len > 0 ? prompt_len : 1;

    const auto decode_start = Clock::now();
    for (int step = 0; step < gen_cfg.max_tokens; ++step) {
        output_tokens.push_back(next_token);
        last_metrics_.generated_tokens = static_cast<int32_t>(output_tokens.size());
        std::string decoded_text = tokenizer_->decode(output_tokens);

        bool stopped = false;
        std::string visible_text = apply_stop_strings(decoded_text, gen_cfg.stop_strings, stopped);
        if (!stopped) {
            visible_text = hold_stop_prefix_suffix(visible_text, gen_cfg.stop_strings);
        }
        bool finished = stopped || next_token == eos;

        if (visible_text.size() > emitted_text.size()) {
            callback(visible_text.substr(emitted_text.size()), finished);
            emitted_text = std::move(visible_text);
        } else if (finished) {
            callback("", true);
        }

        if (finished) break;

        Eigen::VectorXf logits = forward(next_token, decode_pos + step);
        next_token = sampler_->sample(logits);
    }

    const auto decode_end = Clock::now();
    last_metrics_.decode_ms = elapsed_ms(decode_start, decode_end);
    last_metrics_.total_ms = elapsed_ms(total_start, decode_end);
    if (last_metrics_.decode_ms > 0.0 && last_metrics_.generated_tokens > 0) {
        last_metrics_.decode_tokens_per_second =
            static_cast<double>(last_metrics_.generated_tokens) * 1000.0 / last_metrics_.decode_ms;
    }
}

void LlamaEngine::reset_kv_cache() {
    if (kv_cache_) kv_cache_->clear();
#ifdef INFERFLOW_ENABLE_CUDA
    if (cuda_ctx_) backend::cuda_context_reset(cuda_ctx_);
#endif
}

}  // namespace llama
