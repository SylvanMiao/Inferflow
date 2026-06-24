#pragma once
/**
 * LlamaEngine — the main inference interface.
 *
 * Maps 1:1 to Python LlamaModel + LlamaForCausalLM.
 *
 * Usage:
 *   LlamaEngine engine;
 *   engine.load("model.gguf", "tokenizer.model");
 *   auto tokens = engine.generate("Hello world", GenerationConfig{});
 */

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "backend/backend.h"
#include "config.h"
#include "kv_cache.h"
#include "quantization.h"
#include "sampler.h"
#include "tokenizer.h"
#include "weights.h"

#ifdef INFERFLOW_ENABLE_CUDA
namespace llama { namespace backend { struct CudaForwardContext; struct CudaMatrix; } }
#endif

namespace llama {

struct InferenceMetrics {
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double first_token_ms = 0.0;
    double total_ms = 0.0;
    double decode_tokens_per_second = 0.0;
};

class LlamaEngine {
public:
    LlamaEngine();
    ~LlamaEngine();

    // ---- Lifecycle ----

    /**
     * Load model weights and tokenizer.
     * Currently supports a simple binary format (see docs).
     * GGUF support planned for Phase 2.
     */
    bool load(const std::string& model_path);
    bool load_tokenizer(const std::string& tokenizer_path);

    bool is_loaded() const { return loaded_; }
    bool has_tokenizer() const { return tokenizer_ != nullptr; }

    // ---- Model info ----

    const ModelConfig& config() const { return config_; }
    const LlamaWeights& weights() const { return weights_; }
    const InferenceMetrics& last_metrics() const { return last_metrics_; }
    backend::BackendKind backend_kind() const { return backend_kind_; }
    bool set_backend(backend::BackendKind backend);
    bool quantize_int8(int group_size = 128);
    bool use_int8_weights(bool enabled);
    bool int8_enabled() const { return int8_enabled_; }
    bool has_int8_weights() const { return int8_ready_; }

    // ---- Inference ----

    /** Callback type: (token_string, is_end_of_sequence) */
    using TokenCallback = std::function<void(const std::string& token, bool eos)>;

    /**
     * Generate tokens with streaming callback.
     * Each token is passed to the callback as it's produced.
     */
    void generate_stream(const std::vector<int>& prompt_tokens,
                         TokenCallback callback,
                         const GenerationConfig& gen_cfg = {});

    /**
     * Generate tokens synchronously. Returns all generated token ids.
     */
    std::vector<int> generate(const std::vector<int>& prompt_tokens,
                              const GenerationConfig& gen_cfg = {});

    /**
     * Generate text from a prompt string. Requires load_tokenizer().
     */
    std::string generate_text(const std::string& prompt,
                              const GenerationConfig& gen_cfg = {});

    /**
     * Generate text with per-token callback. Requires load_tokenizer().
     */
    void generate_text_stream(const std::string& prompt,
                              TokenCallback callback,
                              const GenerationConfig& gen_cfg = {});

    /**
     * Single forward step: input token → output logits.
     * Updates internal KV cache at position `pos`.
     */
    Eigen::VectorXf forward(int token_id, int pos);

    /**
     * Get/clear the KV cache (useful for multi-turn conversations).
     */
    KVCache& kv_cache() { return *kv_cache_; }
    const KVCache& kv_cache() const { return *kv_cache_; }
    void reset_kv_cache();

private:
#ifdef INFERFLOW_ENABLE_CUDA
    struct CudaLayerDeviceWeights;
#endif
    struct Int8LayerWeights;

    void init_buffers();
    void precompute_rope();
    void attention_forward_backend(const Eigen::VectorXf& hidden,
                                   const LayerWeights& lw,
                                   int layer_idx,
                                   int pos,
                                   Eigen::VectorXf& attn_out
#ifdef INFERFLOW_ENABLE_CUDA
                                   , const CudaLayerDeviceWeights* cuda_lw = nullptr
#endif
                                   , const Int8LayerWeights* int8_lw = nullptr
                                   );
    void swiglu_ffn_backend(const Eigen::VectorXf& hidden,
                            const LayerWeights& lw,
                            Eigen::VectorXf& ffn_out
#ifdef INFERFLOW_ENABLE_CUDA
                            , const CudaLayerDeviceWeights* cuda_lw = nullptr
#endif
                            , const Int8LayerWeights* int8_lw = nullptr
                            );
    void rmsnorm_backend(const Eigen::VectorXf& input,
                         const Eigen::VectorXf& weight,
                         Eigen::VectorXf& output);
    void linear_backend(const Eigen::VectorXf& input,
                        const Eigen::MatrixXf& weight,
                        Eigen::VectorXf& output,
                        const quant::QuantizedMatrixInt8* int8_weight = nullptr
#ifdef INFERFLOW_ENABLE_CUDA
                        , const backend::CudaMatrix* cuda_weight = nullptr
#endif
                        );
#ifdef INFERFLOW_ENABLE_CUDA
    void ensure_cuda_weights();
    void destroy_cuda_weights();
#endif

    ModelConfig config_;
    LlamaWeights weights_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<Tokenizer> tokenizer_;
    bool loaded_ = false;
    backend::BackendKind backend_kind_ = backend::BackendKind::CPU;
    bool int8_ready_ = false;
    bool int8_enabled_ = false;
    int int8_group_size_ = 128;
    InferenceMetrics last_metrics_;

    // RoPE precomputed cache
    Eigen::MatrixXf cos_cache_;  // [max_seq_len, head_dim/2]
    Eigen::MatrixXf sin_cache_;

    // Intermediate buffers (allocated once to avoid per-step mallocs)
    Eigen::VectorXf hidden_;       // [dim]
    Eigen::VectorXf q_buffer_;     // [n_heads * head_dim]
    Eigen::VectorXf k_buffer_;     // [kv_dim]
    Eigen::VectorXf v_buffer_;     // [kv_dim]
    Eigen::VectorXf attn_buffer_;  // [dim]
    Eigen::VectorXf ffn_buffer_;   // [dim]

#ifdef INFERFLOW_ENABLE_CUDA
    struct CudaLayerDeviceWeights {
        backend::CudaMatrix* attn_q = nullptr;
        backend::CudaMatrix* attn_k = nullptr;
        backend::CudaMatrix* attn_v = nullptr;
        backend::CudaMatrix* attn_output = nullptr;
        backend::CudaMatrix* ffn_gate = nullptr;
        backend::CudaMatrix* ffn_up = nullptr;
        backend::CudaMatrix* ffn_down = nullptr;
    };

    struct CudaDeviceWeights {
        std::vector<CudaLayerDeviceWeights> layers;
        backend::CudaMatrix* lm_head = nullptr;
    };

    backend::CudaForwardContext* cuda_ctx_ = nullptr;
    CudaDeviceWeights cuda_weights_;
    bool cuda_weights_ready_ = false;
#endif

    struct Int8LayerWeights {
        quant::QuantizedMatrixInt8 attn_q;
        quant::QuantizedMatrixInt8 attn_k;
        quant::QuantizedMatrixInt8 attn_v;
        quant::QuantizedMatrixInt8 attn_output;
        quant::QuantizedMatrixInt8 ffn_gate;
        quant::QuantizedMatrixInt8 ffn_up;
        quant::QuantizedMatrixInt8 ffn_down;
    };

    struct Int8Weights {
        quant::QuantizedMatrixInt8 lm_head;
        std::vector<Int8LayerWeights> layers;
    };

    Int8Weights int8_weights_;
};

}  // namespace llama
