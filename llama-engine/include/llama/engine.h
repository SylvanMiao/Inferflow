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
#include "config.h"
#include "kv_cache.h"
#include "sampler.h"
#include "tokenizer.h"
#include "weights.h"

namespace llama {

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
    void init_buffers();
    void precompute_rope();

    ModelConfig config_;
    LlamaWeights weights_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<Tokenizer> tokenizer_;
    bool loaded_ = false;

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
};

}  // namespace llama
