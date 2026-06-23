#pragma once
/**
 * LLaMA Model Configuration
 * ==================================
 *
 * This header defines the model hyperparameters and generation settings.
 * It maps 1:1 to the Python LlamaConfig dataclass in python_prototype/llama_forward.py.
 *
 * All values default to LLaMA3 8B. For other models, construct with appropriate values.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace llama {

struct ModelConfig {
    // ---- Transformer dimensions ----
    int32_t dim         = 4096;    // Hidden dimension (d_model)
    int32_t n_layers    = 32;      // Number of decoder layers
    int32_t n_heads     = 32;      // Query attention heads
    int32_t n_kv_heads  = 8;       // Key/Value heads (GQA: n_heads/n_kv_heads groups)
    int32_t head_dim    = 128;     // Dimension per head (= dim / n_heads)
    int32_t hidden_dim  = 14336;   // FFN intermediate dimension
    int32_t vocab_size  = 128256;  // Vocabulary size
    int32_t max_seq_len = 8192;    // Maximum sequence length
    int32_t kv_dim      = 1024;    // n_kv_heads * head_dim (derived; set at init)

    // ---- Normalization ----
    float norm_eps      = 1e-5f;   // RMSNorm epsilon
    float rope_theta    = 10000.0f;  // RoPE base frequency (LLaMA/Llama2 default; LLaMA3 = 500000)

    // ---- Special tokens ----
    int32_t bos_token_id = 128000;  // Beginning of sequence
    int32_t eos_token_id = 128001;  // End of sequence

    // ---- Weight sharing ----
    bool is_shared_weight = true;   // Token embedding shared with LM head

    /**
     * Derived values set at initialization time.
     */
    int32_t n_groups() const { return n_heads / n_kv_heads; }

    /**
     * Validate configuration consistency.
     * Returns empty string on success, error message on failure.
     */
    std::string validate() const {
        if (head_dim * n_heads != dim)
            return "head_dim * n_heads != dim";
        if (n_heads % n_kv_heads != 0)
            return "n_heads must be divisible by n_kv_heads";
        if (kv_dim != n_kv_heads * head_dim)
            return "kv_dim != n_kv_heads * head_dim";
        return {};
    }

    /**
     * TinyLlama 1.1B preset — useful for testing.
     */
    static ModelConfig tiny_llama() {
        ModelConfig c;
        c.dim = 2048;
        c.n_layers = 22;
        c.n_heads = 32;
        c.n_kv_heads = 4;
        c.head_dim = 64;
        c.hidden_dim = 5632;
        c.vocab_size = 32000;
        c.max_seq_len = 2048;
        c.rope_theta = 10000.0f;
        c.kv_dim = c.n_kv_heads * c.head_dim;
        c.bos_token_id = 1;
        c.eos_token_id = 2;
        return c;
    }
};


struct GenerationConfig {
    int32_t max_tokens            = 256;
    float   temperature           = 0.7f;
    float   top_p                 = 0.9f;
    int32_t top_k                 = 50;
    float   min_p                 = 0.05f;
    float   repetition_penalty    = 1.1f;
    int32_t seed                  = -1;  // -1 = random

    std::vector<std::string> stop_strings;  // Stop generation on these substrings
};

}  // namespace llama
