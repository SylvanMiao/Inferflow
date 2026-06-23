/**
 * LlamaEngine implementation — the main inference loop.
 *
 * Maps 1:1 to Python LlamaModel.forward() + LlamaForCausalLM.generate().
 */

#include "llama/engine.h"
#include "llama/ops.h"
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
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
LlamaEngine::~LlamaEngine() = default;

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

    loaded_ = true;
    std::cout << "Model loaded successfully." << std::endl;
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

Eigen::VectorXf LlamaEngine::forward(int token_id, int pos) {
    // Embedding lookup
    hidden_ = weights_.token_embd.row(token_id);

    // ---- Transformer Layers ----
    for (int i = 0; i < config_.n_layers; ++i) {
        // Attention sub-layer
        ops::attention_forward(hidden_, weights_.layers[i], *kv_cache_,
                               i, pos, config_, cos_cache_, sin_cache_,
                               attn_buffer_);
        // Residual
        hidden_ += attn_buffer_;

        // FFN sub-layer
        ops::swiglu_ffn(hidden_, weights_.layers[i], config_, ffn_buffer_);
        // Residual
        hidden_ += ffn_buffer_;
    }

    // Final RMSNorm
    Eigen::VectorXf final_hidden(config_.dim);
    ops::rmsnorm(hidden_, weights_.output_norm, config_.norm_eps, final_hidden);

    // LM Head: logits = final_hidden @ lm_head^T
    Eigen::VectorXf logits(config_.vocab_size);
    ops::linear(final_hidden, weights_.lm_head, logits);

    return logits;
}

void LlamaEngine::generate_stream(const std::vector<int>& prompt_tokens,
                                   TokenCallback callback,
                                   const GenerationConfig& gen_cfg) {
    if (!loaded_) return;
    if (gen_cfg.max_tokens <= 0) return;

    sampler_->set_config(gen_cfg);
    kv_cache_->clear();

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
    kv_cache_->clear();

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
    kv_cache_->clear();

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
}

}  // namespace llama
