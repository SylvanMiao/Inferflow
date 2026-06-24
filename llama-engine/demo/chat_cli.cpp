#include "llama/engine.h"
#include "llama/backend/backend.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef INFERFLOW_DEFAULT_MODEL_PATH
#define INFERFLOW_DEFAULT_MODEL_PATH "../test/tinyllama.bin"
#endif

#ifndef INFERFLOW_DEFAULT_TOKENIZER_PATH
#define INFERFLOW_DEFAULT_TOKENIZER_PATH "../../../models/tokenizer.model"
#endif

namespace {

std::string getenv_or_default(const char* key, const std::string& fallback) {
    const char* value = std::getenv(key);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return value;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [model.bin] [tokenizer.model] [max_tokens] [backend] [precision]\n"
              << "  backend: cpu or cuda. Default: INFERFLOW_BACKEND or cpu.\n"
              << "  precision: fp32 or int8. Default: INFERFLOW_PRECISION / INFERFLOW_INT8 or fp32.\n"
              << "Environment overrides: INFERFLOW_MODEL_PATH, INFERFLOW_TOKENIZER_PATH, INFERFLOW_BACKEND, INFERFLOW_PRECISION, INFERFLOW_INT8\n";
}

void print_metrics(const llama::InferenceMetrics& metrics) {
    std::cout << "[metrics] prompt_tokens=" << metrics.prompt_tokens
              << " generated_tokens=" << metrics.generated_tokens
              << " prefill_ms=" << metrics.prefill_ms
              << " first_token_ms=" << metrics.first_token_ms
              << " decode_ms=" << metrics.decode_ms
              << " total_ms=" << metrics.total_ms
              << " decode_tok_s=" << metrics.decode_tokens_per_second
              << std::endl;
}

std::string build_chat_prompt(const std::vector<std::pair<std::string, std::string>>& history,
                              const std::string& user_input) {
    std::string prompt;
    prompt += "<|system|>\n";
    prompt += "You are a helpful assistant. Answer concisely.\n";
    prompt += "</s>\n";

    for (const auto& turn : history) {
        prompt += "<|user|>\n";
        prompt += turn.first;
        prompt += "\n</s>\n";
        prompt += "<|assistant|>\n";
        prompt += turn.second;
        prompt += "\n</s>\n";
    }

    prompt += "<|user|>\n";
    prompt += user_input;
    prompt += "\n</s>\n";
    prompt += "<|assistant|>\n";
    return prompt;
}

llama::backend::BackendKind parse_backend(const std::string& value) {
    if (value == "cpu" || value == "CPU") {
        return llama::backend::BackendKind::CPU;
    }
    if (value == "cuda" || value == "CUDA") {
        return llama::backend::BackendKind::CUDA;
    }
    throw std::runtime_error("backend must be cpu or cuda");
}

bool parse_int8_enabled(const std::string& value) {
    if (value == "int8" || value == "INT8" ||
        value == "1" || value == "true" || value == "TRUE" ||
        value == "on" || value == "ON" || value == "yes" || value == "YES") {
        return true;
    }
    if (value == "fp32" || value == "FP32" ||
        value == "0" || value == "false" || value == "FALSE" ||
        value == "off" || value == "OFF" || value == "no" || value == "NO") {
        return false;
    }
    throw std::runtime_error("precision must be fp32 or int8");
}

std::string default_precision() {
    const char* precision = std::getenv("INFERFLOW_PRECISION");
    if (precision != nullptr && *precision != '\0') {
        return precision;
    }
    const char* int8 = std::getenv("INFERFLOW_INT8");
    if (int8 != nullptr && *int8 != '\0') {
        return int8;
    }
    return "fp32";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    std::string model_path = argc > 1
        ? argv[1]
        : getenv_or_default("INFERFLOW_MODEL_PATH", INFERFLOW_DEFAULT_MODEL_PATH);
    std::string tokenizer_path = argc > 2
        ? argv[2]
        : getenv_or_default("INFERFLOW_TOKENIZER_PATH", INFERFLOW_DEFAULT_TOKENIZER_PATH);
    std::string backend_name = argc > 4
        ? argv[4]
        : getenv_or_default("INFERFLOW_BACKEND", "cpu");
    std::string precision_name = argc > 5 ? argv[5] : default_precision();

    if (backend_name == "int8" || backend_name == "INT8" ||
        backend_name == "fp32" || backend_name == "FP32") {
        precision_name = backend_name;
        backend_name = getenv_or_default("INFERFLOW_BACKEND", "cpu");
    }

    llama::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = argc > 3 ? std::stoi(argv[3]) : 64;
    gen_cfg.temperature = 0.0f;
    gen_cfg.top_k = 0;
    gen_cfg.top_p = 1.0f;
    gen_cfg.stop_strings = {"</s>", "<|user|>", "<|assistant|>", "<|system|>"};

    llama::LlamaEngine engine;
    std::cout << "Loading model: " << model_path << std::endl;
    if (!engine.load(model_path)) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    std::cout << "Loading tokenizer: " << tokenizer_path << std::endl;
    if (!engine.load_tokenizer(tokenizer_path)) {
        std::cerr << "Failed to load tokenizer." << std::endl;
        return 1;
    }

    llama::backend::BackendKind backend;
    bool int8_enabled = false;
    try {
        backend = parse_backend(backend_name);
        int8_enabled = parse_int8_enabled(precision_name);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (!engine.set_backend(backend)) {
        std::cerr << "Failed to enable backend: " << backend_name
                  << ". Was llama-engine built with --cuda and is a CUDA device visible?"
                  << std::endl;
        return 1;
    }

    if (int8_enabled) {
        if (engine.backend_kind() != llama::backend::BackendKind::CPU) {
            std::cerr << "Int8 full-forward path is currently CPU-only. Use backend=cpu for int8 CLI inference."
                      << std::endl;
            return 1;
        }
        if (!engine.quantize_int8(128) || !engine.use_int8_weights(true)) {
            std::cerr << "Failed to enable Int8 weights." << std::endl;
            return 1;
        }
    }

    std::cout << "InferFlow CLI ready. backend="
              << llama::backend::backend_name(engine.backend_kind())
              << " precision=" << (engine.int8_enabled() ? "int8" : "fp32")
              << ". Type /exit to quit, /clear to reset history.\n";

    std::string line;
    std::vector<std::pair<std::string, std::string>> history;
    while (true) {
        std::cout << "\nyou> " << std::flush;
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line == "/exit" || line == "/quit") {
            break;
        }
        if (line == "/clear") {
            history.clear();
            std::cout << "history cleared.\n";
            continue;
        }
        if (line.empty()) {
            continue;
        }

        const std::string prompt = build_chat_prompt(history, line);
        std::string reply;

        std::cout << "bot> " << std::flush;
        engine.generate_text_stream(
            prompt,
            [&reply](const std::string& token, bool) {
                reply += token;
                std::cout << token << std::flush;
            },
            gen_cfg);
        std::cout << std::endl;
        print_metrics(engine.last_metrics());

        history.push_back({line, reply});
    }

    return 0;
}
