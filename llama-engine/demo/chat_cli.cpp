#include "llama/engine.h"

#include <cstdlib>
#include <iostream>
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
    std::cout << "Usage: " << program << " [model.bin] [tokenizer.model] [max_tokens]\n"
              << "Environment overrides: INFERFLOW_MODEL_PATH, INFERFLOW_TOKENIZER_PATH\n";
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

    std::cout << "InferFlow CLI ready. Type /exit to quit, /clear to reset history.\n";

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
