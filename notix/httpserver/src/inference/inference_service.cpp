#include "inference/inference_service.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace inference
{
  namespace
  {
    std::string getenv_or_default(const char *key, const std::string &fallback)
    {
      const char *value = std::getenv(key);
      if (value == nullptr || *value == '\0')
      {
        return fallback;
      }
      return value;
    }

    std::string trim_copy(const std::string &text)
    {
      const auto first = text.find_first_not_of(" \t\r\n");
      if (first == std::string::npos)
      {
        return {};
      }

      const auto last = text.find_last_not_of(" \t\r\n");
      return text.substr(first, last - first + 1);
    }
  }

  InferenceService &InferenceService::instance()
  {
    static InferenceService service;
    return service;
  }

  llama::LlamaEngine &InferenceService::engine()
  {
    std::call_once(init_flag_, [this]()
                   {
                     const auto model_path = getenv_or_default(
                         "INFERFLOW_MODEL_PATH", INFERFLOW_DEFAULT_MODEL_PATH);
                     const auto tokenizer_path = getenv_or_default(
                         "INFERFLOW_TOKENIZER_PATH", INFERFLOW_DEFAULT_TOKENIZER_PATH);

                     ready_ = engine_.load(model_path) && engine_.load_tokenizer(tokenizer_path);
                     if (!ready_)
                     {
                       std::cerr << "Failed to initialize inference engine. model="
                                 << model_path << " tokenizer=" << tokenizer_path << std::endl;
                     }
                   });

    if (!ready_)
    {
      throw std::runtime_error("inference engine is not ready");
    }
    return engine_;
  }

  GenerateResponse InferenceService::generate(const GenerateRequest &request)
  {
    if (request.max_tokens <= 0 || request.max_tokens > 256)
    {
      throw std::invalid_argument("max_tokens must be in range [1, 256]");
    }

    llama::GenerationConfig gen_cfg;
    gen_cfg.temperature = request.temperature;
    gen_cfg.max_tokens = request.max_tokens;
    gen_cfg.stop_strings = {"\nUser:", "\nAssistant:", "\n用户:", "\n助手:"};

    GenerateResponse response;
    auto &llama_engine = engine();
    response.text = trim_copy(llama_engine.generate_text(request.prompt, gen_cfg));
    response.model_name = "tinyllama";
    response.max_tokens = gen_cfg.max_tokens;
    response.metrics = llama_engine.last_metrics();
    return response;
  }

  bool InferenceService::reset_cache_if_loaded()
  {
    if (!ready_)
    {
      return false;
    }

    engine_.reset_kv_cache();
    return true;
  }
}
