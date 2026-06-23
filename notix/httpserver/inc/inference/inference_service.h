#pragma once

#include "llama/config.h"
#include "llama/engine.h"

#include <mutex>
#include <string>

namespace inference
{
  struct GenerateRequest
  {
    std::string prompt;
    int max_tokens{128};
    float temperature{0.0f};
  };

  struct GenerateResponse
  {
    std::string text;
    std::string model_name;
    int max_tokens{0};
    llama::InferenceMetrics metrics;
  };

  class InferenceService
  {
  public:
    static InferenceService &instance();

    GenerateResponse generate(const GenerateRequest &request);
    bool reset_cache_if_loaded();

  private:
    InferenceService() = default;

    llama::LlamaEngine &engine();

    llama::LlamaEngine engine_;
    std::once_flag init_flag_;
    bool ready_{false};
  };
}
