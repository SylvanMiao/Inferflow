#pragma once

#include "session/session_manager.h"

#include <string>
#include <vector>

namespace inference
{
  struct ChatPrompt
  {
    std::string text;
    std::size_t message_count{0};
  };

  class ChatTemplate
  {
  public:
    ChatPrompt build_prompt(const std::vector<ChatMessage> &history,
                            const std::string &user_text) const;

    std::string trim(const std::string &text) const;
  };
}
