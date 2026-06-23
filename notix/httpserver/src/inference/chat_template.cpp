#include "inference/chat_template.h"

#include <sstream>

namespace inference
{
  std::string ChatTemplate::trim(const std::string &text) const
  {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
      return {};
    }

    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
  }

  ChatPrompt ChatTemplate::build_prompt(const std::vector<ChatMessage> &history,
                                        const std::string &user_text) const
  {
    ChatPrompt result;
    std::ostringstream prompt;
    prompt << "You are Inferflow, a concise and helpful assistant.\n"
           << "Continue the conversation naturally. Answer the latest user message.\n\n";

    for (const auto &item : history)
    {
      const auto content = trim(item.content);
      if (content.empty())
      {
        continue;
      }

      if (item.role == "user")
      {
        prompt << "User: " << content << "\n";
        ++result.message_count;
      }
      else if (item.role == "assistant")
      {
        prompt << "Assistant: " << content << "\n";
        ++result.message_count;
      }
    }

    prompt << "User: " << trim(user_text) << "\n"
           << "Assistant:";
    ++result.message_count;
    result.text = prompt.str();
    return result;
  }
}
