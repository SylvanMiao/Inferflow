#include "router_rules.h"
#include <vector>


namespace router
{  
  namespace
  {
    std::vector<std::string> split_segments(const std::string &path)
    {
      std::vector<std::string> segments;
      std::size_t start = 0;

      while (start < path.size())
      {
        while (start < path.size() && path[start] == '/')
        {
          ++start;
        }

        if (start >= path.size())
        {
          break;
        }

        auto end = path.find('/', start);
        if (end == std::string::npos)
        {
          segments.emplace_back(path.substr(start));
          break;
        }

        segments.emplace_back(path.substr(start, end - start));
        start = end + 1;
      }

      return segments;
    }

    bool is_placeholder(const std::string &segment)
    {
      return segment.size() > 2 && segment.front() == '{' && segment.back() == '}';
    }

    std::string placeholder_name(const std::string &segment)
    {
      return segment.substr(1, segment.size() - 2);
    }
  }

  std::string normalize_target(std::string_view raw_target)
  {
    std::string path(raw_target);
    auto query_pos = path.find('?');
    if (query_pos != std::string::npos)
    {
      path = path.substr(0, query_pos);
    }

    auto fragment_pos = path.find('#');
    if (fragment_pos != std::string::npos)
    {
      path = path.substr(0, fragment_pos);
    }

    if (path.empty())
    {
      return "/";
    }

    while (path.size() > 1 && path.back() == '/')
    {
      path.pop_back();
    }

    return path;
  }

  static_route match_static_route(http::verb method, std::string_view target)
  {
    const auto normalized_target = normalize_target(target);

    if (method == http::verb::get)
    {
      if (normalized_target == CountRoute)
      {
        return static_route::count;
      }
      if (normalized_target == TimeRoute)
      {
        return static_route::time;
      }
      if (normalized_target == LoginRoute)
      {
        return static_route::login_page;
      }
      if (normalized_target == RegisterRoute)
      {
        return static_route::register_page;
      }
      if (normalized_target == AppRoute)
      {
        return static_route::app_page;
      }
      if (normalized_target == ChatHistoryRoute)
      {
        return static_route::chat_history;
      }
      if (normalized_target == MeRoute)
      {
        return static_route::me;
      }
      return static_route::none;
    }

    if (method == http::verb::post)
    {
      if (normalized_target == EmailRoute)
      {
        return static_route::email;
      }
      if (normalized_target == LoginRoute)
      {
        return static_route::login_submit;
      }
      if (normalized_target == RegisterRoute)
      {
        return static_route::register_submit;
      }
      if (normalized_target == ChatEchoRoute)
      {
        return static_route::chat_echo;
      }
      if (normalized_target == ChatCompletionsRoute)
      {
        return static_route::chat_completions;
      }
      if (normalized_target == ImageProcessRoute)
      {
        return static_route::image_process;
      }
      return static_route::none;
    }

    return static_route::none;
  }

  bool match_dynamic_route(const dynamic_route &route,
                           http::verb method,
                           std::string_view target,
                           route_params &params)
  {
    params.clear();

    if (route.method != method)
    {
      return false;
    }

    const auto normalized_target = normalize_target(target);
    const auto normalized_pattern = normalize_target(route.pattern);

    const auto target_segments = split_segments(normalized_target);
    const auto pattern_segments = split_segments(normalized_pattern);

    if (target_segments.size() != pattern_segments.size())
    {
      return false;
    }

    for (std::size_t i = 0; i < pattern_segments.size(); ++i)
    {
      const auto &pattern_part = pattern_segments[i];
      const auto &target_part = target_segments[i];

      if (is_placeholder(pattern_part))
      {
        const auto key = placeholder_name(pattern_part);
        if (key.empty())
        {
          return false;
        }
        params[key] = target_part;
        continue;
      }

      if (pattern_part != target_part)
      {
        return false;
      }
    }

    return true;
  }

}
