#include "beast_http.h"
#include "program_state_test.h"
#include "router.h"
#include "session/session_manager.h"
#include "llama/engine.h"

#include <mutex>

http_connection::http_connection(tcp::socket socket) : _socket(std::move(socket))
{
  _request_parser.body_limit(MaxBodySize);
  setup_middlewares();
  register_route_handlers();
}

void http_connection::start()
{
  read_request();
  check_deadline();
}

void http_connection::read_request()
{
  auto self = shared_from_this();
  _request_parser.body_limit(MaxBodySize);
  http::async_read(_socket, _buffer, _request_parser,
          [self](beast::error_code ec, std::size_t bytes_transferred)
          {
            boost::ignore_unused(bytes_transferred);
            if (!ec)
            {
              self->_request = self->_request_parser.release();
              self->process_request();
              return;
            }

            if (ec == http::error::body_limit)
            {
              self->send_simple_error(http::status::payload_too_large, 1400,
                                      "request body too large");
              return;
            }

            self->log_transport_error("read_request", ec.message());
            boost::system::error_code close_ec;
            self->_socket.close(close_ec);
          });
}

void http_connection::check_deadline()
{
  auto self = shared_from_this();
  _timer.async_wait([self](boost::system::error_code ec)
                    {
      if (!ec){
        self->_socket.close(ec);
      } });
}

void http_connection::process_request()
{
  _request_start = std::chrono::steady_clock::now();
  _last_error_reason.clear();
  _response.body().clear();

  _response.version(_request.version());
  _response.keep_alive(false);
  _response.set(http::field::server, "Beast");

  const auto method = _request.method();
  const auto route_result = router::Router::default_router().match(method, _request.target());

  middleware::HttpContext ctx(_request, _response, route_result, _last_error_reason);
  try {
    _middleware_pipeline.execute(ctx, [this](middleware::HttpContext &ctx)
                               { dispatch_route(ctx); });
  } catch (const std::exception& e) {
    std::cerr << "Handler Exception: " << e.what() << std::endl;
    send_error(http::status::internal_server_error, 1500, e.what());
  }
  write_response();
}

void http_connection::dispatch_route(middleware::HttpContext &ctx)
{
  const auto &route_result = ctx.route_match;
  // 先根据 Router 的匹配结果选择 handler，避免在这里硬编码 target 分支。
  if (route_result.kind == router::Router::match_kind::static_match)
  {
    auto it = _static_handlers.find(route_result.static_target);
    if (it != _static_handlers.end())
    {
      it->second(ctx);
      return;
    }

    send_error(http::status::not_found, 1404, "route not found");
    return;
  }

  if (route_result.kind == router::Router::match_kind::dynamic_match)
  {
    auto it = _dynamic_handlers.find(route_result.dynamic_target);
    if (it != _dynamic_handlers.end())
    {
      it->second(ctx, route_result.params);
      return;
    }

    send_error(http::status::not_found, 1404, "route not found");
    return;
  }

  send_error(http::status::not_found, 1404, "route not found");
}

void http_connection::setup_middlewares()
{
  middleware::register_default_middlewares(*this, _middleware_pipeline);
}

void http_connection::register_route_handlers()
{
  // 静态路由 handler 注册
  _static_handlers[router::static_route::count] =
      [this](middleware::HttpContext &ctx)
  {
    handle_count(ctx);
  };

  _static_handlers[router::static_route::time] =
      [this](middleware::HttpContext &ctx)
  {
    handle_time(ctx);
  };

  _static_handlers[router::static_route::email] =
      [this](middleware::HttpContext &ctx)
  {
    handle_email(ctx);
  };

  _static_handlers[router::static_route::login_page] =
      [this](middleware::HttpContext &ctx)
  {
    handle_login_page(ctx);
  };

  _static_handlers[router::static_route::login_submit] =
      [this](middleware::HttpContext &ctx)
  {
    handle_login_submit(ctx);
  };

  _static_handlers[router::static_route::register_page] =
      [this](middleware::HttpContext &ctx)
  {
    handle_register_page(ctx);
  };

  _static_handlers[router::static_route::register_submit] =
      [this](middleware::HttpContext &ctx)
  {
    handle_register_submit(ctx);
  };

  _static_handlers[router::static_route::app_page] =
      [this](middleware::HttpContext &ctx)
  {
    handle_app_page(ctx);
  };

  _static_handlers[router::static_route::chat_echo] =
      [this](middleware::HttpContext &ctx)
  {
    handle_chat_echo(ctx);
  };

  _static_handlers[router::static_route::chat_completions] =
      [this](middleware::HttpContext &ctx)
  {
    handle_chat_completions(ctx);
  };

  _static_handlers[router::static_route::chat_history] =
      [this](middleware::HttpContext &ctx)
  {
    handle_chat_history(ctx);
  };

  _static_handlers[router::static_route::image_process] =
      [this](middleware::HttpContext &ctx)
  {
    handle_image_process(ctx);
  };

  _static_handlers[router::static_route::me] =
      [this](middleware::HttpContext &ctx)
  {
    handle_me(ctx);
  };

  // 动态路由 handler 注册
  _dynamic_handlers[router::dynamic_route_id::time_by_zone] =
      [this](middleware::HttpContext &ctx, const router::route_params &params)
  {
    handle_time_by_zone(ctx, params);
  };

  _dynamic_handlers[router::dynamic_route_id::email_by_source] =
      [this](middleware::HttpContext &ctx, const router::route_params &params)
  {
    handle_email_by_source(ctx, params);
  };
}

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

  llama::LlamaEngine &inference_engine()
  {
    static llama::LlamaEngine engine;
    static std::once_flag init_flag;
    static bool ready = false;

    std::call_once(init_flag, []()
                   {
                     const auto model_path = getenv_or_default(
                         "INFERFLOW_MODEL_PATH", INFERFLOW_DEFAULT_MODEL_PATH);
                     const auto tokenizer_path = getenv_or_default(
                         "INFERFLOW_TOKENIZER_PATH", INFERFLOW_DEFAULT_TOKENIZER_PATH);

                     ready = engine.load(model_path) && engine.load_tokenizer(tokenizer_path);
                     if (!ready)
                     {
                       std::cerr << "Failed to initialize inference engine. model="
                                 << model_path << " tokenizer=" << tokenizer_path << std::endl;
                     }
                   });

    if (!ready)
    {
      throw std::runtime_error("inference engine is not ready");
    }
    return engine;
  }

  std::string read_cookie_value(const http::request<http::dynamic_body> &request,
                                const std::string &key)
  {
    if (!request.base().count(http::field::cookie))
    {
      return {};
    }

    std::string cookies(request[http::field::cookie]);
    std::size_t start = 0;
    while (start < cookies.size())
    {
      auto end = cookies.find(';', start);
      if (end == std::string::npos)
      {
        end = cookies.size();
      }

      auto pair = cookies.substr(start, end - start);
      auto eq = pair.find('=');
      if (eq != std::string::npos)
      {
        auto name = pair.substr(0, eq);
        auto value = pair.substr(eq + 1);
        while (!name.empty() && name.front() == ' ')
        {
          name.erase(name.begin());
        }
        if (name == key)
        {
          return value;
        }
      }

      start = end + 1;
    }

    return {};
  }
}

void http_connection::handle_count(middleware::HttpContext &ctx)
{
  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "text/html");

  std::string html;
  if (program_state_test::load_text_file("static/count.html", html))
  {
    program_state_test::replace_all(html, "{{count}}", std::to_string(program_state_test::request_count()));
    beast::ostream(ctx.response.body()) << html;
    return;
  }

  beast::ostream(ctx.response.body())
      << "<html>\n"
      << "<head><title>Request count</title></head>\n"
      << "<body>\n"
      << "<h1>Request count</h1>\n"
      << "<p>There have been "
      << program_state_test::request_count()
      << " requests so far.</p>\n"
      << "</body>\n"
      << "</html>\n";
}

void http_connection::handle_time(middleware::HttpContext &ctx)
{
  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "text/html");

  std::string html;
  if (program_state_test::load_text_file("static/time.html", html))
  {
    program_state_test::replace_all(html, "{{now}}", std::to_string(program_state_test::now()));
    beast::ostream(ctx.response.body()) << html;
    return;
  }

  beast::ostream(ctx.response.body())
      << "<html>\n"
      << "<head><title>Server Time</title></head>\n"
      << "<body>\n"
      << "<h1>Server Time</h1>\n"
      << "<p>Current unix timestamp: "
      << program_state_test::now()
      << "</p>\n"
      << "</body>\n"
      << "</html>\n";
}

void http_connection::handle_email(middleware::HttpContext &ctx)
{
  ctx.response.result(http::status::ok);

  Json::Value src_root;
  if (!parse_email_body(src_root))
  {
    return;
  }

  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");
  Json::Value root;
  auto email = src_root["email"].asString();
  std::cout << "email is " << email << std::endl;
  root["error"] = 0;
  root["email"] = src_root["email"];
  root["msg"] = "receive email post success";
  std::string jsonstr = root.toStyledString();
  beast::ostream(ctx.response.body()) << jsonstr;
}

void http_connection::handle_login_page(middleware::HttpContext &ctx)
{
  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "text/html; charset=utf-8");

  std::string html;
  if (program_state_test::load_text_file("static/login.html", html))
  {
    beast::ostream(ctx.response.body()) << html;
    return;
  }

  beast::ostream(ctx.response.body())
      << "<html><head><title>Login</title></head><body>"
      << "<h1>Login</h1>"
      << "<p>login.html not found</p>"
      << "</body></html>";
}

void http_connection::handle_login_submit(middleware::HttpContext &ctx)
{
  Json::Value root;
  if (!parse_login_body(root))
  {
    return;
  }

  SessionConfig config;
  config.user_file = "static/users.json";
  config.db_config_file = "static/db.json";
  config.ttl_seconds = 3600;

  auto &manager = SessionManager::instance();
  if (!manager.ensure_initialized(config))
  {
    send_error(http::status::internal_server_error, 1500, "session manager init failed");
    return;
  }

  const auto username = root["username"].asString();
  const auto password = root["password"].asString();
  auto user_id = manager.authenticate(username, password);
  if (!user_id)
  {
    send_error(http::status::unauthorized, 1403, "invalid username or password");
    return;
  }

  const auto session_id = manager.create_session(*user_id);
  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");
  ctx.response.set(http::field::set_cookie,
                   "SID=" + session_id + "; Path=/; HttpOnly; Max-Age=3600");

  Json::Value resp;
  resp["error"] = 0;
  resp["user"] = *user_id;
  resp["msg"] = "login success";
  resp["redirect"] = "/app";
  beast::ostream(ctx.response.body()) << resp.toStyledString();
}

void http_connection::handle_register_page(middleware::HttpContext &ctx)
{
  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "text/html; charset=utf-8");

  std::string html;
  if (program_state_test::load_text_file("static/register.html", html))
  {
    beast::ostream(ctx.response.body()) << html;
    return;
  }

  beast::ostream(ctx.response.body())
      << "<html><head><title>Register</title></head><body>"
      << "<h1>Register</h1>"
      << "<p>register.html not found</p>"
      << "</body></html>";
}

void http_connection::handle_register_submit(middleware::HttpContext &ctx)
{
  Json::Value root;
  if (!parse_login_body(root))
  {
    return;
  }

  SessionConfig config;
  config.user_file = "static/users.json";
  config.db_config_file = "static/db.json";
  config.ttl_seconds = 3600;

  auto &manager = SessionManager::instance();
  if (!manager.ensure_initialized(config))
  {
    send_error(http::status::internal_server_error, 1500, "session manager init failed");
    return;
  }

  std::string error_message;
  const auto username = root["username"].asString();
  const auto password = root["password"].asString();

  if (!manager.register_user(username, password, error_message))
  {
    send_error(http::status::bad_request, 1406, error_message);
    return;
  }

  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value resp;
  resp["error"] = 0;
  resp["user"] = username;
  resp["msg"] = "register success";
  beast::ostream(ctx.response.body()) << resp.toStyledString();
}

void http_connection::handle_me(middleware::HttpContext &ctx)
{
  SessionConfig config;
  config.user_file = "static/users.json";
  config.db_config_file = "static/db.json";
  config.ttl_seconds = 3600;

  auto &manager = SessionManager::instance();
  if (!manager.ensure_initialized(config))
  {
    send_error(http::status::internal_server_error, 1500, "session manager init failed");
    return;
  }

  const auto session_id = read_cookie_value(ctx.request, "SID");
  if (session_id.empty())
  {
    send_error(http::status::unauthorized, 1403, "missing session");
    return;
  }

  auto record = manager.get_session(session_id);
  if (!record)
  {
    send_error(http::status::unauthorized, 1403, "session expired");
    return;
  }

  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value resp;
  resp["error"] = 0;
  resp["user"] = record->user_id;
  resp["msg"] = "session valid";
  beast::ostream(ctx.response.body()) << resp.toStyledString();
}

void http_connection::handle_app_page(middleware::HttpContext &ctx)
{
  SessionConfig config;
  config.user_file = "static/users.json";
  config.db_config_file = "static/db.json";
  config.ttl_seconds = 3600;

  auto &manager = SessionManager::instance();
  if (!manager.ensure_initialized(config))
  {
    send_error(http::status::internal_server_error, 1500, "session manager init failed");
    return;
  }

  const auto session_id = read_cookie_value(ctx.request, "SID");
  if (session_id.empty() || !manager.get_session(session_id))
  {
    send_error(http::status::unauthorized, 1403, "login required");
    return;
  }

  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "text/html; charset=utf-8");

  std::string html;
  if (program_state_test::load_text_file("static/app.html", html))
  {
    beast::ostream(ctx.response.body()) << html;
    return;
  }

  beast::ostream(ctx.response.body())
      << "<html><head><title>App</title></head><body>"
      << "<h1>App</h1>"
      << "<p>app.html not found</p>"
      << "</body></html>";
}

void http_connection::handle_chat_echo(middleware::HttpContext &ctx)
{
  Json::Value root;
  if (!parse_chat_body(root))
  {
    return;
  }

  SessionConfig config;
  config.user_file = "static/users.json";
  config.db_config_file = "static/db.json";
  config.ttl_seconds = 3600;

  auto &manager = SessionManager::instance();
  if (!manager.ensure_initialized(config))
  {
    send_error(http::status::internal_server_error, 1500, "session manager init failed");
    return;
  }

  const auto session_id = read_cookie_value(ctx.request, "SID");
  if (session_id.empty())
  {
    send_error(http::status::unauthorized, 1403, "missing session");
    return;
  }

  auto record = manager.get_session(session_id);
  if (!record)
  {
    send_error(http::status::unauthorized, 1403, "session expired");
    return;
  }

  const auto user_text = root["message"].asString();
  const auto reply_text = std::string("echo: ") + user_text;

  std::string db_error;
  if (!manager.append_chat_message(record->session_id, record->user_id, "user", user_text, db_error))
  {
    send_error(http::status::internal_server_error, 1501, db_error);
    return;
  }

  if (!manager.append_chat_message(record->session_id, record->user_id, "assistant", reply_text, db_error))
  {
    send_error(http::status::internal_server_error, 1501, db_error);
    return;
  }

  auto history = manager.list_user_chat_messages(record->user_id, 50, db_error);
  if (!db_error.empty())
  {
    send_error(http::status::internal_server_error, 1502, db_error);
    return;
  }

  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value resp;
  resp["error"] = 0;
  resp["reply"] = reply_text;
  resp["user"] = record->user_id;
  resp["history"] = Json::arrayValue;
  for (const auto &item : history)
  {
    Json::Value row;
    row["role"] = item.role;
    row["content"] = item.content;
    row["created_at"] = static_cast<Json::Int64>(item.created_at_epoch);
    resp["history"].append(row);
  }

  beast::ostream(ctx.response.body()) << resp.toStyledString();
}

void http_connection::handle_chat_completions(middleware::HttpContext &ctx)
{
  Json::Value root;
  if (!parse_chat_body(root))
  {
    return;
  }

  SessionConfig config;
  config.user_file = "static/users.json";
  config.db_config_file = "static/db.json";
  config.ttl_seconds = 3600;

  auto &manager = SessionManager::instance();
  if (!manager.ensure_initialized(config))
  {
    send_error(http::status::internal_server_error, 1500, "session manager init failed");
    return;
  }

  const auto session_id = read_cookie_value(ctx.request, "SID");
  if (session_id.empty())
  {
    send_error(http::status::unauthorized, 1403, "missing session");
    return;
  }

  auto record = manager.get_session(session_id);
  if (!record)
  {
    send_error(http::status::unauthorized, 1403, "session expired");
    return;
  }

  const auto user_text = root["message"].asString();
  llama::GenerationConfig gen_cfg;
  gen_cfg.temperature = 0.0f;
  gen_cfg.max_tokens = root.get("max_tokens", 32).asInt();
  if (gen_cfg.max_tokens <= 0 || gen_cfg.max_tokens > 256)
  {
    send_error(http::status::bad_request, 1006, "max_tokens must be in range [1, 256]");
    return;
  }

  std::string reply_text;
  try
  {
    reply_text = inference_engine().generate_text(user_text, gen_cfg);
  }
  catch (const std::exception &e)
  {
    send_error(http::status::internal_server_error, 1503, e.what());
    return;
  }

  std::string db_error;
  if (!manager.append_chat_message(record->session_id, record->user_id, "user", user_text, db_error))
  {
    send_error(http::status::internal_server_error, 1501, db_error);
    return;
  }

  if (!manager.append_chat_message(record->session_id, record->user_id, "assistant", reply_text, db_error))
  {
    send_error(http::status::internal_server_error, 1501, db_error);
    return;
  }

  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value resp;
  resp["error"] = 0;
  resp["reply"] = reply_text;
  resp["user"] = record->user_id;
  resp["model"] = "tinyllama";
  resp["max_tokens"] = gen_cfg.max_tokens;
  beast::ostream(ctx.response.body()) << resp.toStyledString();
}

void http_connection::handle_chat_history(middleware::HttpContext &ctx)
{
  SessionConfig config;
  config.user_file = "static/users.json";
  config.db_config_file = "static/db.json";
  config.ttl_seconds = 3600;

  auto &manager = SessionManager::instance();
  if (!manager.ensure_initialized(config))
  {
    send_error(http::status::internal_server_error, 1500, "session manager init failed");
    return;
  }

  const auto session_id = read_cookie_value(ctx.request, "SID");
  if (session_id.empty())
  {
    send_error(http::status::unauthorized, 1403, "missing session");
    return;
  }

  auto record = manager.get_session(session_id);
  if (!record)
  {
    send_error(http::status::unauthorized, 1403, "session expired");
    return;
  }

  std::string db_error;
  auto history = manager.list_user_chat_messages(record->user_id, 50, db_error);
  if (!db_error.empty())
  {
    send_error(http::status::internal_server_error, 1502, db_error);
    return;
  }

  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value resp;
  resp["error"] = 0;
  resp["user"] = record->user_id;
  resp["history"] = Json::arrayValue;
  for (const auto &item : history)
  {
    Json::Value row;
    row["role"] = item.role;
    row["content"] = item.content;
    row["created_at"] = static_cast<Json::Int64>(item.created_at_epoch);
    resp["history"].append(row);
  }

  beast::ostream(ctx.response.body()) << resp.toStyledString();
}

void http_connection::handle_image_process(middleware::HttpContext &ctx)
{
  ctx.response.result(http::status::not_implemented);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value resp;
  resp["error"] = 0;
  resp["msg"] = "image process endpoint reserved";
  resp["next"] = "python route integration";
  beast::ostream(ctx.response.body()) << resp.toStyledString();
}

void http_connection::handle_time_by_zone(middleware::HttpContext &ctx, const router::route_params &params)
{
  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value root;
  root["error"] = 0;
  root["route"] = "dynamic";
  root["timestamp"] = static_cast<Json::Int64>(program_state_test::now());

  auto zone_it = params.find("zone");
  root["zone"] = zone_it != params.end() ? zone_it->second : "unknown";

  beast::ostream(ctx.response.body()) << root.toStyledString();
}

void http_connection::handle_email_by_source(middleware::HttpContext &ctx, const router::route_params &params)
{
  Json::Value src_root;
  if (!parse_email_body(src_root))
  {
    return;
  }

  ctx.response.result(http::status::ok);
  ctx.response.set(http::field::content_type, "application/json; charset=utf-8");

  Json::Value root;
  root["error"] = 0;
  root["route"] = "dynamic";
  root["email"] = src_root["email"];

  auto source_it = params.find("source");
  root["source"] = source_it != params.end() ? source_it->second : "unknown";
  root["msg"] = "receive dynamic email post success";

  beast::ostream(ctx.response.body()) << root.toStyledString();
}

bool http_connection::parse_email_body(Json::Value &src_root)
{
  if (!is_json_content_type())
  {
    send_error(http::status::unsupported_media_type, 1415,
               "content-type must be application/json");
    return false;
  }

  if (_request.body().size() == 0)
  {
    send_error(http::status::bad_request, 1402, "request body is empty");
    return false;
  }

  auto &body = this->_request.body();
  auto body_str = boost::beast::buffers_to_string(body.data());
  std::cout << "receive body is " << body_str << std::endl;

  Json::Reader reader;
  const bool parse_success = reader.parse(body_str, src_root);
  if (!parse_success)
  {
    send_error(http::status::bad_request, 1001, "invalid json body");
    return false;
  }

  if (!src_root.isMember("email") || !src_root["email"].isString() ||
      src_root["email"].asString().empty())
  {
    send_error(http::status::bad_request, 1002,
               "field 'email' is required and must be a non-empty string");
    return false;
  }

  return true;
}

bool http_connection::parse_login_body(Json::Value &src_root)
{
  if (!is_json_content_type())
  {
    send_error(http::status::unsupported_media_type, 1415,
               "content-type must be application/json");
    return false;
  }

  if (_request.body().size() == 0)
  {
    send_error(http::status::bad_request, 1402, "request body is empty");
    return false;
  }

  auto &body = _request.body();
  auto body_str = boost::beast::buffers_to_string(body.data());

  Json::Reader reader;
  const bool parse_success = reader.parse(body_str, src_root);
  if (!parse_success)
  {
    send_error(http::status::bad_request, 1001, "invalid json body");
    return false;
  }

  if (!src_root.isMember("username") || !src_root["username"].isString() ||
      src_root["username"].asString().empty())
  {
    send_error(http::status::bad_request, 1003,
               "field 'username' is required and must be a non-empty string");
    return false;
  }

  if (!src_root.isMember("password") || !src_root["password"].isString() ||
      src_root["password"].asString().empty())
  {
    send_error(http::status::bad_request, 1004,
               "field 'password' is required and must be a non-empty string");
    return false;
  }

  return true;
}

bool http_connection::parse_chat_body(Json::Value &src_root)
{
  if (!is_json_content_type())
  {
    send_error(http::status::unsupported_media_type, 1415,
               "content-type must be application/json");
    return false;
  }

  if (_request.body().size() == 0)
  {
    send_error(http::status::bad_request, 1402, "request body is empty");
    return false;
  }

  auto body_str = boost::beast::buffers_to_string(_request.body().data());
  Json::Reader reader;
  if (!reader.parse(body_str, src_root))
  {
    send_error(http::status::bad_request, 1001, "invalid json body");
    return false;
  }

  if (!src_root.isMember("message") || !src_root["message"].isString() ||
      src_root["message"].asString().empty())
  {
    send_error(http::status::bad_request, 1005,
               "field 'message' is required and must be a non-empty string");
    return false;
  }

  return true;
}

void http_connection::write_response()
{
  auto self = shared_from_this();
  _response.content_length(_response.body().size());
  http::async_write(
      _socket,
      _response,
      [self](beast::error_code ec, std::size_t)
      {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - self->_request_start)
                              .count();
        std::cout << "[HTTP] " << self->_request.method_string() << " "
                  << self->_request.target() << " status="
                  << static_cast<unsigned>(self->_response.result_int())
                  << " cost_ms=" << elapsed_ms;
        if (!self->_last_error_reason.empty())
        {
          std::cout << " error=\"" << self->_last_error_reason << "\"";
        }
        if (ec)
        {
          std::cout << " write_error=\"" << ec.message() << "\"";
        }
        std::cout << std::endl;

        self->_socket.shutdown(tcp::socket::shutdown_send, ec);
        boost::system::error_code close_ec;
        self->_socket.close(close_ec);
        self->_timer.cancel();
      });
}

void http_connection::send_simple_error(http::status status, int error_code, const std::string &message)
{
  _request_start = std::chrono::steady_clock::now();
  _response.version(11);
  _response.keep_alive(false);
  _response.set(http::field::server, "Beast");
  send_error(status, error_code, message);
  write_response();
}

void http_connection::send_error(http::status status, int error_code, const std::string &message)
{
  _response.result(status);
  _response.set(http::field::content_type, "application/json; charset=utf-8");
  _last_error_reason = message;

  _response.body().clear();
  Json::Value root;
  root["error"] = error_code;
  root["message"] = message;
  root["status"] = static_cast<int>(status);
  beast::ostream(_response.body()) << root.toStyledString();
}

bool http_connection::is_json_content_type() const
{
  if (!_request.base().count(http::field::content_type))
  {
    return false;
  }

  std::string content_type(_request[http::field::content_type]);
  std::transform(content_type.begin(), content_type.end(), content_type.begin(),
                 [](unsigned char c)
                 { return static_cast<char>(std::tolower(c)); });
  return content_type.find("application/json") != std::string::npos;
}

void http_connection::log_transport_error(const std::string &phase, const std::string &error)
{
  std::cout << "[HTTP] transport_error phase=" << phase << " reason=\""
            << error << "\"" << std::endl;
}
