#pragma once
#include "global.h"
#include "router.h"
#include "middleware.h"
#include <functional>
#include <iostream>
#include <unordered_map>

/*
beast的http类
*/
class http_connection : public std::enable_shared_from_this<http_connection>
{
public:
  static constexpr std::size_t MaxBodySize = 1024 * 1024; // 1MB

  /**
   * @brief 构造一个 HTTP 连接对象。
   * @param socket 已建立的 TCP socket，将被移动到连接对象内部管理。
   */
  http_connection(tcp::socket socket);

  /**
   * @brief 启动连接处理流程。
   *
   * 会触发异步读取请求并启动超时检测。
   * 即 read_request 与 check_deadline
   */
  void start();

  /**
   * @brief 异步读取 HTTP 请求。
   *
   * 内部会应用请求体大小限制，并在读取完成后进入请求处理逻辑。
   */
  void read_request();

  // 检查超时连接
  /**
   * @brief 检查连接超时并在必要时关闭连接。
   */
  void check_deadline();

  /**
   * @brief 根据 HTTP Method 分发请求。
   *        目前支持get与post
   */
  void process_request();

  /**
   * @brief 根据路由匹配结果执行最终业务处理
   * @param ctx 中间件上下文
   */
  void dispatch_route(middleware::HttpContext &ctx);

  // get消息路由
  /**
   * @brief 注册路由与 handler 的对应关系。
   */
  void register_route_handlers();

  /**
   * @brief 静态路由处理：/count
   */
  void handle_count(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/time
   */
  void handle_time(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/email (POST)
   */
  void handle_email(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/login (GET)
   */
  void handle_login_page(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/login (POST)
   */
  void handle_login_submit(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/register (GET)
   */
  void handle_register_page(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/register (POST)
   */
  void handle_register_submit(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/me
   */
  void handle_me(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/app
   */
  void handle_app_page(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/chat/echo
   */
  void handle_chat_echo(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/chat/completions
   */
  void handle_chat_completions(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/chat/history
   */
  void handle_chat_history(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/chat/clear
   */
  void handle_chat_clear(middleware::HttpContext &ctx);

  /**
   * @brief 静态路由处理：/image/process（占位）
   */
  void handle_image_process(middleware::HttpContext &ctx);

  /**
   * @brief 动态路由处理：/time/{zone}
   */
  void handle_time_by_zone(middleware::HttpContext &ctx, const router::route_params &params);

  /**
   * @brief 动态路由处理：/email/{source}
   */
  void handle_email_by_source(middleware::HttpContext &ctx, const router::route_params &params);

  /**
   * @brief 将响应异步写回客户端并记录访问日志。
   */
  void write_response();

  /**
   * @brief 在读取阶段等早期错误场景下返回简化错误响应。
   * @param status HTTP 状态码。
   * @param error_code 业务错误码。
   * @param message 错误描述。
   */
  void send_simple_error(http::status status, int error_code, const std::string &message);

  /**
   * @brief 构造统一 JSON 错误响应。
   * @param status HTTP 状态码。
   * @param error_code 业务错误码。
   * @param message 错误描述。
   */
  void send_error(http::status status, int error_code, const std::string &message);

  /**
   * @brief 判断请求的 Content-Type 是否为 JSON。
   * @return true 表示 Content-Type 包含 application/json；否则返回 false。
   */
  bool is_json_content_type() const;

  /**
   * @brief 解析并校验 email JSON 请求体。
   * @param src_root 解析后的 JSON 对象。
   * @return true 表示成功；false 表示已写入错误响应。
   */
  bool parse_email_body(Json::Value &src_root);

  /**
   * @brief 解析并校验登录 JSON 请求体。
   * @param src_root 解析后的 JSON 对象。
   * @return true 表示成功；false 表示已写入错误响应。
   */
  bool parse_login_body(Json::Value &src_root);

  /**
   * @brief 解析并校验聊天请求体。
   */
  bool parse_chat_body(Json::Value &src_root);

  /**
   * @brief 记录传输层错误日志。
   * @param phase 错误发生阶段（如 read/write）。
   * @param error 错误信息文本。
   */
  void log_transport_error(const std::string &phase, const std::string &error);

  /**
   * @brief 注册连接级默认中间件。
   */
  void setup_middlewares();

private:
  tcp::socket _socket;
  beast::flat_buffer _buffer{8192};
  http::request_parser<http::dynamic_body> _request_parser;
  http::request<http::dynamic_body> _request;
  http::response<http::dynamic_body> _response;
  std::chrono::steady_clock::time_point _request_start;
  std::string _last_error_reason;
  MiddlewarePipeline _middleware_pipeline;

  using StaticHandler = std::function<void(middleware::HttpContext &)>;
  using DynamicHandler = std::function<void(middleware::HttpContext &, const router::route_params &)>;

  // 使用枚举作为 key，避免在 http_connection 中写大量 if/else 分支。
  std::unordered_map<router::static_route, StaticHandler> _static_handlers;
  std::unordered_map<router::dynamic_route_id, DynamicHandler> _dynamic_handlers;
  // 定时器
  net::steady_timer _timer{
      _socket.get_executor(), std::chrono::seconds(60)};
};
