#pragma once

#include "global.h"
#include <string>
#include <string_view>
#include <unordered_map>

namespace router
{
    /**
     * @brief 动态路由参数集合
     *
     * key 为参数名，value 为路径中提取出的参数值
     */
    using route_params = std::unordered_map<std::string, std::string>;

	/**
	 * @brief 静态路由枚举
	 */
	enum class static_route
	{
		none,  ///< 未命中静态路由
		count, ///< `/count`
		time,  ///< `/time`
		email, ///< `/email`
		login_page,   ///< `GET /login`
		login_submit, ///< `POST /login`
		register_page,   ///< `GET /register`
		register_submit, ///< `POST /register`
		app_page,         ///< `GET /app`
		chat_history,     ///< `GET /chat/history`
		chat_echo,        ///< `POST /chat/echo`
		chat_completions, ///< `POST /chat/completions`
		image_process,    ///< `POST /image/process`
		me               ///< `GET /me`
	};

	/**
	 * @brief 动态路由枚举
	 */
	enum class dynamic_route_id
	{
		none,          ///< 未命中动态路由
		time_by_zone,  ///< `/time/{zone}`
		email_by_source ///< `/email/{source}`
	};

    /**
     * @brief 动态路由规则
     */
	struct dynamic_route
	{
		http::verb method;      ///< 规则对应 HTTP 方法
		std::string pattern;    ///< 动态路由模板，如 `/email/{source}`
		dynamic_route_id id;    ///< 动态路由标识，用于 handler 注册
	};

	/** @brief 静态路由：请求计数页面 */
	inline constexpr std::string_view CountRoute = "/count";
	/** @brief 静态路由：时间页面 */
	inline constexpr std::string_view TimeRoute = "/time";
	/** @brief 静态路由：邮箱提交 */
	inline constexpr std::string_view EmailRoute = "/email";
	/** @brief 静态路由：登录 */
	inline constexpr std::string_view LoginRoute = "/login";
	/** @brief 静态路由：注册 */
	inline constexpr std::string_view RegisterRoute = "/register";
	/** @brief 静态路由：功能页 */
	inline constexpr std::string_view AppRoute = "/app";
	/** @brief 静态路由：会话 echo */
	inline constexpr std::string_view ChatEchoRoute = "/chat/echo";
	/** @brief 静态路由：LLM 对话补全 */
	inline constexpr std::string_view ChatCompletionsRoute = "/chat/completions";
	/** @brief 静态路由：聊天历史 */
	inline constexpr std::string_view ChatHistoryRoute = "/chat/history";
	/** @brief 静态路由：图像处理占位 */
	inline constexpr std::string_view ImageProcessRoute = "/image/process";
	/** @brief 静态路由：登录态查询 */
	inline constexpr std::string_view MeRoute = "/me";
    /** @brief 动态路由：按时区查询时间 */
    inline constexpr std::string_view TimeByZoneRoute = "/time/{zone}";
	/** @brief 动态路由：按来源提交邮箱 */
	inline constexpr std::string_view EmailBySourceRoute = "/email/{source}";

	/**
	 * @brief 规范化请求 target，仅保留 path 并去除末尾多余 '/'.
	 * @param raw_target 原始请求目标字符串
	 * @return std::string 规范化后的路径字符串
	 */
	std::string normalize_target(std::string_view raw_target);

	/**
	 * @brief 按 method + target 匹配静态路由
	 * @param method 请求 HTTP 方法
	 * @param target 请求目标路径
	 * @return static_route 命中的静态路由类型，未命中时返回 `static_route::none`
	 */
	static_route match_static_route(http::verb method, std::string_view target);

	/**
	 * @brief 匹配动态路由模板，成功时写入路径参数
	 * @param route 动态路由规则
	 * @param method 当前请求 HTTP 方法
	 * @param target 当前请求路径
	 * @param params 输出参数，命中时写入提取出的路径参数
	 * @return true 命中动态路由；false 未命中
	 */
	bool match_dynamic_route(const dynamic_route &route,
													 http::verb method,
													 std::string_view target,
													 route_params &params);
}
