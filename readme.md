# Notix 项目说明

## 项目简述

开发平台：`Ubuntu 22.04`

当前核心目标是基于 Asio + Beast 构建可演进的 HTTP 网关框架，并逐步补齐中间件、路由、会话与生产化能力。
Local-first AI inference with a browser-based control interface

已规划：

1. 使用 Asio 构建异步网络服务器
2. 使用 Beast 实现 HTTP 网关能力
3. 用户信息录入与管理
4. 本地小模型推理能力集成
5. LLM 问答模块集成

---

## 当前目录（HTTP 相关）

- `notix/httpserver/`
	- `src/http/`：Beast 连接处理与请求分发
	- `src/router/`：静态/动态路由匹配
	- `src/middleware/`：中间件函数对象与注册
	- `inc/middleware/http_context.h`：中间件上下文
	- `static/`：HTML 模板资源

---

## 当前能力状态（2026-04-19）

### 已实现能力

- 基础 HTTP 网关：GET/POST 分发、统一 JSON 错误返回、请求体大小限制
- 路由框架：静态路由 + 动态路由 + handler 注册分发
- 中间件机制：before/after 管线、方法守卫、上下文封装
- 用户体系：登录、注册、会话 Cookie、Session 持久化
- 数据库能力：MySQL 连接池、Session 持久化、聊天记录持久化
- 页面能力：登录页、注册页、功能页（对话区 + 图像处理占位区）
- AI 接口雏形：`/chat/echo`、`/image/process`（占位）

### 现阶段结论

- C++ 端 HTTP 框架已具备作为 AI 网关的基础条件。
- 下一步重点从“功能可用”升级为“稳定可扩展”，并推进 Agent 与 ONNX 推理实接。

---

## HTTP 框架待加强（近期）

### 阶段 C：补齐 Web 基础能力（P1）

- 状态：**进行中**
- 加强项：
	- CORS（含 OPTIONS 预检）
	- 鉴权中间件（白名单 + 登录校验）
	- 静态文件服务化（统一资源处理）
	- 请求参数校验与统一错误码规范
	- 配置集中化（端口、超时、数据库配置）

### 阶段 D：生产化增强（P2）

- 状态：**未开始**
- 加强项：
	- TLS/HTTPS
	- 连接策略增强（keep-alive、超时、优雅关闭）
	- 限流与防刷（IP、用户、路由维度）
	- 可观测性指标（请求量、失败率、耗时分位）
	- 日志结构化与链路追踪字段

---

## AI 能力后续计划

### 阶段 E：Python Agent 工作流接入（P0）

- 路由主入口：`/chat/*`
- 目标：通过 C++ 网关接入 Python Agent 服务，支持输入输出编排
- 计划项：
	- 新增网关到 Python 服务的调用层（HTTP）
	- Chat 路由支持上下文透传（user/session/history）
	- 超时、重试、熔断与错误映射
	- 保留 Echo 兜底模式，确保服务可回退
	- 聊天记录落库策略完善（请求、响应、耗时、状态）

### 阶段 F：ONNX / vLLM 本地推理接入

- 路由主入口：`/image/process`
- 目标：在 C++ 侧完成模型加载、推理执行、结果返回
- 计划项：
	- 接入 ONNX Runtime（CMake 依赖、模型目录管理）
	- 图像输入协议统一（multipart 或 base64）
	- 预处理/后处理流水线封装
	- 推理会话复用与并发控制
	- 结果持久化（任务记录、参数、输出摘要）

---

## 最近开发里程碑

- **2026-03-10**：验证 Boost.Beast 与 JsonCpp 基础可用
- **2026-04-13**：完成基础 Beast HTTP 框架与路由雏形
- **2026-04-16**：中间件框架落地（before/after + 可视化验证）
- **2026-04-18**：将路由回调修改为 handler 风格，降低 `http_connection` 内分支复杂度
- **2026-04-19**：实现用户与数据基础能力（注册/登录/会话持久化/MySQL 连接池），并开放 Echo 与图像处理占位接口
---


