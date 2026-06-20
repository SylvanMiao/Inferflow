# Notix 项目说明

## 项目简述

开发平台：`Ubuntu 22.04`

当前核心目标是基于 Asio + Beast 构建可演进的 HTTP 网关框架，在此基础上集成**本地 LLM 推理引擎**，实现 local-first 的 AI 对话服务。

已规划：

1. 使用 Asio 构建异步网络服务器
2. 使用 Beast 实现 HTTP 网关能力
3. 用户信息录入与管理
4. **本地 LLM 推理能力集成**（自研 C++ 推理引擎）
5. LLM 问答模块集成

---

## 项目目录

```
notix/
├── httpserver/                    # HTTP 服务框架（已有）
│   ├── inc/
│   │   ├── http/                  # Beast HTTP 连接处理
│   │   ├── router/                # 静态/动态路由
│   │   ├── middleware/             # 中间件（before/after 管线）
│   │   ├── session/               # Session 管理
│   │   └── db/                    # MySQL 连接池
│   ├── src/                       # 实现文件
│   └── static/                    # 前端页面（HTML）
│
├── python_prototype/              # Phase 0: NumPy 参考实现
│   ├── llama_forward.py            # 完整 LLaMA 前向传播（已验证）
│   ├── verify_against_hf.py        # 与 HuggingFace 对比验证
│   ├── export_model.py             # 导出 HF 权重 → C++ 二进制格式
│   └── README.md
│
├── llama-engine/                  # Phase 1: C++ 推理引擎
│   ├── CMakeLists.txt
│   ├── third_party/eigen3/        # Eigen 3.4（线性代数，header-only）
│   ├── include/llama/
│   │   ├── config.h                # ModelConfig + GenerationConfig
│   │   ├── weights.h               # 权重结构体
│   │   ├── ops.h                   # 算子（rmsnorm, rope, attention 等）
│   │   ├── kv_cache.h              # KV 缓存
│   │   ├── sampler.h               # 采样器（temperature, top-k, top-p）
│   │   └── engine.h                # 主引擎接口
│   ├── src/                        # 算子 + 引擎实现
│   └── test/                       # 单元测试 + HF 对比验证
│
└── docs/
    ├── llm-inference-engine-plan.md   # 技术方案（详细设计）
    └── phase1-cpp-engine-summary.md   # Phase 1 总结
```

---

## 当前能力状态（2026-06-20）

### HTTP 框架

- 基础 HTTP 网关：GET/POST 分发、统一 JSON 错误返回、请求体大小限制
- 路由框架：静态路由 + 动态路由 + handler 注册分发
- 中间件机制：before/after 管线、方法守卫、上下文封装
- 用户体系：登录、注册、会话 Cookie、Session 持久化
- 数据库能力：MySQL 连接池、Session 持久化、聊天记录持久化
- 前端页面：登录页、注册页、功能页（对话区 + 图像处理占位区）
- AI 接口雏形：`/chat/echo`、`/image/process`

### LLM 推理引擎

- **Phase 0: NumPy 原型** ✅ — 完整 LLaMA 前向传播实现，已与 HuggingFace 逐 token 验证通过
- **Phase 1: C++ 引擎** ✅ — 核心算子（RMSNorm, RoPE, Attention, SwiGLU）全部实现，与 HF 输出完全一致（max_diff < 5e-5）
- 模型支持：LLaMA 架构（已验证 TinyLlama-1.1B），支持 GQA
- 采样策略：temperature、top-k、top-p
- 待完成：Tokenizer 集成、GGUF 加载、Notix InferenceWorker、SSE 流式端点

---

## AI 能力路线图

### 已完成

- **Phase 0**：Python/NumPy 原型，与 HuggingFace 对比验证（5/5 全 PASS）
- **Phase 1**：C++ 推理引擎核心，与 HF 前向对比验证（6/6 全 PASS，12 项单元测试全通过）

### 计划中

**Phase 2：可用化**

| 任务 | 说明 |
|------|------|
| Tokenizer 集成 | 封装 SentencePiece，文本 ↔ token |
| GGUF 模型加载 | 替换自定义 binary，支持社区模型 |
| Notix InferenceWorker | 推理线程池 + 异步请求队列 |
| SSE `/chat/completions` | 流式输出，对标 OpenAI API |
| 前端流式适配 | app.html 接入流式对话 |

**Phase 3：性能优化**

| 任务 | 说明 |
|------|------|
| OpenBLAS 加速 | 大矩阵乘法 3-5× 提升 |
| FP16 推理 | 内存占用减半 |
| CUDA 后端 | GPU 推理支持 |

**Phase 4：功能扩展**

| 任务 | 说明 |
|------|------|
| 多模型架构 | Qwen、Mistral 等 |
| Tool calling | Function calling |
| 多模态 | 图像理解模型 |

---

## 技术架构（LLM 引擎部分）

```
HTTP Request → Notix Router → InferenceGateway
                                  │
                    boost::asio::post (异步)
                                  │
                    ┌─────────────▼─────────────┐
                    │   Inference Thread Pool    │
                    │  ┌───────────────────────┐ │
                    │  │    LlamaEngine         │ │
                    │  │  forward(token, pos)   │ │
                    │  │    → embedding         │ │
                    │  │    → [Layer × N]       │ │
                    │  │       ├─ RMSNorm       │ │
                    │  │       ├─ QKV + RoPE    │ │
                    │  │       ├─ Attention     │ │
                    │  │       └─ SwiGLU FFN    │ │
                    │  │    → RMSNorm → LM Head │ │
                    │  │    → Sampler → token   │ │
                    │  └───────────────────────┘ │
                    └───────────────────────────┘
                                  │
                    SSE stream → HTTP Response
```

每个 C++ 算子的实现均有 NumPy 版本作为参考标准。

---

## 开发里程碑

- **2026-03-10**：验证 Boost.Beast 与 JsonCpp 基础可用
- **2026-04-13**：完成基础 Beast HTTP 框架与路由雏形
- **2026-04-16**：中间件框架落地（before/after + 可视化验证）
- **2026-04-18**：将路由回调修改为 handler 风格
- **2026-04-19**：实现用户与数据基础能力（注册/登录/会话/MySQL），开放 Echo 与图像处理占位接口
- **2026-06-20**：Phase 0 NumPy 原型完成，与 HuggingFace 全量验证通过
- **2026-06-20**：Phase 1 C++ 推理引擎完成，与 HF 前向对比 6/6 全 PASS，12 项单元测试全通过

---

## 构建与运行

### 推理引擎

```bash
cd llama-engine
mkdir -p build && cd build
cmake ..
make -j$(nproc)
./test_ops      # 算子单元测试
./test_engine   # 引擎集成测试
```

### Python 原型

```bash
cd python_prototype
pip install -r requirements.txt
python llama_forward.py                        # 独立测试
python verify_against_hf.py --model ../model/TinyLlama-1.1B-Chat-v1.0  # HF 对比
```

### HTTP 服务

```bash
cd httpserver
mkdir -p build && cd build
cmake ..
make
./http_server
```
