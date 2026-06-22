# InferFlow 大模型推理服务框架

InferFlow 是一个面向本地与私有化部署场景的轻量级大模型推理服务框架。当前重点围绕 decoder-only LLM 构建模型加载、Tokenizer、KV Cache、token-level decoding、采样与命令行交互能力，并为后续 HTTP API、CUDA backend、Int8 量化和多模态 embedding 扩展预留接口。

当前验证模型为 TinyLlama-1.1B，推理流程与 LLaMA 类 decoder-only 模型保持一致，适合用于调试 RMSNorm、RoPE、Attention、SwiGLU、KV Cache 和自回归生成流程。

---

## 当前能力

### 推理引擎

- 支持 LLaMA 类 decoder-only Transformer 前向推理。
- 实现 RMSNorm、RoPE、GQA Attention、SwiGLU、LM Head 等核心模块。
- 实现 KV Cache 加速的 autoregressive decoding。
- 实现 temperature、top-k、top-p 采样器。
- 实现 `generate()`、`generate_text()`、`generate_text_stream()`。
- 接入 C++ SentencePiece tokenizer，支持文本 encode/decode。
- 支持 stop string 截断，避免输出 `</s>`、`<|user|>` 等模板标记。
- 提供命令行对话 demo `inferflow_cli`。

### 正确性验证

- Python/NumPy 原型用于对齐 HuggingFace 行为。
- C++ 推理引擎已与 HuggingFace 逐 token 前向结果对齐。
- TinyLlama 测试样例中 top-5 logits match 通过，最大误差约 `4.77e-05`。

### 服务化承载

项目内保留 Boost.Asio/Beast HTTP 服务模块，用于后续将本地推理能力封装为 `/chat/completions` API。该部分包含基础路由、中间件、Cookie Session、MySQL 会话与聊天历史持久化，但当前 README 不再以 Web 网关为主要叙事。

---

## 目录结构

```text
notix/
├── llama-engine/                  # 推理引擎主体
│   ├── include/llama/
│   │   ├── config.h                # ModelConfig / GenerationConfig
│   │   ├── weights.h               # 权重结构
│   │   ├── ops.h                   # RMSNorm / RoPE / Attention / FFN
│   │   ├── kv_cache.h              # KV Cache
│   │   ├── sampler.h               # 采样器
│   │   ├── tokenizer.h             # Tokenizer 抽象与 SentencePiece 实现
│   │   └── engine.h                # LlamaEngine 主接口
│   ├── src/                        # 推理引擎实现
│   ├── demo/chat_cli.cpp           # 命令行对话 demo
│   └── test/                       # 单元测试与 HF 对齐测试
│
├── python_prototype/               # NumPy 参考实现与 HF 对齐脚本
├── notix/httpserver/               # HTTP 服务化承载层
└── docs/                           # 设计文档与阶段总结
```

---

## 构建推理引擎

依赖:

- CMake >= 3.16
- C++17 编译器
- Eigen 3.4，已放在 `llama-engine/third_party/eigen3`
- C++ SentencePiece，默认从 `/usr/local/include` 与 `/usr/local/lib` 查找

构建:

```bash
cd notix/llama-engine
cmake -S . -B build
cmake --build build -j2
```

---

## 运行命令行推理

默认模型路径:

- `notix/llama-engine/test/tinyllama.bin`
- `models/tokenizer.model`

运行:

```bash
cd notix/llama-engine/build
./inferflow_cli
```

也可以显式传入模型、tokenizer 和最大生成 token 数:

```bash
./inferflow_cli ../test/tinyllama.bin ../../../models/tokenizer.model 64
```

支持环境变量覆盖:

```bash
INFERFLOW_MODEL_PATH=/path/to/model.bin \
INFERFLOW_TOKENIZER_PATH=/path/to/tokenizer.model \
./inferflow_cli
```

CLI 命令:

```text
/exit   退出
/quit   退出
/clear  清空对话历史
```

---

## 运行测试

```bash
cd notix/llama-engine/build
./test_ops
./test_engine
./test_forward ../test/tinyllama.bin
```

当前验证结果:

```text
test_ops: 8/8
test_engine: 8/8
test_forward: Top-5 match 6/6, max_diff=4.76837e-05
```

---

## 推理流程

```text
Prompt
  -> SentencePiece Tokenizer
  -> Token IDs
  -> Embedding
  -> Transformer Layer x N
       -> RMSNorm
       -> QKV Projection
       -> RoPE
       -> Attention with KV Cache
       -> SwiGLU FFN
  -> Final RMSNorm
  -> LM Head
  -> Sampler
  -> Next Token
  -> Decode / Stream Output
```

CLI 当前在 demo 层使用 TinyLlama-Chat 风格 prompt template:

```text
<|system|>
You are a helpful assistant. Answer concisely.
</s>
<|user|>
...
</s>
<|assistant|>
```

---

## 已完成里程碑

- Phase 0: Python/NumPy LLaMA 前向原型，与 HuggingFace 对齐。
- Phase 1: C++ LLaMA 推理核心实现，完成 TinyLlama 前向验证。
- Phase 2-A: `generate()` 生成闭环、SentencePiece tokenizer、文本级生成接口。
- Phase 2-B: `inferflow_cli` 命令行对话 demo、chat prompt template、stop string 截断。

---

## 下一步计划

### 可用化

- 抽象独立 `ChatTemplate` 模块。
- 完成 HTTP `/chat/completions` 非流式接口。
- 增加 SSE token streaming。
- 增加 InferenceWorker，避免推理阻塞 HTTP event loop。

### 模型格式

- 支持 GGUF 或 safetensors 加载。
- 支持 Llama-2-7B 权重导出与验证。
- 扩展 Llama3 / Qwen BPE tokenizer。

### 性能优化

- 设计统一 Tensor / allocator 抽象。
- 设计 CPU/GPU backend interface。
- 将 KV Cache、activation buffer、中间结果纳入设备侧 Tensor 管理。
- 接入 RMSNorm、MatMul、Attention CUDA backend。
- 预研 Int8 group-wise weight-only quantization 与动态反量化 MatMul。

### 多模态扩展

- 预留 Vision Encoder 接口。
- 预留 multimodal embedding 拼接接口。
- 为后续 LLaVA / Qwen-VL 类模型接入准备模型执行图抽象。

---

## 当前限制

- 当前主路径仍是 CPU 推理，速度较慢。
- 模型加载仍使用自定义 binary 格式。
- tokenizer 首版只支持 SentencePiece。
- HTTP 服务模块依赖 Boost，本机未安装 Boost 时只建议运行 `llama-engine` 与 CLI。
- 多轮对话目前通过 prompt history 拼接实现，尚未复用会话级 KV Cache。
