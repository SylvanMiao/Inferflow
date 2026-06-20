# LLM 推理引擎从零实现 & Notix 集成技术方案

> 版本: v1.0 | 日期: 2026-06-20 | 状态: 规划阶段

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [总体架构设计](#2-总体架构设计)
3. [推理引擎核心设计](#3-推理引擎核心设计)
4. [Notix 集成方案](#4-notix-集成方案)
5. [实施路线图](#5-实施路线图)
6. [Kuiper 参考索引](#6-kuiper-参考索引)
7. [测试与验证策略](#7-测试与验证策略)
8. [风险与缓解](#8-风险与缓解)
9. [附录](#9-附录)

---

## 1. 项目背景与目标

### 1.1 现状

- **Notix**: 基于 Boost.Asio + Boost.Beast 的 C++17 HTTP 服务框架，已具备路由、中间件、用户认证、Session 管理、MySQL 持久化等能力
- **Kuiper**: 现有 C++/CUDA LLM 推理引擎（~17,000 行），支持 LLaMA2/3、Qwen2/3，但架构存在多项问题（宏驱动模型选择、LOG(FATAL) 错误处理、无流式输出、仅 argmax 采样）

### 1.2 目标

从零实现一个**精简、可维护、与 Notix 原生集成**的 LLM 推理引擎：

- 支持 LLaMA3 架构（后续可扩展）
- CPU 优先（验证正确性），预留 GPU 扩展点
- 流式 token 输出（SSE）
- 多轮对话（KV Cache 复用）
- 完善的采样策略（temperature, top-k, top-p, min-p）
- GGUF 模型格式支持

### 1.3 为什么从零实现而非迁移 Kuiper

| 维度 | 迁移 Kuiper | 从零实现 |
|------|------------|---------|
| 代码理解成本 | 需消化 17,000 行业余代码 | 每行都自己写的 |
| 架构适配 | 需要绕过 LOG(FATAL)、同步接口、宏定义 | 原生设计适配 Notix 异步模型 |
| 模型格式 | 自定义 binary 格式（无文档） | 直接支持 GGUF（生态成熟） |
| 算子效率 | 手写 kernel 但未优化（无 tiling/SIMD） | 用 Eigen/OpenBLAS，自动获得 SIMD |
| 最终代码量 | ~17,000 行（含大量冗余） | ~2,500 行核心代码 |

---

## 2. 总体架构设计

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                       Notix HTTP Server                         │
│                  (Boost.Asio event loop × 1)                    │
│                                                                 │
│  Router ──→ Middleware Pipeline ──→ Handler                     │
│    │                                    │                       │
│    │  /chat/completions (SSE)           │                       │
│    │  /chat/completions (non-stream)   │                       │
│    │  /models/list                      │                       │
│    │  /health                           │                       │
│    └────────────────┬───────────────────┘                       │
│                     │                                           │
│              ┌──────▼──────────┐                                │
│              │ InferenceGateway │   ← 新增模块                   │
│              │ - 请求队列       │                                │
│              │ - 会话路由       │                                │
│              │ - SSE 编码       │                                │
│              └──────┬──────────┘                                │
│                     │ boost::asio::post                         │
└─────────────────────┼───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                 Inference Thread Pool                            │
│                   (1 ~ N threads)                                │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Worker 1     │  │ Worker 2     │  │ Worker N     │           │
│  │ LlamaEngine  │  │ LlamaEngine  │  │ LlamaEngine  │           │
│  │ + KV Cache   │  │ + KV Cache   │  │ + KV Cache   │           │
│  │ + Sampler    │  │ + Sampler    │  │ + Sampler    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   libllama.so                             │   │
│  │                                                          │   │
│  │  ┌─────────┐ ┌──────────┐ ┌───────┐ ┌────────────────┐  │   │
│  │  │ Weights │ │Tokenizer │ │ Layers│ │    Sampler     │  │   │
│  │  │ Loader  │ │(SPM/BPE) │ │ Ops   │ │(temp/topk/top)│  │   │
│  │  └─────────┘ └──────────┘ └───────┘ └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 职责 | 新写/复用 |
|------|------|----------|
| **WeightLoader** | GGUF 格式解析、权重内存映射 | 新写 |
| **Tokenizer** | 文本 ↔ token 转换 | 封装 SentencePiece |
| **LayerOps** | RMSNorm, MatMul, RoPE, MHA, SwiGLU | 新写（用 Eigen） |
| **KVCache** | 键值缓存管理、滑动窗口 | 新写 |
| **Sampler** | temperature, top-k, top-p, min-p | 新写 |
| **LlamaEngine** | 组装上述模块，提供 generate() 接口 | 新写 |
| **ChatTemplate** | 对话历史 → 模型 prompt 格式化 | 新写 |
| **InferenceGateway** | Notix 集成：线程池、SSE、会话管理 | 新写 |
| **SessionManager** | 用户会话 → KV Cache 映射 | 复用 Notix 现有 |

### 2.3 数据流

```
HTTP POST /chat/completions
  {"messages": [{"role":"user","content":"你好"}], "stream": true}
       │
       ▼
  Middleware: MethodGuard → SessionAuth → RateLimit
       │
       ▼
  InferenceGateway::handle_chat()
       │
       ├─ 1. 解析 JSON body
       ├─ 2. 获取/创建 InferenceSession (per user)
       ├─ 3. ChatTemplate.apply(messages) → prompt string
       ├─ 4. post 到推理线程池
       │
       ▼
  LlamaEngine::generate_stream(prompt, callback)
       │
       ├─ tokenize(prompt) → token_ids[]
       ├─ for pos in 0..max_tokens:
       │    ├─ embedding(token_ids[pos])
       │    ├─ for layer in layers:
       │    │    ├─ attention_rmsnorm → QKV project → RoPE
       │    │    ├─ MHA with KV cache
       │    │    └─ FFN (SwiGLU)
       │    ├─ final_rmsnorm → lm_head → logits
       │    ├─ sampler.sample(logits) → next_token
       │    ├─ callback(token_string, finished=false)
       │    └─ if next_token == eos: break
       └─ callback("", finished=true)
       │
       ▼
  SSE Response:
    data: {"token": "你"}
    data: {"token": "好"}
    data: {"token": "！"}
    data: [DONE]
```

---

## 3. 推理引擎核心设计

### 3.1 张量计算：Eigen

选用 [Eigen](https://eigen.tuxfamily.org/) 作为线性代数后端：

```cpp
#include <Eigen/Dense>

// 矩阵乘法
Eigen::MatrixXf C = A * B.transpose();

// 逐元素操作
Eigen::MatrixXf result = x.array().square().mean() + eps;
result = x.array() / result.sqrt();

// 自动获得 AVX2/AVX-512 SIMD 加速，无需手写向量化
```

**为什么不用 Armadillo（Kuiper 的做法）：**
- Eigen 是纯头文件，无需链接
- Eigen 的表达式模板比 Armadillo 性能更好
- Eigen 文档和社区更活跃

### 3.2 模型权重结构

#### GGUF 格式概述

GGUF 是 llama.cpp 的标准格式，文件结构：

```
[Header] [Metadata KV Pairs] [Tensor Infos] [Tensor Data]
```

- Header: magic number (`GGUF`), version, tensor count, metadata count
- Metadata: key-value pairs (`general.architecture`, `llama.context_length`, etc.)
- Tensor Infos: name, dimensions, type, offset
- Tensor Data: raw binary blob

**推荐方式**：不手写 GGUF 解析器，使用社区维护的 [gguf.h](https://github.com/ggerganov/llama.cpp/blob/master/gguf.h) 或自写一个精简版解析器（约 300 行）。

#### 权重内存布局

LLaMA3 架构每层需要的权重：

```
每层 (32 层 for 8B):
  blk.{i}.attn_norm.weight       [4096]           RMSNorm
  blk.{i}.attn_q.weight          [4096, 4096]     Query 投影
  blk.{i}.attn_k.weight          [1024, 4096]     Key 投影 (GQA: 8 kv_heads)
  blk.{i}.attn_v.weight          [1024, 4096]     Value 投影
  blk.{i}.attn_output.weight     [4096, 4096]     Output 投影
  blk.{i}.ffn_norm.weight        [4096]           FFN RMSNorm
  blk.{i}.ffn_gate.weight        [14336, 4096]    SwiGLU gate
  blk.{i}.ffn_up.weight          [14336, 4096]    SwiGLU up
  blk.{i}.ffn_down.weight        [4096, 14336]    SwiGLU down

全局:
  token_embd.weight              [128256, 4096]   Token Embedding
  output.weight                  [128256, 4096]   LM Head（通常与 embedding 共享）
  output_norm.weight             [4096]           最终 RMSNorm
```

#### 权重加载伪代码

```cpp
struct LayerWeights {
    Eigen::MatrixXf attn_norm;    // [dim]
    Eigen::MatrixXf attn_q;       // [dim, dim]
    Eigen::MatrixXf attn_k;       // [kv_dim, dim]
    Eigen::MatrixXf attn_v;       // [kv_dim, dim]
    Eigen::MatrixXf attn_output;  // [dim, dim]
    Eigen::MatrixXf ffn_norm;     // [dim]
    Eigen::MatrixXf ffn_gate;     // [hidden_dim, dim]
    Eigen::MatrixXf ffn_up;       // [hidden_dim, dim]
    Eigen::MatrixXf ffn_down;     // [dim, hidden_dim]
};

struct LlamaWeights {
    Eigen::MatrixXf token_embd;   // [vocab_size, dim]
    Eigen::MatrixXf output;       // [vocab_size, dim] (可能共享 token_embd)
    Eigen::MatrixXf output_norm;  // [dim]
    std::vector<LayerWeights> layers;
};

// 加载器
class WeightLoader {
public:
    static std::optional<LlamaWeights> load_gguf(const std::string& path);
    static std::optional<LlamaWeights> load_safetensors(const std::string& path);
};
```

### 3.3 算子系统设计

#### 设计原则

与 Kuiper 不同，**不做过度的 OOP 抽象**。每个算子是一个简单的函数：

```cpp
// 函数式风格 — 输入不可变，输出通过参数返回
namespace ops {

// RMSNorm: output = input / sqrt(mean(input^2) + eps) * weight
void rmsnorm(
    const Eigen::MatrixXf& input,   // [batch, dim]
    const Eigen::VectorXf& weight,  // [dim]
    Eigen::MatrixXf& output,        // [batch, dim]
    float eps = 1e-6
);

// MatMul (Linear): output = input * weight^T
void linear(
    const Eigen::MatrixXf& input,   // [batch, in_features]
    const Eigen::MatrixXf& weight,  // [out_features, in_features]
    Eigen::MatrixXf& output         // [batch, out_features]
);

// RoPE: 旋转位置编码（原地修改）
void rope_inplace(
    Eigen::MatrixXf& q,             // [n_heads, head_dim]
    Eigen::MatrixXf& k,             // [n_kv_heads, head_dim]
    int pos,
    const Eigen::VectorXf& sin_cache,  // [max_seq_len * head_dim/2]
    const Eigen::VectorXf& cos_cache
);

// Scaled Dot-Product Attention (带 KV cache)
void attention(
    const Eigen::MatrixXf& q,       // [n_heads, head_dim]
    const Eigen::MatrixXf& k_cache, // [n_kv_heads, seq_len, head_dim]
    const Eigen::MatrixXf& v_cache, // [n_kv_heads, seq_len, head_dim]
    int n_kv_heads,
    int current_pos,
    Eigen::MatrixXf& output         // [n_heads, head_dim]
);

// SwiGLU FFN
void swiglu_ffn(
    const Eigen::MatrixXf& input,   // [1, dim]
    const LayerWeights& w,
    Eigen::MatrixXf& output         // [1, dim]
);

// Softmax (稳定版本)
void softmax_inplace(Eigen::VectorXf& x);

} // namespace ops
```

#### RMSNorm 实现参考

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot w
$$

```cpp
void ops::rmsnorm(const Eigen::MatrixXf& input, const Eigen::VectorXf& weight,
                  Eigen::MatrixXf& output, float eps) {
    // x^2
    Eigen::MatrixXf sq = input.array().square();
    // mean(x^2) + eps
    float rms = std::sqrt(sq.mean() + eps);
    // x / rms * weight
    output = (input.array() / rms) * weight.transpose().array();
}
```

> **验证方法**：对比 HuggingFace `transformers.models.llama.modeling_llama.LlamaRMSNorm.forward()` 输出。

#### RoPE 实现参考

```cpp
void ops::rope_inplace(Eigen::MatrixXf& q, Eigen::MatrixXf& k, int pos,
                       const Eigen::VectorXf& sin_cache, const Eigen::VectorXf& cos_cache) {
    int head_dim = q.cols();
    int rot_dim = head_dim;  // LLaMA3: 全部维度做旋转

    for (int i = 0; i < rot_dim / 2; i++) {
        float sin_val = sin_cache(pos * rot_dim / 2 + i);
        float cos_val = cos_cache(pos * rot_dim / 2 + i);

        // 对 Q 的每对 (2i, 2i+1) 做旋转
        for (int h = 0; h < q.rows(); h++) {
            float x0 = q(h, 2 * i);
            float x1 = q(h, 2 * i + 1);
            q(h, 2 * i)     = x0 * cos_val - x1 * sin_val;
            q(h, 2 * i + 1) = x0 * sin_val + x1 * cos_val;
        }
        // 对 K 同理
        for (int h = 0; h < k.rows(); h++) {
            float x0 = k(h, 2 * i);
            float x1 = k(h, 2 * i + 1);
            k(h, 2 * i)     = x0 * cos_val - x1 * sin_val;
            k(h, 2 * i + 1) = x0 * sin_val + x1 * cos_val;
        }
    }
}
```

> **性能提示**：上述是朴素实现（每个位置标量操作）。后续优化可以用 Eigen 的向量化操作一次性处理所有 head。

#### MHA with KV Cache 实现参考

```cpp
void ops::attention(
    const Eigen::MatrixXf& q,         // [n_heads, head_dim]
    const Eigen::MatrixXf& k_cache,   // [n_kv_heads, seq_len, head_dim]
    const Eigen::MatrixXf& v_cache,   // [n_kv_heads, seq_len, head_dim]
    int n_kv_heads,
    int current_pos,
    Eigen::MatrixXf& output           // [n_heads, head_dim]
) {
    int n_heads = q.rows();
    int head_dim = q.cols();
    int n_groups = n_heads / n_kv_heads;  // GQA 组数
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    output.setZero();

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_groups;  // GQA: 多个 Q head 共享一组 KV

        // 当前 head 的 query
        Eigen::VectorXf q_head = q.row(h);

        // 计算 attention scores: q @ k^T / sqrt(d)
        Eigen::VectorXf scores(current_pos + 1);
        for (int t = 0; t <= current_pos; t++) {
            scores(t) = q_head.dot(k_cache.slice(kv_h, t)) * scale;
        }

        // softmax
        softmax_inplace(scores);

        // 加权求和: sum(scores[t] * v[t])
        for (int t = 0; t <= current_pos; t++) {
            output.row(h) += scores(t) * v_cache.slice(kv_h, t);
        }
    }
}
```

> **参考 Kuiper**: `kuiper/source/op/kernels/cpu/mha_kernel.cpp` — 同样的逻辑，但 Kuiper 版本在循环内频繁创建临时 Tensor 对象，效率更低。

### 3.4 KV Cache 设计

```cpp
class KVCache {
public:
    KVCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim)
        : n_layers_(n_layers), n_kv_heads_(n_kv_heads),
          max_seq_len_(max_seq_len), head_dim_(head_dim), current_len_(0)
    {
        // 预分配: [n_layers, n_kv_heads, max_seq_len, head_dim]
        k_cache_.resize(n_layers);
        v_cache_.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            k_cache_[i] = Eigen::MatrixXf::Zero(n_kv_heads * max_seq_len, head_dim);
            v_cache_[i] = Eigen::MatrixXf::Zero(n_kv_heads * max_seq_len, head_dim);
        }
    }

    // 写入新 token 的 KV
    void write(int layer_idx, int kv_head, int pos,
               const Eigen::VectorXf& k, const Eigen::VectorXf& v) {
        k_cache_[layer_idx].row(kv_head * max_seq_len_ + pos) = k;
        v_cache_[layer_idx].row(kv_head * max_seq_len_ + pos) = v;
    }

    // 读取某个 head 的前 seq_len 个 KV
    Eigen::Block<Eigen::MatrixXf> get_k(int layer_idx, int kv_head, int seq_len) {
        return k_cache_[layer_idx].block(kv_head * max_seq_len_, 0, seq_len, head_dim_);
    }

    // 截断（多轮对话时可能需要）
    void truncate(int new_len) {
        current_len_ = new_len;
    }

    int current_len() const { return current_len_; }

private:
    int n_layers_, n_kv_heads_, max_seq_len_, head_dim_, current_len_;
    std::vector<Eigen::MatrixXf> k_cache_;
    std::vector<Eigen::MatrixXf> v_cache_;
};
```

### 3.5 采样器设计

```cpp
struct GenerationConfig {
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    float min_p = 0.05f;
    float repetition_penalty = 1.1f;
    int seed = -1;  // -1 = random
    std::vector<std::string> stop_strings;
};

class Sampler {
public:
    Sampler(const GenerationConfig& config);

    // 从 logits 中采样下一个 token
    int sample(const Eigen::VectorXf& logits,
               const std::vector<int>& previous_tokens = {});

private:
    void apply_temperature(Eigen::VectorXf& logits);
    void apply_top_k(Eigen::VectorXf& logits);
    void apply_top_p(Eigen::VectorXf& logits);
    void apply_min_p(Eigen::VectorXf& logits);
    void apply_repetition_penalty(Eigen::VectorXf& logits,
                                   const std::vector<int>& previous_tokens);
    int sample_multinomial(const Eigen::VectorXf& probs);

    GenerationConfig config_;
    std::mt19937 rng_;
};
```

### 3.6 LlamaEngine 主接口

```cpp
class LlamaEngine {
public:
    // 生命周期
    bool load(const std::string& model_path,   // GGUF 文件路径
              const std::string& tokenizer_path);
    bool is_loaded() const;
    void unload();

    // 推理接口
    struct TokenCallback {
        std::function<void(const std::string& token, bool is_end)> on_token;
    };

    void generate_stream(
        const std::string& prompt,
        TokenCallback callback,
        const GenerationConfig& config = {}
    );

    std::string generate_sync(
        const std::string& prompt,
        const GenerationConfig& config = {}
    );

    // 模型信息
    const ModelInfo& info() const;

private:
    // 单步前向传播
    int forward_step(int token, int pos, KVCache& kv_cache);

    // 内部状态
    ModelInfo info_;
    ModelConfig model_config_;
    LlamaWeights weights_;
    std::unique_ptr<SentencePieceProcessor> tokenizer_;
    Sampler sampler_;

    // RoPE 预计算缓存
    Eigen::VectorXf sin_cache_;
    Eigen::VectorXf cos_cache_;

    // 中间激活缓冲区（避免每步分配）
    Eigen::MatrixXf hidden_state_;   // [1, dim]
    Eigen::MatrixXf q_buffer_;       // [n_heads, head_dim]
    Eigen::MatrixXf k_buffer_;       // [n_kv_heads, head_dim]
    Eigen::MatrixXf v_buffer_;       // [n_kv_heads, head_dim]
    Eigen::MatrixXf attn_buffer_;    // [n_heads, head_dim]
    Eigen::VectorXf logits_buffer_;  // [vocab_size]
};
```

### 3.7 Chat Template

```cpp
class ChatTemplate {
public:
    // LLaMA3 格式:
    // <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
    // {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
    // {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

    struct Message {
        std::string role;     // "system", "user", "assistant"
        std::string content;
    };

    std::string apply(const std::vector<Message>& messages,
                      bool add_generation_prompt = true);

    // 从 tokenizer_config.json 加载模板
    static std::optional<ChatTemplate> from_file(const std::string& path);

private:
    std::string bos_token_ = "<|begin_of_text|>";
    std::string eos_token_ = "<|eot_id|>";
    struct HeaderTokens {
        std::string start = "<|start_header_id|>";
        std::string end = "<|end_header_id|>";
    };
    std::map<std::string, HeaderTokens> role_headers_;
};
```

---

## 4. Notix 集成方案

### 4.1 InferenceGateway

```cpp
// notix/httpserver/inc/inference/inference_gateway.h

class InferenceGateway {
public:
    InferenceGateway(boost::asio::io_context& ioc, size_t num_workers);

    // 异步推理请求
    void handle_chat_completions(
        const Json::Value& request_body,
        std::function<void(const std::string& sse_data)> send_sse,
        std::function<void(const Json::Value& response)> on_complete,
        std::function<void(int code, const std::string& msg)> on_error
    );

    // 获取可用模型列表
    Json::Value list_models() const;

private:
    // 推理线程池
    boost::asio::thread_pool inference_pool_;
    std::vector<std::unique_ptr<InferenceWorker>> workers_;

    // 会话管理（复用 KV Cache）
    std::mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<InferenceSession>> sessions_;
};
```

### 4.2 线程安全模型

每个 worker 线程持有一个独立的 `LlamaEngine` 实例。会话在 worker 之间不迁移。

```
Worker 1: Engine1 + { session_A, session_C }
Worker 2: Engine2 + { session_B }
Worker 3: Engine3 + { session_D, session_E }
```

**显存策略**：如果未来上 GPU，每 worker 共享同一个模型权重（只读），仅 KV Cache 独立分配。

### 4.3 SSE 流式端点

```
POST /chat/completions
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "你好"}],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 256
}

--- Response (text/event-stream) ---
data: {"choices":[{"delta":{"content":"你"},"index":0}]}

data: {"choices":[{"delta":{"content":"好"},"index":0}]}

data: {"choices":[{"delta":{"content":"！"},"index":0}]}

data: {"choices":[{"delta":{"content":""},"finish_reason":"stop","index":0}]}

data: [DONE]
```

对标 OpenAI Chat Completions API 格式，前端可以直接用 `openai` npm 包。

### 4.4 路由注册

```cpp
// 在 router_rules.h 中新增
enum class static_route {
    // ... 已有 ...
    chat_completions,   // POST /chat/completions
    models_list,        // GET  /models
    health,             // GET  /health
};

// 在 router_rules.cpp 中注册
void register_inference_routes(Router& router) {
    router.add_static_route("/chat/completions", static_route::chat_completions);
    router.add_static_route("/models", static_route::models_list);
    router.add_static_route("/health", static_route::health);
}
```

---

## 5. 实施路线图

### Phase 0: Python 原型验证（2-3 天）

在写 C++ 之前，用 NumPy 实现完整的前向传播：

```python
# python_prototype/llama_forward.py

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LlamaForwardNumpy:
    """纯 NumPy 实现的 LLaMA3 forward pass"""
    def __init__(self, hf_model_path):
        # 从 HuggingFace 模型加载权重并转为 NumPy
        self.weights = self._load_weights(hf_model_path)
        self.config = self._load_config(hf_model_path)

    def forward(self, token_ids, past_kv=None):
        """单步前向传播"""
        # Embedding
        # For each layer:
        #   RMSNorm → QKV → RoPE → Attention (with KV cache) → Output
        #   RMSNorm → SwiGLU FFN → Residual
        # Final RMSNorm → LM Head → logits
        pass

    def generate(self, prompt, max_tokens=50, temperature=0.7):
        """生成循环"""
        pass

# 验证：与 HuggingFace 模型对比每个中间层的输出
def verify_against_hf(np_model, hf_model, test_prompts):
    for prompt in test_prompts:
        np_output = np_model.generate(prompt)
        hf_output = hf_model.generate(prompt)
        assert np_output == hf_output, f"Mismatch for prompt: {prompt}"
    print("All outputs match! ✓")
```

**这一步不可跳过**——它验证了你对每个算子的理解，并且 NumPy 的调试体验比 C++ 快 10 倍。

### Phase 1: 引擎骨架（1 周）

1. **CMake 项目搭建**
   - 创建 `llama-engine/` 目录结构
   - CMakeLists.txt（C++17, Eigen, SentencePiece, gguf 解析）
   - 单元测试框架（Catch2 或 doctest）

2. **模型配置 & 信息结构**
   - `ModelConfig` struct
   - `ModelInfo` struct

3. **GGUF 加载器**
   - 解析 GGUF header 和 metadata
   - 读取 tensor info
   - mmap 权重文件

4. **Tokenizer 封装**
   - 封装 SentencePiece Processor
   - encode / decode / vocab_size 接口

### Phase 2: 算子实现（1-2 周）

按依赖顺序实现，每完成一个立即与 HF 输出对比验证：

1. **RMSNorm** — 最简单的，先写
2. **Linear (MatMul)** — 用 Eigen::gemm
3. **RoPE** — 需要预计算 sin/cos cache
4. **SwiGLU FFN** — 三个 Linear + SiLU 激活
5. **Attention + KV Cache** — 最复杂的，需要仔细调
6. **完整单层测试** — 一层 LLaMA DecoderLayer 端到端

### Phase 3: 推理循环 & 采样（3-5 天）

1. **Generation loop** — 贪心解码先跑通
2. **Temparature sampler** — 最简单的随机采样
3. **Top-K sampler**
4. **Top-P (nucleus) sampler**
5. **Min-P sampler**
6. **Repetition penalty**
7. **Stop strings 检测**

### Phase 4: Notix 集成（1 周）

1. **InferenceWorker 线程池**
2. **InferenceSession 管理**
3. **`/chat/completions` SSE 端点**
4. **ChatTemplate 格式化**
5. **前端 app.html 适配流式输出**

### Phase 5: 性能优化（按需，1-2 周）

1. **OpenBLAS 替换 Eigen** — 大矩阵乘法加速 3-5×
2. **权重预打包** — 避免每步重复 packing
3. **KV Cache 量化** — KV 用 FP16/INT8 节省内存
4. **多线程 batch** — 同时处理多个用户请求

---

## 6. Kuiper 参考索引

以下是 Kuiper 代码中值得参考的部分（按重要性排序）：

### 6.1 核心推理流程

| 文件 | 行数 | 参考价值 | 说明 |
|------|------|---------|------|
| `demo/main.cpp:6-47` | 42 | ⭐⭐⭐⭐⭐ | 完整 `generate()` 循环，最直观的参考 |
| `kuiper/source/model/model.cpp` | 260 | ⭐⭐⭐⭐ | `Model` 基类：init/read_model_file/create_encode_layer |
| `kuiper/source/model/llama3.cpp:147-167` | 21 | ⭐⭐⭐⭐⭐ | `forward()` — 逐层处理的主循环 |
| `kuiper/source/model/llama3.cpp:600-720` | 121 | ⭐⭐⭐⭐⭐ | `attention_rms` / `attention_qkv` / `attention_mha` / `feed_forward` — 单层内部逻辑 |
| `kuiper/source/model/llama3.cpp:733-745` | 13 | ⭐⭐⭐⭐ | `post_processing` — logits → sampler |

### 6.2 算子 CPU Kernel

| 文件 | 行数 | 参考价值 | 说明 |
|------|------|---------|------|
| `kuiper/source/op/kernels/cpu/rmsnorm_kernel.cpp` | ~20 | ⭐⭐⭐⭐ | RMSNorm CPU 实现 |
| `kuiper/source/op/kernels/cpu/rope_kernel.cpp` | ~30 | ⭐⭐⭐⭐ | RoPE CPU 实现 |
| `kuiper/source/op/kernels/cpu/mha_kernel.cpp` | 61 | ⭐⭐⭐ | MHA CPU 实现（效率低但逻辑正确） |
| `kuiper/source/op/kernels/cpu/swiglu_kernel.cpp` | ~20 | ⭐⭐⭐⭐ | SwiGLU CPU 实现 |
| `kuiper/source/op/kernels/cpu/softmax_kernel.cpp` | ~20 | ⭐⭐⭐ | Softmax CPU 实现 |
| `kuiper/source/op/kernels/cpu/matmul_kernel.cpp` | ~30 | ⭐⭐⭐ | MatMul CPU 实现（用 Armadillo） |
| `kuiper/source/op/kernels/cpu/add_kernel.cpp` | ~10 | ⭐⭐ | 向量加法（残差连接用） |

### 6.3 数据结构

| 文件 | 参考价值 | 说明 |
|------|---------|------|
| `kuiper/include/tensor/tensor.h` | ⭐⭐⭐ | Tensor 类的接口设计参考 |
| `kuiper/include/base/buffer.h` | ⭐⭐⭐ | 内存管理抽象 |
| `kuiper/include/base/base.h` | ⭐⭐⭐ | Status/StatusCode 错误处理模式 |
| `kuiper/include/op/layer.h` | ⭐⭐ | Layer 类型枚举（但别学它的继承层次） |

### 6.4 要注意的坑（Kuiper 的问题）

以下是 Kuiper 中应该**避免**的做法：

1. **`LOG(FATAL)` 终止进程** → 改用 `std::expected` 或异常
2. **`const_cast` 修改输出** → 函数参数明确标注 mutable
3. **5 个 `forward` 重载** → 用一个 `std::vector<Tensor>` 传参
4. **权重偏移硬编码** → 用 name-based lookup
5. **宏驱动模型选择** → 用注册/工厂模式
6. **C 风格 `va_list`** → 用 `std::initializer_list`
7. **无 RAII 的文件句柄** → 用 `std::unique_ptr<FILE, decltype(&fclose)>`

---

## 7. 测试与验证策略

### 7.1 单元测试层级

```
Level 1: 算子级（每个算子独立测试）
  ├─ rmsnorm_test: 对比 HF output
  ├─ linear_test: 对比 Eigen @ weight vs PyTorch F.linear
  ├─ rope_test: 对比旋转前后向量模长不变
  ├─ attention_test: 对比 causal mask 正确性
  └─ swiglu_test: 对比 SiLU(x*w1) * (x*w3) vs HF output

Level 2: 单层级（一层 DecoderLayer）
  └─ decoder_layer_test: 输入相同 hidden_state，对比 HF 输出

Level 3: 模型级（端到端）
  ├─ forward_test: 相同输入 token，对比 HF 的 logits
  └─ generate_test: 相同 prompt，对比 HF 的生成结果

Level 4: 集成级
  ├─ http_endpoint_test: HTTP 请求 → 推理 → SSE 响应
  └─ multi_turn_test: 多轮对话 KV Cache 正确性
```

### 7.2 测试用模型

| 模型 | 大小 | 用途 |
|------|------|------|
| [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | ~4GB FP32 | 快速迭代测试 |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | ~2GB FP32 | 更小的测试模型 |
| [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | ~7GB FP32 | 质量测试 |

### 7.3 精度验证脚本

```python
# scripts/verify_forward.py
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def verify_forward_pass(cpp_logits_path, hf_model_path, test_input):
    """对比 C++ 引擎输出的 logits 与 HF 输出"""
    # 读取 C++ 引擎 dump 的 logits
    cpp_logits = np.fromfile(cpp_logits_path, dtype=np.float32)

    # HF forward
    model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    with torch.no_grad():
        hf_output = model(torch.tensor([test_input]))
        hf_logits = hf_output.logits[0, -1].numpy()

    # 对比
    diff = np.abs(cpp_logits - hf_logits)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"All close (1e-4): {np.allclose(cpp_logits, hf_logits, atol=1e-4)}")
```

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **数值精度不匹配** | 高 | 高 | 每个算子写完立即与 HF 输出对比；Python 原型先行验证 |
| **CPU 推理太慢** | 高 | 中 | 7B 纯 CPU ~2-5 tok/s；先用小模型测；后续换 OpenBLAS |
| **内存占用过大** | 中 | 中 | 7B FP32 ≈ 28GB；测试时用 TinyLlama (4GB)；正式部署上 FP16 |
| **GGUF 格式解析 Bug** | 中 | 高 | 先用自定义简单 binary 格式做 MVP，GGUF 支持作为 Phase 2 |
| **KV Cache 越界/错位** | 中 | 高 | 多写 assert；用单轮对话先验证；Python 原型先调通 |
| **Eigen 性能瓶颈** | 低 | 中 | Eigen 自带 SIMD；不够时换 OpenBLAS（API 兼容） |
| **多线程竞争** | 中 | 中 | 每 worker 独立 Engine 实例，避免共享状态 |

---

## 9. 附录

### 9.1 推荐目录结构

```
llama-engine/
├── CMakeLists.txt
├── include/
│   └── llama/
│       ├── engine.h           # LlamaEngine 主接口
│       ├── config.h           # ModelConfig, GenerationConfig
│       ├── weights.h          # LlamaWeights, LayerWeights
│       ├── ops.h              # rmsnorm, linear, rope, attention, swiglu
│       ├── kv_cache.h         # KVCache 类
│       ├── sampler.h          # Sampler 类
│       ├── tokenizer.h        # Tokenizer 封装
│       ├── chat_template.h    # ChatTemplate
│       └── gguf_loader.h      # GGUF 格式加载器
├── src/
│   ├── engine.cpp
│   ├── ops.cpp
│   ├── kv_cache.cpp
│   ├── sampler.cpp
│   ├── tokenizer.cpp
│   ├── chat_template.cpp
│   └── gguf_loader.cpp
├── test/
│   ├── CMakeLists.txt
│   ├── test_rmsnorm.cpp
│   ├── test_attention.cpp
│   ├── test_rope.cpp
│   ├── test_sampler.cpp
│   └── test_engine.cpp
└── python_prototype/
    ├── llama_forward.py        # NumPy 前向传播
    └── verify_against_hf.py    # 与 HF 对比验证
```

### 9.2 关键依赖

| 库 | 用途 | 版本 | License |
|----|------|------|---------|
| Eigen 3.4 | 线性代数 | ≥ 3.4.0 | MPL2 |
| SentencePiece | Tokenizer | ≥ 0.2.0 | Apache 2.0 |
| Boost (Asio, Beast) | HTTP 服务（Notix 已用） | ≥ 1.80 | BSL |
| Catch2 / doctest | 单元测试 | latest | BSL / MIT |
| nlohmann/json | JSON 解析（Notix 侧） | ≥ 3.11 | MIT |

**可选依赖：**

| 库 | 用途 | 何时引入 |
|----|------|---------|
| OpenBLAS | 加速 MatMul | Eigen 性能不够时 |
| CUDA/cuBLAS | GPU 推理 | 需要 GPU 部署时 |

### 9.3 参考资料

| 资源 | 说明 |
|------|------|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | LLaMA 架构论文 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | RoPE 论文 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | RMSNorm 论文 |
| [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) | GGUF 格式文档 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | 最成熟的 C++ LLM 推理工程参考 |
| [HuggingFace LLaMA Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | 算子的权威 Python 实现 |
| [Kuiper KuiperLLama](https://github.com/wong-1994/KuiperLLama) | C++ 推理引擎参考实现（已有代码） |

### 9.4 LLaMA3 8B 关键参数

```
dim:            4096
n_layers:       32
n_heads:        32
n_kv_heads:     8        (GQA: 4 query heads per KV head)
head_dim:       128
hidden_dim:     14336    (FFN intermediate size)
vocab_size:     128256
max_seq_len:    8192
rope_theta:     500000.0
norm_eps:       1e-5
```

---

> **下一步行动**：Phase 0 — 用 Python/NumPy 写一个完整的前向传播，验证对 Transformer 每个算子的理解，对比 HuggingFace 输出。
