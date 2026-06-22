# Notix LLM 推理框架两周实现计划

> 日期: 2026-06-22  
> 周期: 14 天  
> 目标: 在 Notix 中完成一个可展示、可测试、可说明性能路径的轻量级 decoder-only LLM 推理框架。

---

## 一、总体策略

当前 Notix 的 `llama-engine` 已经完成 LLaMA decoder block 的 C++ 正确性原型，适合作为推理框架的接口层和服务集成基础；主仓库 `kuiper` 已经具备 LLaMA/Qwen、CUDA kernel、KV Cache、Int8 group-wise 量化路径等能力，适合作为性能后端参考或迁移来源。

两周内不建议从 Notix 的 Eigen CPU 实现直接硬优化到 8B 高性能推理。更稳妥的路线是：

1. Notix 保留清晰、可维护的 engine facade 和 HTTP/SSE 集成。
2. 复用或迁移 Kuiper 中已有的 CUDA、Int8、Qwen/LLaMA 相关能力。
3. 先做出 LLaMA/TinyLlama 正确性闭环，再扩展到 8B Int8 benchmark。
4. 对多模态只实现接口预留，不承诺真实 LLaVA/Qwen-VL 推理。

---

## 二、目标拆解

| 模块 | 两周目标 | 验收标准 | 风险 |
|------|----------|----------|------|
| 推理框架 | 统一 `Engine` / `Backend` / `ModelExecutor` 接口 | CPU backend 可跑通，CUDA backend 可切换 | 中 |
| 模型支持 | LLaMA 优先，Qwen 复用 Kuiper 路径 | 至少 LLaMA Tiny 模型端到端生成；Qwen 有接口或 demo | 中 |
| KV Cache | 保留 autoregressive decoding cache | prompt prefill + decode 位置正确 | 低 |
| Tokenizer | 接入 SentencePiece/BPE 或复用 Kuiper tokenizer | 输入文本可 encode，输出 token 可 decode | 中 |
| CUDA 算子 | 接入 RMSNorm、MatMul、Attention CUDA kernel | GPU 路径能执行并通过基础对比 | 高 |
| Int8 量化 | 支持 group-wise Int8 权重与动态反量化 MatMul | 显存占用下降，有 logits/perplexity 对比 | 高 |
| 性能分析 | 使用 Nsight Systems 做 token-level profile | 输出 profile 截图/表格/瓶颈说明 | 中 |
| 服务集成 | Notix 提供 `/chat/completions` SSE 流式接口 | 前端或 curl 能看到流式 token | 中 |
| 扩展接口 | 预留 Vision Encoder 与 multimodal embedding | 有接口、数据结构和 stub 实现 | 低 |

---

## 三、14 天执行计划

| 天数 | 任务 | 产出 | 验收方式 |
|------|------|------|----------|
| Day 1 | 梳理 Notix `llama-engine` 与 Kuiper 能力边界，确定复用文件清单 | 技术决策记录、迁移清单 | 能明确哪些代码留在 Notix，哪些从 Kuiper 迁移 |
| Day 2 | 重构 Notix engine 接口，抽象 `InferenceEngine`、`ModelExecutor`、`Backend` | 新接口头文件与 CPU backend 适配 | 原有 TinyLLaMA 前向测试不破坏 |
| Day 3 | 补齐 `generate()`、`generate_stream()` 的 token id 返回逻辑，去掉占位实现 | 可用 generation API | 单元测试能拿到生成 token 序列 |
| Day 4 | 集成 tokenizer，优先复用 Kuiper tokenizer 或封装 SentencePiece/BPE | `Tokenizer` 接口与 encode/decode 实现 | 文本输入可以变成 token，token 可以解码为文本 |
| Day 5 | 接入 Notix HTTP server，新增 `/chat/completions` 非流式接口 | HTTP 推理 demo | `curl` 能返回 JSON 回复 |
| Day 6 | 实现 SSE 流式接口和基础前端适配 | `/chat/completions` stream 模式 | `curl -N` 或页面能看到 token 流 |
| Day 7 | 从 Kuiper 迁移 CUDA backend 基础设施：device allocator、tensor、stream 管理 | CUDA backend skeleton | CUDA 编译通过，能完成简单 tensor op |
| Day 8 | 接入 CUDA RMSNorm、RoPE、MatMul 基础算子 | GPU op 测试 | CPU/GPU 单算子误差在阈值内 |
| Day 9 | 接入 CUDA Attention + KV Cache 路径 | GPU decoder layer demo | 单层或小模型 forward 与 CPU 对齐 |
| Day 10 | 跑通 LLaMA CUDA decode pipeline | GPU 生成 demo | 小模型 GPU 端到端生成成功 |
| Day 11 | 接入 Int8 group-wise 权重格式与动态反量化 MatMul | Int8 MatMul 路径 | fp32/int8 logits diff 或 top-k 对比通过 |
| Day 12 | 准备 8B Int8 benchmark，记录 token/s、显存、首 token 延迟 | benchmark 表格 | RTX 4060 上有可复现实测数据 |
| Day 13 | 使用 Nsight Systems profile，优化明显瓶颈：同步、拷贝、临时分配 | Nsight 报告与优化记录 | 有优化前后对比 |
| Day 14 | 整理最终文档、架构图、演示脚本、降级说明 | 项目总结与 demo checklist | 可以完整演示和答辩 |

---

## 四、阶段里程碑

### Milestone 1: 可用 CPU 推理闭环

时间: Day 1 - Day 4

必须完成：

- `generate()` 返回真实 token id。
- `generate_stream()` 通过 callback 输出真实 token。
- tokenizer 完成 encode/decode。
- TinyLLaMA 或更小模型可以从文本输入生成文本输出。

### Milestone 2: Notix 服务闭环

时间: Day 5 - Day 6

必须完成：

- `/chat/completions` 支持 JSON 请求。
- 支持非流式和 SSE 流式两种返回。
- 前端或 curl 能直接验证。

### Milestone 3: CUDA 后端闭环

时间: Day 7 - Day 10

必须完成：

- CUDA backend 可通过配置切换。
- RMSNorm、MatMul、Attention 至少有基础 CUDA 实现。
- 小模型 GPU forward/generate 能跑通。

### Milestone 4: Int8 与性能数据

时间: Day 11 - Day 13

必须完成：

- Int8 group-wise 权重加载。
- 动态反量化 MatMul。
- benchmark 输出 token/s、显存占用、首 token 延迟。
- Nsight Systems 有 profile 证据。

### Milestone 5: 展示与扩展性

时间: Day 14

必须完成：

- 统一架构图。
- Vision Encoder / multimodal embedding 接口 stub。
- README 或 summary 文档更新。
- demo 命令一键可复现。

---

## 五、性能目标与口径

优先级从高到低：

| 指标 | 基准目标 | 理想目标 | 说明 |
|------|----------|----------|------|
| 小模型 CPU 生成 | 能稳定生成 | 输出与 HF top-k 基本一致 | 正确性优先 |
| 小模型 GPU 生成 | 能端到端跑通 | GPU 明显快于 CPU | 验证 CUDA backend |
| 8B Int8 显存 | 明显低于 fp32 | 降低约 40% 或以上 | 以实测 `nvidia-smi` 为准 |
| 8B Int8 速度 | 有可复现实测 | RTX 4060 约 22 token/s | 不提前写死，按实测报告 |
| 精度损失 | logits/top-k/perplexity 小集对比 | 损失 < 1% | 需明确数据集和评价指标 |

---

## 六、降级方案

如果 CUDA 路径进展不顺：

- 保留 Notix CPU 正确性闭环。
- 展示 Kuiper CUDA kernel 的单算子 benchmark。
- 将完整 GPU decode 标记为 experimental backend。

如果 8B 模型加载或显存不足：

- 改用 1B/3B 模型完成端到端演示。
- 8B 只展示模型格式、量化权重统计和部分 layer benchmark。

如果 Int8 精度无法达到目标：

- 改为报告 logits diff、top-k match、perplexity 小集结果。
- 明确量化策略为 weight-only group-wise Int8。
- 将 `<1%` 作为优化目标，而不是最终承诺。

如果 Qwen 支持时间不足：

- LLaMA 作为主路径。
- Qwen 通过统一 config、tokenizer、model registry 预留接口。
- 复用 Kuiper Qwen2/Qwen3 代码作为后续迁移依据。

---

## 七、最终交付清单

代码交付：

- `llama-engine` 统一推理接口。
- CPU backend。
- CUDA backend prototype。
- tokenizer wrapper。
- Int8 group-wise quantized matmul path。
- Notix `/chat/completions` 与 SSE endpoint。
- Vision Encoder / multimodal embedding stub。

测试交付：

- 算子单元测试。
- CPU/GPU 误差对比测试。
- tokenizer encode/decode 测试。
- generation smoke test。
- HTTP endpoint smoke test。

文档交付：

- 架构图。
- benchmark 表格。
- Nsight Systems profile 说明。
- 精度验证说明。
- 风险与后续计划。

演示交付：

- 本地启动脚本。
- curl 示例。
- 前端流式对话 demo。
- benchmark 复现命令。

---

## 八、建议最终项目表述

建议最终对外表述为：

> 设计并实现轻量级 decoder-only LLM 推理框架，支持 LLaMA/Qwen 系列模型的 autoregressive decoding 与 KV Cache；实现 CPU Eigen 后端与 CUDA 后端原型，针对 RMSNorm、MatMul、Attention 编写 CUDA kernel，并支持 Int8 group-wise 权重量化与动态反量化推理；通过 Nsight Systems 分析 token-level 推理瓶颈，在 RTX 4060 上完成 8B Int8 模型推理性能测试；抽象模型执行接口，预留 Vision Encoder 与多模态 embedding 扩展点。

这个表述比直接承诺固定 token/s 和固定精度损失更稳，也更符合两周实现周期的工程现实。
