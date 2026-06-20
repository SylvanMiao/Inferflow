# Phase 0：LLaMA 前向传播 Python/NumPy 参考实现

## 目录结构

| 文件 | 作用 |
|------|------|
| `llama_forward.py` | NumPy 实现的完整 LLaMA 前向传播与自回归生成 |
| `verify_against_hf.py` | 对比脚本 — 将上述实现与 HuggingFace transformers 逐层验证 |
| `requirements.txt` | Python 依赖（numpy, torch, transformers, sentencepiece） |

---

## 两个脚本分别干什么

### `llama_forward.py` — 推理引擎的 Python 实现（"标准答案"）

这个文件实现了 LLaMA 模型推理所需的**全部模块**，每个函数对应后续 C++ 引擎中的一个模块：

- **LlamaConfig** — 模型超参数（维度、层数、注意力头数等）
- **LlamaWeights / LayerWeights** — 所有权重矩阵的容器
- **KVCache** — 键值缓存，支持多轮对话复用
- **算子**：`rmsnorm()`、`apply_rotary_emb()`（RoPE）、`attention_forward()`（含 GQA）、`feed_forward()`（SwiGLU）
- **LlamaModel** — 完整 Transformer 模型（22 层 × [Attention + FFN]）
- **LlamaForCausalLM** — 带 LM Head 的因果语言模型 + `generate()` 自回归生成循环
- **load_weights_from_hf()** — 从 HuggingFace 模型目录加载权重

直接运行 `python llama_forward.py` 会执行**7 项独立测试**（用随机权重验证 shape 正确性），不需要下载任何模型。

### `verify_against_hf.py` — 与 HuggingFace 的逐层对比验证

这个脚本下载一个真实的 TinyLlama 模型，然后：

1. **配置验证** — 确认我们解析的模型参数和 HF 完全一致
2. **RMSNorm 验证** — 对同一个输入，对比算子和 HF 的输出数值
3. **RoPE 验证** — 验证旋转位置编码的数学性质（pos=0 是恒等变换、范数不变）
4. **前向传播验证** — 同一个 token 序列，逐位置对比 logits 的 top-5 是否匹配
5. **生成验证** — 同一个 prompt，对比生成的 token 序列是否一字不差

最终输出一个 PASS/FAIL 汇总表。

---

## 使用方法

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 运行独立测试（不需下载模型）

```powershell
python llama_forward.py
```

输出 7 项测试结果。用**随机权重**运行，仅验证结构正确性（shape、范数不变性等），不验证数值精度。

### 3. 与 HuggingFace 对比验证（需要下载模型）

```powershell
# 下载模型到本地（首次约 2.2GB）
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ../model/TinyLlama-1.1B-Chat-v1.0

# 运行验证
python verify_against_hf.py --model ../model/TinyLlama-1.1B-Chat-v1.0

# 跳过生成测试（更快）
python verify_against_hf.py --model ../model/TinyLlama-1.1B-Chat-v1.0 --quick

# 自定义测试 prompt
python verify_against_hf.py --model ../model/TinyLlama-1.1B-Chat-v1.0 --prompt "你好世界"
```

---

## 验证结果（TinyLlama-1.1B）

```
Config Verification:   ✓ (7/7 参数匹配)
RMSNorm Verification:  ✓ (max_diff = 0.00000001)
RoPE Verification:     ✓ (恒等性 + 范数不变)
Forward Pass:          ✓ (8 个 token，top-5 全部一致)
Generation:            ✓ (15 token 生成，和 HF 一字不差)
```

调试过程中发现并修复了三个关键问题：

| 问题 | 原因 | 影响 |
|------|------|------|
| RoPE 维度配对错误 | 用 `(0↔1, 2↔3…)` 而非 HF 的 `(0↔32, 1↔33…)` | pos>0 即跑偏 |
| `rope_theta` 默认值错误 | 默认 500000（LLaMA3）但 TinyLlama 用 10000 | cos/sin 数值全错 |
| 局部变量 `dim` 未定义 | `feed_forward` 函数内缺少 `dim = config.dim` | 运行时报错 |

---

## NumPy → C++ 对照表

每个 Python 模块在 C++ 引擎中都有 1:1 对应：

```
llama_forward.py                         llama-engine/include/llama/
─────────────────────────────────        ────────────────────────────
LlamaConfig                              config.h      (已创建)
LlamaWeights / LayerWeights              weights.h
KVCache                                  kv_cache.h
rmsnorm()                                ops.h :: rmsnorm()
apply_rotary_emb()                       ops.h :: rope()
attention_forward()                      ops.h :: attention()
feed_forward()                           ops.h :: swiglu_ffn()
LlamaModel.forward(token, pos, cache)    engine.h :: forward_step()
LlamaForCausalLM.generate(prompt, ...)   engine.h :: generate()
load_weights_from_hf()                   gguf_loader.h
```

## 设计决策（带入 C++ 阶段）

1. **纯函数式算子** — 输入不可变，通过返回值输出（便于测试和调试）
2. **全程 FP32** — 不引入混合精度，降低初始调试复杂度
3. **行优先内存布局** — 和 Eigen 默认一致
4. **KV Cache 连续分配** — `[n_layers, max_seq_len, n_kv_heads * head_dim]`
5. **逐 token 前向** — `forward(token_id, pos, kv_cache) → logits`，和 C++ API 相同
6. **GQA 通过 repeat_kv 展开** — 显式将 KV 头扩展到 Q 头数量

## 下一步：Phase 1

所有验证通过后，进入 `../llama-engine/` 开始 C++ 实现：

1. CMake + Eigen 项目搭建
2. 逐个算子翻译成 C++（每个算子完成后与 NumPy 输出对比）
3. GGUF 模型加载器
4. InferenceWorker 与 Notix 集成
