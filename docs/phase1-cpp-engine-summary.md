# Phase 1 总结：C++ LLM 推理引擎实现与验证

> 日期: 2026-06-20 | 状态: ✅ 完成

---

## 一、项目总览

在 Notix HTTP 框架内从零实现了一个 LLaMA 架构的 C++ 推理引擎，并与 HuggingFace 的输出逐 token 验证通过。

```
notix/
├── python_prototype/        # Phase 0: NumPy 参考实现 + 验证
│   ├── llama_forward.py      # 完整 LLaMA 前向传播
│   ├── verify_against_hf.py  # 与 HF 对比验证
│   ├── export_model.py       # 导出权重为 C++ 可读格式
│   └── README.md
├── llama-engine/             # Phase 1: C++ 引擎
│   ├── CMakeLists.txt
│   ├── third_party/eigen3/   # Eigen 3.4
│   ├── include/llama/
│   │   ├── config.h          # 模型 + 生成配置
│   │   ├── weights.h         # 权重结构体
│   │   ├── ops.h             # 算子声明
│   │   ├── kv_cache.h        # KV 缓存
│   │   ├── sampler.h         # 采样器
│   │   └── engine.h          # 主引擎接口
│   ├── src/
│   │   ├── ops.cpp           # rmsnorm, linear, rope, attention, swiglu
│   │   ├── kv_cache.cpp
│   │   ├── sampler.cpp       # temperature, top-k, top-p
│   │   └── engine.cpp        # 模型加载 + forward + generate
│   └── test/
│       ├── test_ops.cpp      # 算子单元测试 (8 项)
│       ├── test_engine.cpp   # 引擎集成测试 (4 项)
│       └── test_forward.cpp  # 与 HF 前向对比验证
└── docs/
    ├── llm-inference-engine-plan.md  # 原始技术方案
    └── phase1-cpp-engine-summary.md  # 本文档
```

---

## 二、Phase 0 → Phase 1 对照

每个 Python 模块都有 1:1 的 C++ 实现：

| Python (`llama_forward.py`) | C++ (`llama-engine/`) | 状态 |
|------------------------------|------------------------|------|
| `LlamaConfig` | `config.h :: ModelConfig` | ✅ |
| `LayerWeights / LlamaWeights` | `weights.h` | ✅ |
| `KVCache` | `kv_cache.h / kv_cache.cpp` | ✅ |
| `rmsnorm()` | `ops.cpp :: rmsnorm()` | ✅ |
| `apply_rotary_emb()` | `ops.cpp :: rope()` | ✅ |
| `attention_forward()` | `ops.cpp :: attention_forward()` | ✅ |
| `feed_forward()` (SwiGLU) | `ops.cpp :: swiglu_ffn()` | ✅ |
| `LlamaModel.forward()` | `engine.cpp :: forward()` | ✅ |
| `generate()` | `engine.cpp :: generate()` | ✅ |
| `Sampler` | `sampler.cpp` | ✅ |
| `load_weights_from_hf()` | `export_model.py` + `engine.cpp::load()` | ✅ |

---

## 三、验证结果

### 3.1 算子单元测试 (8/8 通过)

```
[test] rmsnorm_output_shape... PASSED ✓
[test] linear_output_size... PASSED ✓
[test] rope_preserves_norm... PASSED ✓
[test] rope_pos0_is_identity... PASSED ✓
[test] softmax_sums_to_one... PASSED ✓
[test] repeat_kv_correct_shape... PASSED ✓
[test] attention_output_shape... PASSED ✓
[test] swiglu_output_shape... PASSED ✓
```

### 3.2 引擎集成测试 (4/4 通过)

```
[test] config_validation... PASSED ✓
[test] kv_cache_init... PASSED ✓
[test] kv_cache_write_read... PASSED ✓
[test] rope_cache_precompute... PASSED ✓
```

### 3.3 与 HuggingFace 前向对比 (6/6 通过)

```
模型: TinyLlama-1.1B-Chat-v1.0 (dim=2048, 22 层, vocab=32000)
测试输入: "The capital of France is" (6 tokens)

pos=0  max_diff=4.77e-05  top5_match=✓
pos=1  max_diff=1.34e-05  top5_match=✓
pos=2  max_diff=1.34e-05  top5_match=✓
pos=3  max_diff=1.24e-05  top5_match=✓
pos=4  max_diff=1.24e-05  top5_match=✓
pos=5  max_diff=2.19e-05  top5_match=✓

结果: ALL PASSED ✓ — C++ matches HuggingFace!
```

---

## 四、调试中发现并修复的问题

### 4.1 Python 端 (Phase 0)

| # | 问题 | 根因 | 影响 |
|---|------|------|------|
| 1 | RoPE 维度配对错误 | 用 `(0↔1, 2↔3…)` 而非 HF 的 `(0↔32, 1↔33…)` | pos>0 即跑偏 |
| 2 | `rope_theta` 默认值错误 | 默认 500000 (LLaMA3) 但 TinyLlama 用 10000 | cos/sin 数值全错 |
| 3 | 局部变量 `dim` 未定义 | `feed_forward` 内缺少 `dim = config.dim` | 运行时报错 |

### 4.2 C++ 端 (Phase 1)

| # | 问题 | 根因 | 影响 |
|---|------|------|------|
| 4 | `Eigen::Map` 不能绑定到 `MatrixXf&` | Map 是临时对象，不能做非 const 引用 | 编译错误 |
| 5 | Head reshape 被转置 | Eigen 默认列优先，NumPy 行优先 | RoPE 旋转了错误的维度对 |
| 6 | Attention 输出 flatten 在列优先矩阵上 | 同上 | 线性层收到错误顺序的向量 |
| 7 | 权重文件行优先被当列优先读 | 导出脚本用 NumPy 行优先 `tobytes()`，Eigen 默认列优先 | **所有权重矩阵被转置**（核心 bug） |
| 8 | 测试数据索引列/行混淆 | 同上 | logits 对比全错 |
| 9 | Eigen Map 赋值问题 | Q/K heads 用 Map 直接传 rope() 后值未回写 | RoPE 结果丢失 |

### 4.3 关键技术决策

**NumPy ↔ Eigen 布局不匹配是最大陷阱。** 解决方案：

1. **权重加载**：文件保持 NumPy 行优先格式，C++ 读取时逐元素拷贝做行列转换
2. **Head reshape**：Q/K/V 头矩阵统一使用 `Eigen::RowMajor`，匹配 NumPy 的 `[h * head_dim + d]` 索引
3. **Attention 中间张量**：`k_all`、`v_all`、`k_exp`、`v_exp`、`attn_heads` 全部用 `RowMajor`

```cpp
// 关键 typedef
using RowMat = Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor>;
```

---

## 五、构建与运行

### 构建

```bash
cd notix/llama-engine
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 运行测试

```bash
# 算子单元测试
./test_ops

# 引擎集成测试
./test_engine

# 与 HF 对比验证（需要模型文件）
python ../python_prototype/export_model.py \
    --model ../model/TinyLlama-1.1B-Chat-v1.0 \
    --output ../test/tinyllama.bin
./test_forward ../test/tinyllama.bin
```

### 依赖

| 库 | 用途 | 获取方式 |
|----|------|---------|
| Eigen 3.4 | 线性代数 | 已集成在 `third_party/eigen3/` |
| g++ ≥ 11 | 编译器 | apt install g++ |
| CMake ≥ 3.16 | 构建 | apt install cmake |

---

## 六、已知限制与后续计划

### 当前 MVP 不支持

| 功能 | 优先级 | 说明 |
|------|--------|------|
| GGUF 模型格式 | P1 | 当前使用自定义 binary 格式 |
| Tokenizer | P1 | 需集成 SentencePiece |
| 流式 SSE 输出 | P1 | 需集成到 Notix InferenceGateway |
| FP16 推理 | P2 | 减半内存 |
| CUDA 后端 | P2 | GPU 加速 |
| Continuous batching | P3 | 并发多请求 |
| 多模型架构支持 | P3 | Qwen2/3 等 |

### Phase 2 计划

1. 集成 SentencePiece tokenizer
2. 实现 GGUF 模型加载
3. InferenceWorker 线程池 + Notix SSE 端点
4. 前端 `app.html` 流式输出适配

---

## 七、文件大小

| 组件 | 文件数 | 代码行数 |
|------|--------|---------|
| C++ 头文件 | 6 | ~320 |
| C++ 实现 | 4 | ~440 |
| C++ 测试 | 3 | ~380 |
| Python 原型 | 2 | ~850 |
| **总计** | **15** | **~2000** |

核心推理引擎（不含测试/原型）不到 800 行 C++。
