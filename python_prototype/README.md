# Python Reference

这个目录放的是 InferFlow 的 Python/NumPy 参考实现，用来验证 LLaMA 类模型的前向计算、生成流程和 HuggingFace 输出是否一致。

## 文件说明

| 文件 | 作用 |
|------|------|
| `llama_forward.py` | NumPy 版 LLaMA 前向推理与自回归生成 |
| `verify_against_hf.py` | 与 HuggingFace 模型做 logits / token 对齐验证 |
| `export_model.py` | 将 HuggingFace 权重导出为 C++ 引擎使用的 binary 格式 |
| `requirements.txt` | Python 依赖 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行基础测试

不需要下载模型，只检查算子 shape、KV Cache、RoPE、生成循环等基础逻辑：

```bash
python llama_forward.py
```

## 与 HuggingFace 对齐

先下载 TinyLlama：

```bash
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ../model/TinyLlama-1.1B-Chat-v1.0
```

运行验证：

```bash
python verify_against_hf.py --model ../model/TinyLlama-1.1B-Chat-v1.0
```

快速验证可以跳过生成测试：

```bash
python verify_against_hf.py --model ../model/TinyLlama-1.1B-Chat-v1.0 --quick
```

## 导出模型

```bash
python export_model.py \
  --model ../model/TinyLlama-1.1B-Chat-v1.0 \
  --output ../llama-engine/test/tinyllama.bin
```

导出的 `tinyllama.bin` 可被 `../llama-engine` 中的 C++ 推理引擎加载。

## 当前用途

- 作为 C++ 算子实现的参考版本。
- 用于定位 RoPE、RMSNorm、Attention、SwiGLU、KV Cache 等数值问题。
- 用于导出小模型权重，方便 HTTP 服务和 CLI demo 做端到端验证。
