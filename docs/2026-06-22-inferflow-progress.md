# InferFlow 今日进展

> 日期: 2026-06-22

## 完成

- `generate()` 从空返回改为真实 token 生成。
- 接入 C++ SentencePiece tokenizer。
- 增加 `generate_text()` 和 `generate_text_stream()`。
- 增加 CLI 对话 demo `inferflow_cli`。
- CLI 增加 chat prompt template 和多轮 history。
- 增加 stop string 识别，避免输出 `</s>`、`<|user|>` 这类模板标记。

## 验证

```bash
cd notix/llama-engine
cmake -S . -B build
cmake --build build -j2
cd build
./test_ops
./test_engine
./test_forward ../test/tinyllama.bin
./inferflow_cli ../test/tinyllama.bin ../../../models/tokenizer.model 8
```

结果:

- `test_ops`: 8/8
- `test_engine`: 8/8
- `test_forward`: TinyLlama 与 HuggingFace 前向对齐通过
- `inferflow_cli`: 可交互输入并返回回答

## 当前限制

- 仍是 CPU 推理，速度较慢。
- 还没接 GGUF / safetensors。
- tokenizer 目前只接 SentencePiece。
- HTTP 服务端仍受 Boost 依赖限制，未在本机完成编译验证。

## 下一步

1. 把 CLI 中的 prompt template 抽成独立 `ChatTemplate`。
2. 做会话级 KV Cache 复用。
3. 继续推进 HTTP `/chat/completions`。
4. 开始抽象 Tensor / allocator，为后续 GPU backend 铺路。
