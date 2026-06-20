# Phase 0: Python/NumPy LLaMA Forward Pass Prototype

## Overview

This directory contains the Python reference implementation for the LLaMA inference engine.
Every function here has a planned 1:1 C++ counterpart.

## Files

| File | Purpose |
|------|---------|
| `llama_forward.py` | Complete NumPy implementation of the LLaMA forward pass |
| `verify_against_hf.py` | Verification script — compares against HuggingFace Transformers |
| `requirements.txt` | Python dependencies |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run standalone tests (no model needed)

```bash
python3 llama_forward.py
```

This runs 7 tests using random weights:
- RMSNorm invariants
- RoPE shape preservation & norm preservation
- Attention output shapes
- FFN output shapes
- Full model forward (TinyLlama config)
- Generation loop
- KV cache reuse

### 3. Verify against HuggingFace

```bash
# With TinyLlama (recommended first test — ~4GB download)
python3 verify_against_hf.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Quick test (skip generation)
python3 verify_against_hf.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quick

# Custom prompt
python3 verify_against_hf.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "你好世界"
```

## Architecture (NumPy → C++ Mapping)

```
llama_forward.py                           llama-engine/include/llama/
─────────────────────────────────────      ────────────────────────────
LlamaConfig                                 config.h
LlamaWeights / LayerWeights                 weights.h
KVCache                                     kv_cache.h
rmsnorm()                                   ops.h :: rmsnorm()
apply_rotary_emb()                          ops.h :: rope()
attention_forward()    ← MHA               ops.h :: attention()
feed_forward()         ← SwiGLU            ops.h :: swiglu_ffn()
LlamaModel.forward()                        engine.h :: forward_step()
LlamaForCausalLM.generate()                 engine.h :: generate()
load_weights_from_hf()                      gguf_loader.h
```

## Design Decisions (carried to C++)

1. **Functional operator style**: immutable inputs → outputs (no in-place mutation of weights)
2. **Float32 throughout**: no mixed precision in MVP
3. **Row-major memory layout**: matches Eigen default
4. **KVCache contiguous allocation**: `[n_layers, max_seq_len, n_kv_heads * head_dim]`
5. **Single-token forward**: `forward(token_id, pos, kv_cache) → logits` — same API as C++
6. **GQA via repeat_kv**: explicit expansion of KV heads to match Q heads

## Next: Phase 1

Once all verification passes, move to `../llama-engine/` for C++ implementation:
1. CMake + Eigen setup
2. Port operators one by one (each verified against NumPy reference)
3. GGUF model loader
4. InferenceWorker integration with Notix
