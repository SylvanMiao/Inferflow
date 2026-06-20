"""
Export HuggingFace model weights to C++ engine binary format.

Binary format (matches engine.cpp::load()):
    [int32]  magic  = 0x4C4C414D ("LLAM")
    [int32]  version
    [13×int32 + 2×float]  ModelConfig struct
    [float*] weights in column-major order (Eigen default)

Usage:
    python export_model.py --model ../model/TinyLlama-1.1B-Chat-v1.0 --output ../llama-engine/test/tinyllama.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MAGIC = 0x4C4C414D   # "LLAM"
VERSION = 1


def write_int(f, val: int):
    f.write(struct.pack('<i', val))

def write_float(f, val: float):
    f.write(struct.pack('<f', val))

def write_vec(f, arr: np.ndarray):
    """Write 1D array (no layout conversion needed)."""
    f.write(arr.astype(np.float32).tobytes())

def write_mat(f, arr: np.ndarray):
    """Write 2D matrix in column-major order (Eigen default)."""
    # arr is [rows, cols] row-major -> convert to Fortran (column-major)
    f.write(np.asfortranarray(arr.astype(np.float32)).tobytes())


def export_model(model_path: str, output_path: str):
    print(f"Loading model from {model_path}...")

    # Load config
    hf_config = AutoConfig.from_pretrained(model_path)
    config_dict = hf_config.to_dict()

    # Extract config values
    dim          = config_dict.get("hidden_size", 4096)
    n_layers     = config_dict.get("num_hidden_layers", 32)
    n_heads      = config_dict.get("num_attention_heads", 32)
    n_kv_heads   = config_dict.get("num_key_value_heads", n_heads)
    head_dim     = config_dict.get("head_dim", dim // n_heads)
    hidden_dim   = config_dict.get("intermediate_size", 14336)
    vocab_size   = config_dict.get("vocab_size", 128256)
    max_seq_len  = config_dict.get("max_position_embeddings", 8192)
    kv_dim       = n_kv_heads * head_dim
    rope_theta   = config_dict.get("rope_theta", config_dict.get("rope_theta_base", 10000.0))
    norm_eps     = config_dict.get("rms_norm_eps", 1e-5)
    bos_token_id = config_dict.get("bos_token_id", 1)
    eos_token_id = config_dict.get("eos_token_id", 2)

    print(f"  dim={dim} layers={n_layers} vocab={vocab_size}")
    print(f"  n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim}")
    print(f"  rope_theta={rope_theta}")

    # Load model weights
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, low_cpu_mem_usage=True)
    state = hf_model.state_dict()

    def get_tensor(key: str) -> np.ndarray:
        t = state[key]
        return t.float().numpy()

    # Extract weights (same names as HF)
    token_embd  = get_tensor("model.embed_tokens.weight")       # [vocab_size, dim]
    output_norm = get_tensor("model.norm.weight")                # [dim]
    lm_head     = get_tensor("lm_head.weight")                   # [vocab_size, dim]

    layers = []
    for i in range(n_layers):
        p = f"model.layers.{i}"
        layers.append({
            "attn_norm":    get_tensor(f"{p}.input_layernorm.weight"),
            "attn_q":       get_tensor(f"{p}.self_attn.q_proj.weight"),
            "attn_k":       get_tensor(f"{p}.self_attn.k_proj.weight"),
            "attn_v":       get_tensor(f"{p}.self_attn.v_proj.weight"),
            "attn_output":  get_tensor(f"{p}.self_attn.o_proj.weight"),
            "ffn_norm":     get_tensor(f"{p}.post_attention_layernorm.weight"),
            "ffn_gate":     get_tensor(f"{p}.mlp.gate_proj.weight"),
            "ffn_up":       get_tensor(f"{p}.mlp.up_proj.weight"),
            "ffn_down":     get_tensor(f"{p}.mlp.down_proj.weight"),
        })

    # ---- Write binary ----
    with open(output_path, 'wb') as f:
        write_int(f, MAGIC)
        write_int(f, VERSION)

        # Config struct
        write_int(f, dim)
        write_int(f, n_layers)
        write_int(f, n_heads)
        write_int(f, n_kv_heads)
        write_int(f, head_dim)
        write_int(f, hidden_dim)
        write_int(f, vocab_size)
        write_int(f, max_seq_len)
        write_int(f, kv_dim)
        write_float(f, float(rope_theta))
        write_float(f, float(norm_eps))
        write_int(f, bos_token_id)
        write_int(f, eos_token_id)

        # Global weights (column-major for matrices)
        write_mat(f, token_embd)      # [vocab_size, dim]
        write_vec(f, output_norm)     # [dim]
        write_mat(f, lm_head)         # [vocab_size, dim]

        # Per-layer weights
        for lw in layers:
            write_vec(f, lw["attn_norm"])     # [dim]
            write_mat(f, lw["attn_q"])        # [dim, dim]
            write_mat(f, lw["attn_k"])        # [kv_dim, dim]
            write_mat(f, lw["attn_v"])        # [kv_dim, dim]
            write_mat(f, lw["attn_output"])   # [dim, dim]
            write_vec(f, lw["ffn_norm"])      # [dim]
            write_mat(f, lw["ffn_gate"])      # [hidden_dim, dim]
            write_mat(f, lw["ffn_up"])        # [hidden_dim, dim]
            write_mat(f, lw["ffn_down"])      # [dim, hidden_dim]

    file_size = Path(output_path).stat().st_size
    print(f"Exported to {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    # ---- Also export test data (prompt + expected logits) ----
    print("\nGenerating test vectors (HF forward pass for comparison)...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_prompt = "The capital of France is"
    token_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
    print(f"  Test prompt: '{test_prompt}' → {len(token_ids)} tokens: {token_ids}")

    # Run HF forward for each position
    expected_logits = []
    with torch.no_grad():
        hf_model.eval()
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        outputs = hf_model(input_ids=input_tensor, use_cache=False)
        hf_logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Write test data as raw binary (C++ readable, row-major for simple indexing)
    test_bin_path = output_path.replace('.bin', '_test.bin')
    with open(test_bin_path, 'wb') as f:
        write_int(f, len(token_ids))
        f.write(np.array(token_ids, dtype=np.int32).tobytes())
        # expected logits: [num_tokens, vocab_size] row-major → pos*vocab + token
        f.write(hf_logits.detach().numpy().astype(np.float32).tobytes())
    print(f"  Test data: {test_bin_path} ({Path(test_bin_path).stat().st_size} bytes)")

    return token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HF model to C++ binary format")
    parser.add_argument("--model", required=True, help="HF model path")
    parser.add_argument("--output", required=True, help="Output .bin path")
    args = parser.parse_args()

    export_model(args.model, args.output)
    print("\nDone. Transfer the .bin and _test.npz files to the Linux machine.")
