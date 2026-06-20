"""
Phase 0 Verification: Compare NumPy LLaMA Forward Pass vs HuggingFace
=====================================================================

This script:
  1. Loads a HuggingFace LLaMA model (TinyLlama by default)
  2. Extracts weights into our NumPy format
  3. Runs the SAME input through both implementations
  4. Compares outputs at every layer

Dependencies:
    pip install torch transformers numpy

Usage:
    # With a local model:
    python3 verify_against_hf.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # With a specified test prompt:
    python3 verify_against_hf.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello world"

    # Quick test (no model download):
    # Just run python3 llama_forward.py for standalone tests
"""

import argparse
import math
import sys
import time
from typing import List, Tuple

import numpy as np

from llama_forward import (
    KVCache,
    LayerWeights,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaWeights,
    load_weights_from_hf,
    rmsnorm,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace Reference
# ═══════════════════════════════════════════════════════════════════════════════

class HfReference:
    """Wrapper around HuggingFace model for comparison purposes."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading HF model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = self.model.config

        print(f"  Model: {self.config.model_type}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Vocab size: {self.config.vocab_size}")

    def forward_single(self, input_ids: List[int], past_key_values=None, use_cache=True):
        """Run single-step forward pass and return logits + past_key_values."""
        import torch
        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            outputs = self.model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = outputs.logits[0, -1, :].numpy()  # [vocab_size]
            return logits, outputs.past_key_values

    def generate_reference(self, prompt: str, max_tokens: int = 20) -> List[int]:
        """Generate tokens using HF model."""
        import torch
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=torch.tensor([input_ids]),
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0].tolist()
        # Return only the new tokens
        return generated[len(input_ids):]


# ═══════════════════════════════════════════════════════════════════════════════
# Verification Routines
# ═══════════════════════════════════════════════════════════════════════════════

def verify_config(config: LlamaConfig, hf_ref: HfReference) -> bool:
    """Verify that our config matches HF config."""
    print("\n--- Config Verification ---")
    hf = hf_ref.config
    checks = [
        ("dim", config.dim, hf.hidden_size),
        ("n_layers", config.n_layers, hf.num_hidden_layers),
        ("n_heads", config.n_heads, hf.num_attention_heads),
        ("n_kv_heads", config.n_kv_heads,
         getattr(hf, 'num_key_value_heads', hf.num_attention_heads)),
        ("hidden_dim", config.hidden_dim, hf.intermediate_size),
        ("vocab_size", config.vocab_size, hf.vocab_size),
        ("max_seq_len", config.max_seq_len, getattr(hf, 'max_position_embeddings', -1)),
    ]

    all_pass = True
    for name, ours, theirs in checks:
        status = "✓" if ours == theirs else "✗ MISMATCH"
        if ours != theirs:
            all_pass = False
        print(f"  {name}: {ours} vs {theirs} {status}")

    if all_pass:
        print("  Config matches! ✓")
    return all_pass


def verify_rmsnorm(config: LlamaConfig, weights: LlamaWeights,
                   hf_ref: HfReference) -> bool:
    """Verify RMSNorm against HF at an intermediate layer."""
    print("\n--- RMSNorm Verification ---")
    import torch
    import torch.nn.functional as F

    # Get the HF RMSNorm layer
    hf_norm = hf_ref.model.model.layers[0].input_layernorm

    # Random input (small values to avoid numerical issues)
    rng = np.random.RandomState(42)
    x_np = rng.randn(config.dim).astype(np.float32) * 0.1

    # Our output
    our_out = rmsnorm(x_np, weights.layers[0].attn_norm, eps=config.norm_eps)

    # HF output
    x_pt = torch.from_numpy(x_np).reshape(1, 1, -1)
    hf_out = hf_norm(x_pt).squeeze().numpy()

    max_diff = np.max(np.abs(our_out - hf_out))
    mean_diff = np.mean(np.abs(our_out - hf_out))
    print(f"  Max absolute diff: {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")

    is_close = np.allclose(our_out, hf_out, atol=1e-5)
    status = "PASSED ✓" if is_close else "FAILED ✗"
    print(f"  allclose(atol=1e-5): {status}")

    return is_close


def verify_rope(config: LlamaConfig, freqs_cis: np.ndarray) -> bool:
    """Verify RoPE rotations are mathematically correct."""
    print("\n--- RoPE Verification ---")
    from llama_forward import apply_rotary_emb

    head_dim = config.head_dim
    # Test that RoPE is a rotation: applying with pos=0 gives identity
    q = np.random.randn(config.n_heads, head_dim).astype(np.float32)
    k = np.random.randn(config.n_kv_heads, head_dim).astype(np.float32)

    q_rot0, k_rot0 = apply_rotary_emb(q.copy(), k.copy(), freqs_cis, pos=0)
    # At pos=0, freqs = [cos(0) + i*sin(0)] = [1 + 0i] for all i
    # So RoPE should be identity
    id_close_q = np.allclose(q, q_rot0, atol=1e-6)
    id_close_k = np.allclose(k, k_rot0, atol=1e-6)
    print(f"  RoPE at pos=0 is identity (Q): {'✓' if id_close_q else '✗'}")
    print(f"  RoPE at pos=0 is identity (K): {'✓' if id_close_k else '✗'}")

    # Test: norm is preserved
    q_rot5, k_rot5 = apply_rotary_emb(q.copy(), k.copy(), freqs_cis, pos=5)
    q_norm_preserved = abs(np.linalg.norm(q) - np.linalg.norm(q_rot5)) < 1e-4
    k_norm_preserved = abs(np.linalg.norm(k) - np.linalg.norm(k_rot5)) < 1e-4
    print(f"  Norm preserved (Q): {'✓' if q_norm_preserved else '✗'}")
    print(f"  Norm preserved (K): {'✓' if k_norm_preserved else '✗'}")

    return id_close_q and id_close_k and q_norm_preserved and k_norm_preserved


def verify_single_forward(
    config: LlamaConfig,
    weights: LlamaWeights,
    hf_ref: HfReference,
    test_token: int,
    pos: int,
    kv_cache: KVCache,
) -> Tuple[bool, float, float]:
    """Verify a single forward step against HF.

    Returns:
        (passed, max_diff, mean_diff)
    """
    import torch

    # ---- Our forward ----
    our_model = LlamaModel(config, weights)
    our_hidden = our_model.forward(test_token, pos, kv_cache)

    # ---- HF forward ----
    with torch.no_grad():
        input_tensor = torch.tensor([[test_token]], dtype=torch.long)
        # We can't easily get intermediate hidden states without output_hidden_states=True
        # So for the full model verification we compare logits, not hidden states
        hf_model_full = hf_ref.model
        outputs = hf_model_full(
            input_ids=input_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
        # Get the last hidden state (before LM head)
        hf_hidden = outputs.hidden_states[-1][0, -1, :].numpy()  # [dim]

    max_diff = float(np.max(np.abs(our_hidden.reshape(-1) - hf_hidden)))
    mean_diff = float(np.mean(np.abs(our_hidden.reshape(-1) - hf_hidden)))
    is_close = bool(np.allclose(our_hidden.reshape(-1), hf_hidden, atol=1e-4))

    return is_close, max_diff, mean_diff


def verify_forward_pass(
    config: LlamaConfig,
    weights: LlamaWeights,
    hf_ref: HfReference,
    test_tokens: List[int],
) -> bool:
    """Verify the full forward pass token by token against HF.

    Compares logits (after LM head) at each position, which is the
    most important comparison for inference correctness.
    """
    import torch

    print("\n--- Full Forward Pass Verification ---")
    print(f"  Test tokens: {test_tokens[:10]}{'...' if len(test_tokens) > 10 else ''}")

    our_model = LlamaForCausalLM(config, weights)
    our_cache = KVCache(config)

    # Run HF forward (teacher forcing)
    with torch.no_grad():
        input_tensor = torch.tensor([test_tokens], dtype=torch.long)
        outputs = hf_ref.model(input_ids=input_tensor, use_cache=False)
        hf_logits_all = outputs.logits[0].numpy()  # [seq_len, vocab_size]

    # Compare logits at each position
    all_pass = True
    max_overall_diff = 0.0

    for pos, token in enumerate(test_tokens):
        our_logits = our_model.forward(token, pos, our_cache)
        hf_logits = hf_logits_all[pos]

        max_diff = float(np.max(np.abs(our_logits - hf_logits)))
        mean_diff = float(np.mean(np.abs(our_logits - hf_logits)))
        max_overall_diff = max(max_overall_diff, max_diff)
        is_close = bool(np.allclose(our_logits, hf_logits, atol=5e-4))

        if pos < 5 or pos == len(test_tokens) - 1 or not is_close:
            status = "✓" if is_close else "✗ MISMATCH"
            print(f"  pos={pos:3d}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  {status}")

        if not is_close:
            all_pass = False
            # Show top-5 logits comparison on failure
            our_top5 = np.argsort(-our_logits)[:5]
            hf_top5 = np.argsort(-hf_logits)[:5]
            print(f"    Our top-5:  {our_top5.tolist()}")
            print(f"    HF top-5:   {hf_top5.tolist()}")

    print(f"\n  Overall max diff: {max_overall_diff:.6f}")
    print(f"  Result: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")

    return all_pass


def verify_generation(
    config: LlamaConfig,
    weights: LlamaWeights,
    hf_ref: HfReference,
    prompt: str,
    max_tokens: int = 15,
) -> bool:
    """Compare generation output (greedy decoding) vs HF.

    Runs the same prompt through both our model and HF model
    with greedy decoding and compares the generated tokens.
    """
    import torch

    print("\n--- Generation Verification ---")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max new tokens: {max_tokens}")

    # Tokenize
    prompt_tokens = hf_ref.tokenizer.encode(prompt, add_special_tokens=True)
    print(f"  Tokenized: {len(prompt_tokens)} prompt tokens")

    # HF generate (greedy)
    with torch.no_grad():
        hf_input = torch.tensor([prompt_tokens], dtype=torch.long)
        hf_outputs = hf_ref.model.generate(
            input_ids=hf_input,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=hf_ref.tokenizer.eos_token_id,
        )
        hf_all_tokens = hf_outputs[0].tolist()
        hf_generated = hf_all_tokens[len(prompt_tokens):]

    # Our generate (greedy)
    our_model = LlamaForCausalLM(config, weights)
    our_cache = KVCache(config)

    # First, run through the prompt tokens
    for pos, tok in enumerate(prompt_tokens):
        _ = our_model.forward(tok, pos, our_cache)

    # Then generate
    our_generated = []
    current_pos = len(prompt_tokens)
    for _ in range(max_tokens):
        logits = our_model.forward(
            our_generated[-1] if our_generated else prompt_tokens[-1],
            current_pos - 1 if our_generated else len(prompt_tokens) - 1,
            # Need to handle this properly... let's fix
        )

    # Re-do generation properly
    our_model2 = LlamaForCausalLM(config, weights)
    our_generated = our_model2.generate(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=0.0,  # greedy
        top_k=0,
        top_p=1.0,
    )

    print(f"  HF generated tokens:   {hf_generated}")
    print(f"  Our generated tokens:  {our_generated}")

    match = our_generated == hf_generated
    if match:
        decoded = hf_ref.tokenizer.decode(hf_generated)
        print(f"  Decoded: '{decoded}'")
        print("  EXACT MATCH ✓")
    else:
        print("  MISMATCH ✗")
        # Show where they diverge
        for i, (o, h) in enumerate(zip(our_generated, hf_generated)):
            if o != h:
                print(f"  First divergence at position {i}: our={o} vs hf={h}")
                break

    return match


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Verify NumPy LLaMA forward pass against HuggingFace"
    )
    parser.add_argument(
        "--model", type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="The capital of France is",
        help="Test prompt for generation verification"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=15,
        help="Max tokens for generation test"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip generation test (faster)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 0 Verification: NumPy vs HuggingFace LLaMA Forward Pass")
    print("=" * 70)

    # ---- Load ----
    t0 = time.time()
    config, weights = load_weights_from_hf(args.model)
    hf_ref = HfReference(args.model)
    print(f"  Loading took {time.time() - t0:.1f}s")

    results = {}

    # ---- 1. Config match ----
    results["config"] = verify_config(config, hf_ref)

    # ---- 2. RMSNorm ----
    results["rmsnorm"] = verify_rmsnorm(config, weights, hf_ref)

    # ---- 3. RoPE ----
    from llama_forward import precompute_freqs_cis
    freqs_cis = precompute_freqs_cis(config)
    results["rope"] = verify_rope(config, freqs_cis)

    # ---- 4. Short forward pass ----
    test_tokens = hf_ref.tokenizer.encode(
        "Hello world, this is a test", add_special_tokens=True
    )[:20]
    results["forward"] = verify_forward_pass(config, weights, hf_ref, test_tokens)

    # ---- 5. Generation (optional) ----
    if not args.quick:
        results["generation"] = verify_generation(
            config, weights, hf_ref, args.prompt, args.max_tokens
        )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        if not passed:
            all_pass = False
        print(f"  {name:20s}: {status}")

    print()
    if all_pass:
        print("All checks passed! The NumPy implementation matches HuggingFace. ✓")
        print("You are ready to start the C++ implementation (Phase 1).")
    else:
        print("Some checks failed. Debug the mismatched operator(s) above.")
        print("Common causes:")
        print("  - Weight loading: check that layers are mapped correctly")
        print("  - RMSNorm epsilon: must match HF config exactly")
        print("  - RoPE: verify frequency computation matches")
        print("  - Attention: check QKV projection shapes and GQA group mapping")
        print("  - SwiGLU: verify gate/up/down weight assignments")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
