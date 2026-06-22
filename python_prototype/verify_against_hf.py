"""
Verification: Compare NumPy LLaMA Forward Pass vs HuggingFace
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
    apply_rotary_emb,
    load_weights_from_hf,
    precompute_freqs_cis,
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
            dtype=torch.float32,
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
            logits = outputs.logits[0, -1, :].detach().numpy()  # [vocab_size]
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
    hf_out = hf_norm(x_pt).squeeze().detach().numpy()

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
        hf_hidden = outputs.hidden_states[-1][0, -1, :].detach().numpy()  # [dim]

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

    # Run HF forward (teacher forcing) — also get hidden states for debugging
    with torch.no_grad():
        input_tensor = torch.tensor([test_tokens], dtype=torch.long)
        outputs = hf_ref.model(input_ids=input_tensor, use_cache=False,
                               output_hidden_states=True)
        hf_logits_all = outputs.logits[0].detach().numpy()
        hf_hidden_all = [h.detach().numpy() for h in outputs.hidden_states]
        # hf_hidden_all[layer_idx][batch, seq, dim]
        # layer 0 = embedding, layer 1 = after layer 0, ..., layer 22 = after final layer

    # Compare logits at each position
    all_pass = True
    max_overall_diff = 0.0

    for pos, token in enumerate(test_tokens):
        our_logits = our_model.forward(token, pos, our_cache)
        hf_logits = hf_logits_all[pos]

        max_diff = float(np.max(np.abs(our_logits - hf_logits)))
        mean_diff = float(np.mean(np.abs(our_logits - hf_logits)))
        max_overall_diff = max(max_overall_diff, max_diff)

        # Use top-k match as the real correctness criterion
        our_top5 = set(np.argsort(-our_logits)[:5].tolist())
        hf_top5 = set(np.argsort(-hf_logits)[:5].tolist())
        top5_match = our_top5 == hf_top5

        if pos < 5 or pos == len(test_tokens) - 1 or not top5_match:
            status = "✓" if top5_match else "✗ MISMATCH"
            print(f"  pos={pos:3d}  max_diff={max_diff:.6f}  top5_match={top5_match}  {status}")

        if not top5_match:
            all_pass = False
            print(f"    Our top-5:  {sorted(our_top5)}")
            print(f"    HF top-5:   {sorted(hf_top5)}")
            try:
                _debug_hidden_states(config, weights, hf_hidden_all,
                                     test_tokens[:pos+1], pos, hf_ref)
            except Exception as e:
                print(f"    (debug skipped: {e})")

    print(f"\n  Overall max diff: {max_overall_diff:.6f}")
    print(f"  Result: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")

    return all_pass


def _debug_hidden_states(
    config: LlamaConfig,
    weights: LlamaWeights,
    hf_hidden_all: List[np.ndarray],
    tokens: List[int],
    debug_pos: int,
    hf_ref,
):
    """Compare hidden states at each layer between our model and HF.

    This is called when a forward pass mismatch is detected, to pinpoint
    which layer/operation introduces the numerical divergence.
    """
    import torch

    print(f"\n  [DEBUG] pos={debug_pos}: operator breakdown (layer 0):")
    lw = weights.layers[0]
    layer0_hf = hf_ref.model.model.layers[0]

    # Use the HIDDEN STATE AT DEBUG_POS from HF's embedding output
    # hf_hidden_all[0] = embedding layer output: [batch=1, seq=8, dim=2048]
    hf_hidden_pt = torch.from_numpy(
        hf_hidden_all[0][0:1, debug_pos:debug_pos+1, :].copy()
    )  # [1, 1, dim]

    # 1. Compare RMSNorm output
    with torch.no_grad():
        hf_norm_pt = layer0_hf.input_layernorm(hf_hidden_pt)
        hf_norm_np = hf_norm_pt[0, 0, :].detach().numpy()

    our_norm = rmsnorm(
        hf_hidden_all[0][0, debug_pos, :].reshape(1, -1),
        lw.attn_norm, eps=config.norm_eps
    )
    norm_diff = np.max(np.abs(our_norm.reshape(-1) - hf_norm_np))
    print(f"    RMSNorm:     max_diff={norm_diff:.8f}  {'✓' if norm_diff < 1e-5 else '✗'}")

    # 2. Compare Q projection
    with torch.no_grad():
        hf_q = layer0_hf.self_attn.q_proj(hf_norm_pt)[0, 0, :].detach().numpy()
    our_q = our_norm.reshape(config.dim) @ lw.attn_q.T
    q_diff = np.max(np.abs(our_q - hf_q))
    print(f"    Q_proj:      max_diff={q_diff:.8f}  {'✓' if q_diff < 1e-5 else '✗'}")

    # 3. Compare K projection
    with torch.no_grad():
        hf_k = layer0_hf.self_attn.k_proj(hf_norm_pt)[0, 0, :].detach().numpy()
    our_k = our_norm.reshape(config.dim) @ lw.attn_k.T
    k_diff = np.max(np.abs(our_k - hf_k))
    print(f"    K_proj:      max_diff={k_diff:.8f}  {'✓' if k_diff < 1e-5 else '✗'}")

    # 4. Compare after applying RoPE (the key suspect)
    our_q_heads = our_q.reshape(config.n_heads, config.head_dim)
    our_k_heads = our_k.reshape(config.n_kv_heads, config.head_dim)
    our_q_rope, our_k_rope = apply_rotary_emb(
        our_q_heads.copy(), our_k_heads.copy(),
        precompute_freqs_cis(config), debug_pos
    )

    # HF RoPE: build Q/K tensors and apply rotary
    with torch.no_grad():
        q_pt = hf_norm_pt @ layer0_hf.self_attn.q_proj.weight.T
        k_pt = hf_norm_pt @ layer0_hf.self_attn.k_proj.weight.T
        q_pt_v = q_pt.view(1, 1, config.n_heads, config.head_dim)
        k_pt_v = k_pt.view(1, 1, config.n_kv_heads, config.head_dim)

        # HF stores rotary_emb at different locations depending on version
        cos = sin = None
        for attr_name in ['rotary_emb', 'rotary_emb_layer']:
            if hasattr(layer0_hf.self_attn, attr_name):
                cos, sin = getattr(layer0_hf.self_attn, attr_name)(
                    k_pt_v, position_ids=torch.tensor([[debug_pos]]))
                break
        if cos is None:
            cos, sin = hf_ref.model.model.rotary_emb(
                k_pt_v, position_ids=torch.tensor([[debug_pos]]))

        # HF applies rotary via its own apply_rotary_pos_emb
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_rope
        hf_q_rope, hf_k_rope = hf_rope(q_pt_v, k_pt_v, cos, sin)
        hf_q_rope_np = hf_q_rope[0, 0, :, :].detach().numpy()
        hf_k_rope_np = hf_k_rope[0, 0, :, :].detach().numpy()

    # Compare cos/sin values from both implementations
    our_freqs = precompute_freqs_cis(config)
    our_cos = np.concatenate([np.real(our_freqs[debug_pos, :]),
                               np.real(our_freqs[debug_pos, :])])
    our_sin = np.concatenate([np.imag(our_freqs[debug_pos, :]),
                               np.imag(our_freqs[debug_pos, :])])
    hf_cos_np = cos[0, 0, :].detach().numpy()
    hf_sin_np = sin[0, 0, :].detach().numpy()
    print(f"    cos diff:    max_diff={np.max(np.abs(our_cos - hf_cos_np)):.8f}")
    print(f"    sin diff:    max_diff={np.max(np.abs(our_sin - hf_sin_np)):.8f}")
    # Print first 5 values from both for direct comparison
    print(f"    our cos[:5]:  {our_cos[:5]}")
    print(f"    HF cos[:5]:   {hf_cos_np[:5]}")
    # Find which theta generates these freqs
    inv_freq_ours = 1.0 / (config.rope_theta ** (np.arange(0, config.head_dim, 2).astype(np.float32) / config.head_dim))
    print(f"    our theta[:5]:     {inv_freq_ours[:5]}")
    # Check HF's actual inv_freq stored in the rotary_emb buffer
    hf_inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
    print(f"    expected HF theta[:5]: {hf_inv_freq[:5].numpy()}")
    # But what does the ACTUAL module have?
    re_module = None
    for attr_name in ['rotary_emb', 'rotary_emb_layer']:
        if hasattr(layer0_hf.self_attn, attr_name):
            re_module = getattr(layer0_hf.self_attn, attr_name)
            break
    if re_module is None:
        re_module = hf_ref.model.model.rotary_emb
    if hasattr(re_module, 'inv_freq'):
        print(f"    actual HF inv_freq[:5]: {re_module.inv_freq[:5].detach().numpy()}")
    else:
        print(f"    NO inv_freq buffer found in {type(re_module).__name__}")
    # Also check the position_ids we're passing
    pid = torch.tensor([[debug_pos]])
    print(f"    position_ids shape: {pid.shape}, value: {pid}")

    q_rope_diff = np.max(np.abs(our_q_rope - hf_q_rope_np))
    k_rope_diff = np.max(np.abs(our_k_rope - hf_k_rope_np))
    print(f"    after RoPE Q: max_diff={q_rope_diff:.8f}  {'✓' if q_rope_diff < 1e-5 else '✗'}")
    print(f"    after RoPE K: max_diff={k_rope_diff:.8f}  {'✓' if k_rope_diff < 1e-5 else '✗'}")


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

    # Our generate (greedy — uses same generate() tested in standalone)
    our_model = LlamaForCausalLM(config, weights)
    our_generated = our_model.generate(
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
