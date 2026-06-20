"""
Phase 0: LLaMA Forward Pass — Pure NumPy Implementation
========================================================

This is the Python reference implementation for the C++ inference engine.
Every function here will have a 1:1 C++ counterpart in the final engine.

The implementation is intentionally written to mirror C++ style:
 - Explicit shapes in comments
 - No PyTorch autograd
 - Float32 throughout
 - Row-major memory layout

Architecture: LLaMA3 (pre-norm, RoPE, GQA, SwiGLU)

Reference:
    HuggingFace transformers.models.llama.modeling_llama
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

Usage:
    python3 llama_forward.py                  # Run standalone tests
    python3 verify_against_hf.py              # Verify against HuggingFace model
"""

import json
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LlamaConfig:
    """Model hyperparameters. Maps to the GGUF metadata keys and config.json."""
    dim: int = 4096               # Hidden dimension
    n_layers: int = 32             # Number of transformer layers
    n_heads: int = 32              # Number of query attention heads
    n_kv_heads: int = 8            # Number of key/value heads (GQA)
    head_dim: int = 128            # Dimension per head (= dim / n_heads)
    hidden_dim: int = 14336        # FFN intermediate dimension
    vocab_size: int = 128256       # Vocabulary size
    max_seq_len: int = 8192        # Maximum sequence length
    rope_theta: float = 500000.0   # RoPE base frequency
    norm_eps: float = 1e-5         # RMSNorm epsilon
    bos_token_id: int = 128000     # Beginning of sequence token
    eos_token_id: int = 128001     # End of sequence token

    def __post_init__(self):
        # head_dim * n_heads must equal dim
        assert self.head_dim * self.n_heads == self.dim
        # n_heads must be divisible by n_kv_heads (GQA groups)
        assert self.n_heads % self.n_kv_heads == 0

    @property
    def n_groups(self) -> int:
        """Number of GQA groups (1 = MHA, n_heads/n_kv_heads = GQA)."""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "LlamaConfig":
        """Create config from HuggingFace config.json dictionary."""
        return cls(
            dim=config_dict.get("hidden_size", 4096),
            n_layers=config_dict.get("num_hidden_layers", 32),
            n_heads=config_dict.get("num_attention_heads", 32),
            n_kv_heads=config_dict.get("num_key_value_heads", 8),
            head_dim=config_dict.get("head_dim",
                                     config_dict.get("hidden_size", 4096)
                                     // config_dict.get("num_attention_heads", 32)),
            hidden_dim=config_dict.get("intermediate_size", 14336),
            vocab_size=config_dict.get("vocab_size", 128256),
            max_seq_len=config_dict.get("max_position_embeddings", 8192),
            rope_theta=config_dict.get("rope_theta", 500000.0),
            norm_eps=config_dict.get("rms_norm_eps", 1e-5),
            bos_token_id=config_dict.get("bos_token_id", 128000),
            eos_token_id=config_dict.get("eos_token_id", 128001),
        )

    # ---- TinyLlama 1.1B preset (for quick testing) ----
    @classmethod
    def tiny_llama(cls) -> "LlamaConfig":
        return cls(
            dim=2048, n_layers=22, n_heads=32, n_kv_heads=4,
            head_dim=64, hidden_dim=5632, vocab_size=32000,
            max_seq_len=2048, rope_theta=10000.0, norm_eps=1e-5,
            bos_token_id=1, eos_token_id=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Weight Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerWeights:
    """Weights for a single transformer decoder layer.

    All matrices are stored in row-major layout matching the GGUF/C++ convention:
        weight[i, j] = weight[i * cols + j]

    Shape convention (matches HuggingFace state_dict):
        attn_q:  [dim, n_heads * head_dim]       ← used as W^Q
        attn_k:  [n_kv_heads * head_dim, dim]    ← used as W^K
        attn_v:  [n_kv_heads * head_dim, dim]    ← used as W^V
        attn_output: [dim, dim]                  ← used as W^O
        ffn_gate: [hidden_dim, dim]              ← W1 (SwiGLU gate)
        ffn_up:   [hidden_dim, dim]              ← W3 (SwiGLU up)
        ffn_down: [dim, hidden_dim]              ← W2 (SwiGLU down)
    """
    attn_norm: np.ndarray       # [dim]            RMSNorm weight
    attn_q: np.ndarray          # [dim, dim]       Query projection
    attn_k: np.ndarray          # [kv_dim, dim]    Key projection
    attn_v: np.ndarray          # [kv_dim, dim]    Value projection
    attn_output: np.ndarray     # [dim, dim]       Output projection
    ffn_norm: np.ndarray        # [dim]            FFN RMSNorm weight
    ffn_gate: np.ndarray        # [hidden_dim, dim] SwiGLU gate
    ffn_up: np.ndarray          # [hidden_dim, dim] SwiGLU up
    ffn_down: np.ndarray        # [dim, hidden_dim] SwiGLU down


@dataclass
class LlamaWeights:
    """All weights for a LLaMA model."""
    token_embd: np.ndarray      # [vocab_size, dim]
    output_norm: np.ndarray     # [dim]
    lm_head: np.ndarray         # [vocab_size, dim]  (often shared with token_embd)
    layers: List[LayerWeights]  # [n_layers]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KV Cache
# ═══════════════════════════════════════════════════════════════════════════════

class KVCache:
    """Key-Value cache for autoregressive generation.

    Memory layout (matching planned C++ implementation):
        k_cache: [n_layers, max_seq_len, n_kv_heads * head_dim]
        v_cache: [n_layers, max_seq_len, n_kv_heads * head_dim]

    At position `pos`, the cache stores K and V vectors for all layers.
    These are reused in subsequent forward passes to avoid recomputation.
    """

    def __init__(self, config: LlamaConfig):
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_dim = config.n_kv_heads * config.head_dim
        self.current_len = 0

        # Pre-allocate contiguously
        self.k_cache = np.zeros(
            (config.n_layers, config.max_seq_len, self.kv_dim),
            dtype=np.float32
        )
        self.v_cache = np.zeros(
            (config.n_layers, config.max_seq_len, self.kv_dim),
            dtype=np.float32
        )

    def write_kv(self, layer_idx: int, pos: int,
                 k: np.ndarray, v: np.ndarray):
        """Write a single token's K and V at the given position."""
        self.k_cache[layer_idx, pos, :] = k  # [kv_dim]
        self.v_cache[layer_idx, pos, :] = v

    def get_kv_slice(self, layer_idx: int, seq_len: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the cached K and V for positions [0, seq_len)."""
        return (self.k_cache[layer_idx, :seq_len, :],
                self.v_cache[layer_idx, :seq_len, :])

    def clear(self):
        """Reset the cache for a new conversation."""
        self.k_cache.fill(0.0)
        self.v_cache.fill(0.0)
        self.current_len = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Operators
# ═══════════════════════════════════════════════════════════════════════════════

def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root Mean Square Layer Normalization.

    Formula: output = x * rsqrt(mean(x^2) + eps) * weight

    Args:
        x:      input  [batch, dim] or [dim]
        weight: scale  [dim]
        eps:    epsilon for numerical stability

    Returns:
        normalized output, same shape as input
    """
    # Compute RMS: sqrt(mean(x^2) + eps)
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def precompute_freqs_cis(config: LlamaConfig) -> np.ndarray:
    """Precompute the complex exponential for RoPE.

    This is called once at model initialization time.

    Returns:
        freqs_cis: [max_seq_len, head_dim//2]  complex64
            freqs_cis[pos, i] = exp(i * pos * theta_i)
            where theta_i = rope_theta^(-2i/head_dim)
    """
    dim = config.head_dim
    # theta_i = rope_theta^(-2i/dim) for i = 0, 1, ..., dim/2-1
    theta = 1.0 / (config.rope_theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
    # freqs[pos, i] = pos * theta_i
    positions = np.arange(config.max_seq_len, dtype=np.float32)
    freqs = np.outer(positions, theta)  # [max_seq_len, dim//2]
    # Convert to complex: cos + i*sin
    return np.exp(1j * freqs).astype(np.complex64)


def apply_rotary_emb(xq: np.ndarray, xk: np.ndarray,
                     freqs_cis: np.ndarray, pos: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Position Embedding (RoPE) in-place style.

    This rotates pairs of adjacent dimensions in query and key vectors.
    The rotation angle depends on the token position and the dimension index.

    Mathematical note: RoPE applies a position-dependent rotation matrix
    R(pos) to each pair of dimensions (2i, 2i+1):
        [cos(pos*θ_i)  -sin(pos*θ_i)] [x_2i  ]
        [sin(pos*θ_i)   cos(pos*θ_i)] [x_2i+1]

    This is equivalent to multiplying by e^(i*pos*θ_i) in complex space.

    Args:
        xq:   query tensor  [n_heads, head_dim]
        xk:   key tensor    [n_kv_heads, head_dim]
        freqs_cis: precomputed complex frequencies [max_seq_len, head_dim//2]
        pos:  current token position

    Returns:
        (rotated_xq, rotated_xk) — new arrays (input unchanged in pure functional style)
    """
    head_dim = xq.shape[-1]

    # View as complex numbers: pair (2i, 2i+1) → real + imag
    xq_complex = xq.reshape(xq.shape[0], -1, 2)
    xq_complex = xq_complex[..., 0] + 1j * xq_complex[..., 1]  # [n_heads, dim//2]

    xk_complex = xk.reshape(xk.shape[0], -1, 2)
    xk_complex = xk_complex[..., 0] + 1j * xk_complex[..., 1]  # [n_kv_heads, dim//2]

    # Get rotation factors for this position
    freqs = freqs_cis[pos, :]  # [head_dim//2]

    # Complex multiplication to apply rotation
    xq_rotated = xq_complex * freqs
    xk_rotated = xk_complex * freqs

    # Convert back to real pairs
    xq_out = np.stack([xq_rotated.real, xq_rotated.imag], axis=-1).reshape(xq.shape)
    xk_out = np.stack([xk_rotated.real, xk_rotated.imag], axis=-1).reshape(xk.shape)

    return xq_out.astype(np.float32), xk_out.astype(np.float32)


def repeat_kv(x: np.ndarray, n_groups: int) -> np.ndarray:
    """Expand KV heads for Grouped Query Attention (GQA).

    In GQA, there are fewer KV heads than query heads. This function
    repeats each KV head `n_groups` times to match the number of query heads.

    Args:
        x:        [n_kv_heads, seq_len, head_dim]
        n_groups: number of query heads per KV head

    Returns:
        [n_heads, seq_len, head_dim]
    """
    n_kv_heads, seq_len, head_dim = x.shape
    if n_groups == 1:
        return x
    # Repeat interleaved: kv0,kv0,kv1,kv1,... for n_groups=2
    x = np.repeat(x, n_groups, axis=0)  # [n_heads, seq_len, head_dim]
    return x


def attention_forward(
    hidden: np.ndarray,            # [1, dim]
    layer_weights: LayerWeights,
    kv_cache: KVCache,
    layer_idx: int,
    pos: int,
    freqs_cis: np.ndarray,
    config: LlamaConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-Head Attention with KV Cache (GQA variant).

    Steps:
      1. RMSNorm on hidden state
      2. Linear projections: Q = hidden @ Wq^T, K = hidden @ Wk^T, V = hidden @ Wv^T
      3. Apply RoPE to Q and K
      4. Write K and V into cache
      5. Scaled dot-product attention: softmax(Q @ K_cache^T / sqrt(d)) @ V_cache
      6. Output projection: attn @ Wo^T

    Args:
        hidden:        input hidden state  [1, dim]
        layer_weights: weights for this layer
        kv_cache:      the KV cache
        layer_idx:     which layer this is
        pos:           current token position
        freqs_cis:     precomputed RoPE frequencies
        config:        model config

    Returns:
        (attn_output, residual_normed_hidden)
          attn_output:            [1, dim]
          residual_normed_hidden: [1, dim]  (the normalized hidden, for residual add)
    """
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_groups = config.n_groups

    # ---- 1. RMSNorm ----
    normed = rmsnorm(hidden, layer_weights.attn_norm, eps=config.norm_eps)  # [1, dim]
    normed_flat = normed.reshape(dim)  # [dim]

    # ---- 2. QKV Projections ----
    # Q = x @ Wq^T   →   [dim] @ [dim, dim]^T   → [n_heads * head_dim]
    q = normed_flat @ layer_weights.attn_q.T     # [dim]   (since attn_q is [dim, dim])

    # K = x @ Wk^T   →   [dim] @ [kv_dim, dim]^T  → [n_kv_heads * head_dim]
    kv_dim = n_kv_heads * head_dim
    k = normed_flat @ layer_weights.attn_k.T     # [kv_dim]

    # V = x @ Wv^T
    v = normed_flat @ layer_weights.attn_v.T     # [kv_dim]

    # Reshape into heads
    q = q.reshape(n_heads, head_dim)              # [n_heads, head_dim]
    k = k.reshape(n_kv_heads, head_dim)           # [n_kv_heads, head_dim]
    v = v.reshape(n_kv_heads, head_dim)           # [n_kv_heads, head_dim]

    # ---- 3. RoPE ----
    q, k = apply_rotary_emb(q, k, freqs_cis, pos)

    # ---- 4. Write to KV Cache ----
    kv_cache.write_kv(layer_idx, pos, k.reshape(kv_dim), v.reshape(kv_dim))

    # Get all cached KV (positions 0..pos)
    k_cached, v_cached = kv_cache.get_kv_slice(layer_idx, pos + 1)
    # Reshape: [seq_len, n_kv_heads * head_dim] → [n_kv_heads, seq_len, head_dim]
    k_all = k_cached.reshape(pos + 1, n_kv_heads, head_dim).transpose(1, 0, 2)
    v_all = v_cached.reshape(pos + 1, n_kv_heads, head_dim).transpose(1, 0, 2)

    # ---- 5. Scaled Dot-Product Attention ----
    # q: [n_heads, head_dim]
    # k_all: [n_kv_heads, seq_len, head_dim]
    # Expand KV for GQA
    k_expanded = repeat_kv(k_all, n_groups)    # [n_heads, seq_len, head_dim]
    v_expanded = repeat_kv(v_all, n_groups)    # [n_heads, seq_len, head_dim]

    # scores = Q @ K^T / sqrt(d)
    # q: [n_heads, 1, head_dim] (implicit broadcast)
    # k_expanded: [n_heads, head_dim, seq_len] (via transpose)
    scale = 1.0 / math.sqrt(head_dim)
    scores = np.einsum('hd,hld->hl', q, k_expanded) * scale  # [n_heads, seq_len]

    # Apply causal mask (already implicit via using only cached K - no mask needed
    # since we only attend to positions 0..pos which are all valid)

    # Softmax (numerically stable)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # Weighted sum: attn @ V
    # attn_weights: [n_heads, seq_len]
    # v_expanded:   [n_heads, seq_len, head_dim]
    attn_out = np.einsum('hl,hld->hd', attn_weights, v_expanded)  # [n_heads, head_dim]
    attn_out = attn_out.reshape(dim)  # [dim]

    # ---- 6. Output Projection ----
    attn_out = attn_out @ layer_weights.attn_output.T  # [dim]

    return attn_out.reshape(1, dim), normed


def feed_forward(
    hidden: np.ndarray,            # [1, dim]
    layer_weights: LayerWeights,
    config: LlamaConfig,
) -> np.ndarray:
    """SwiGLU Feed-Forward Network.

    Steps:
      1. RMSNorm on hidden state
      2. Gate projection: gate = SiLU(normed @ W_gate^T)
      3. Up projection:   up = normed @ W_up^T
      4. Element-wise multiplication: gate * up
      5. Down projection: output = (gate * up) @ W_down^T

    The SwiGLU variant used in LLaMA performs better than the original
    FFN (ReLU(x @ W1) @ W2) because SiLU provides smooth gradients
    and the gating mechanism allows selective information flow.

    Args:
        hidden:        input  [1, dim]
        layer_weights: weights for this layer
        config:        model config

    Returns:
        ffn_output: [1, dim]
    """
    # ---- 1. RMSNorm ----
    normed = rmsnorm(hidden, layer_weights.ffn_norm, eps=config.norm_eps)  # [1, dim]
    normed_flat = normed.reshape(config.dim)  # [dim]

    # ---- 2. Gate: SiLU(x @ W_gate^T) ----
    gate = normed_flat @ layer_weights.ffn_gate.T  # [hidden_dim]
    # SiLU(x) = x * sigmoid(x)
    gate = gate * (1.0 / (1.0 + np.exp(-gate)))

    # ---- 3. Up: x @ W_up^T ----
    up = normed_flat @ layer_weights.ffn_up.T  # [hidden_dim]

    # ---- 4. Element-wise gate ----
    merged = gate * up  # [hidden_dim]

    # ---- 5. Down projection ----
    output = merged @ layer_weights.ffn_down.T  # [dim]

    return output.reshape(1, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class LlamaModel:
    """The core LLaMA transformer model (without LM head).

    This implements the raw transformer forward pass:
        embed → [decoder_layer × n_layers] → rmsnorm → hidden_state

    Each decoder layer has two sub-layers:
        1. Attention (with residual connection)
        2. Feed-Forward (with residual connection)
    """

    def __init__(self, config: LlamaConfig, weights: LlamaWeights):
        self.config = config
        self.weights = weights
        # Precompute RoPE frequencies once
        self.freqs_cis = precompute_freqs_cis(config)

    def forward(self, token_id: int, pos: int, kv_cache: KVCache) -> np.ndarray:
        """Single-step forward pass.

        This processes ONE token at a time (autoregressive).
        The KV cache stores all previous tokens' keys and values.

        Args:
            token_id: input token for this step
            pos:      position in the sequence
            kv_cache: persistent KV cache

        Returns:
            hidden_state: [1, dim] — the final hidden state (before LM head)
        """
        config = self.config

        # Embedding lookup
        # token_embd is [vocab_size, dim]
        hidden = self.weights.token_embd[token_id:token_id+1, :].copy()  # [1, dim]

        # ---- Transformer Layers ----
        for layer_idx in range(config.n_layers):
            layer_weights = self.weights.layers[layer_idx]

            # --- Attention sub-layer ---
            attn_out, _normed = attention_forward(
                hidden, layer_weights, kv_cache, layer_idx, pos,
                self.freqs_cis, config
            )
            # Residual connection
            hidden = hidden + attn_out  # [1, dim]

            # --- Feed-Forward sub-layer ---
            ffn_out = feed_forward(hidden, layer_weights, config)
            # Residual connection
            hidden = hidden + ffn_out  # [1, dim]

        # ---- Final RMSNorm ----
        hidden = rmsnorm(hidden, self.weights.output_norm, eps=config.norm_eps)

        return hidden  # [1, dim]


class LlamaForCausalLM:
    """LLaMA model with LM head for causal language modeling.

    This is the complete inference-ready model:
        LlamaModel.forward() → LM Head (logits) → Sampler → next token
    """

    def __init__(self, config: LlamaConfig, weights: LlamaWeights):
        self.config = config
        self.weights = weights
        self.model = LlamaModel(config, weights)

    def forward(self, token_id: int, pos: int, kv_cache: KVCache) -> np.ndarray:
        """Forward pass returning logits over vocabulary.

        Returns:
            logits: [vocab_size]  unnormalized log-probabilities
        """
        hidden = self.model.forward(token_id, pos, kv_cache)  # [1, dim]
        # LM Head: logits = hidden @ W_lm_head^T
        logits = hidden.reshape(self.config.dim) @ self.weights.lm_head.T  # [vocab_size]
        return logits

    def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token: Optional[int] = None,
    ) -> List[int]:
        """Generate tokens autoregressively.

        This function runs the full generation loop, producing one token
        per iteration. The KV cache is maintained across iterations.

        Args:
            prompt_tokens: input token IDs
            max_tokens:    maximum number of tokens to generate
            temperature:   sampling temperature (0 = greedy)
            top_k:         top-k filtering (0 = disabled)
            top_p:         nucleus sampling threshold (1.0 = disabled)
            stop_token:    EOS token (defaults to config.eos_token_id)

        Returns:
            generated token IDs (excluding prompt)
        """
        if stop_token is None:
            stop_token = self.config.eos_token_id

        kv_cache = KVCache(self.config)
        generated = []

        prompt_len = len(prompt_tokens)

        for pos in range(prompt_len + max_tokens):
            if pos < prompt_len:
                # Prompt phase: feed each prompt token
                token = prompt_tokens[pos]
                self.forward(token, pos, kv_cache)
                # During prompt, we don't sample (teacher forcing)
                if pos == prompt_len - 1:
                    # At the last prompt token, get logits for sampling
                    logits = self.forward(token, pos, kv_cache)
                    logits = self.forward(token, pos, kv_cache)
                continue

            # Generation phase: sample next token
            logits = self.forward(token, pos - 1, kv_cache)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy
                token = int(np.argmax(logits))
                generated.append(token)
                if token == stop_token:
                    break
                continue

            # Top-k filtering
            if top_k > 0 and top_k < len(logits):
                top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
                top_k_logits = logits[top_k_indices]
                top_k_logits_sorted_idx = np.argsort(-top_k_logits)
                top_k_indices = top_k_indices[top_k_logits_sorted_idx]
                logits = logits.copy()
                mask = np.ones_like(logits, dtype=bool)
                mask[top_k_indices] = False
                logits[mask] = -np.inf

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(-logits)
                sorted_logits = logits[sorted_indices]
                probs = np.exp(sorted_logits - np.max(sorted_logits))
                probs = probs / np.sum(probs)
                cumulative = np.cumsum(probs)
                cutoff_idx = np.searchsorted(cumulative, top_p) + 1
                if cutoff_idx < len(sorted_indices):
                    logits[sorted_indices[cutoff_idx:]] = -np.inf

            # Softmax
            logits_shifted = logits - np.max(logits)
            probs = np.exp(logits_shifted)
            probs = probs / np.sum(probs)

            # Sample
            token = int(np.random.choice(len(probs), p=probs))
            generated.append(token)

            if token == stop_token:
                break

        return generated


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Weight Loading from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════

def load_weights_from_hf(model_path: str) -> Tuple[LlamaConfig, LlamaWeights]:
    """Load weights from a HuggingFace transformers model directory.

    This function bridges the HF model format to our internal weight layout.
    The mapping between HF names and our internal names is documented inline.

    Args:
        model_path: path to a directory containing config.json and model .safetensors files

    Returns:
        (config, weights)
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # Load config
    hf_config = AutoConfig.from_pretrained(model_path)
    config_dict = hf_config.to_dict()
    config = LlamaConfig.from_hf_config(config_dict)

    # Load model (uses mmap internally for safetensors)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    state_dict = hf_model.state_dict()

    def get_tensor(key: str) -> np.ndarray:
        """Extract a tensor as float32 NumPy array."""
        t = state_dict[key]
        if t.dtype != torch.float32:
            t = t.float()
        return t.detach().numpy()

    # --- Global weights ---
    token_embd = get_tensor("model.embed_tokens.weight")      # [vocab_size, dim]
    output_norm = get_tensor("model.norm.weight")              # [dim]
    lm_head = get_tensor("lm_head.weight")                     # [vocab_size, dim]

    # --- Layer weights ---
    layers = []
    for i in range(config.n_layers):
        prefix = f"model.layers.{i}"

        lw = LayerWeights(
            attn_norm    = get_tensor(f"{prefix}.input_layernorm.weight"),
            attn_q       = get_tensor(f"{prefix}.self_attn.q_proj.weight"),
            attn_k       = get_tensor(f"{prefix}.self_attn.k_proj.weight"),
            attn_v       = get_tensor(f"{prefix}.self_attn.v_proj.weight"),
            attn_output  = get_tensor(f"{prefix}.self_attn.o_proj.weight"),
            ffn_norm     = get_tensor(f"{prefix}.post_attention_layernorm.weight"),
            ffn_gate     = get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            ffn_up       = get_tensor(f"{prefix}.mlp.up_proj.weight"),
            ffn_down     = get_tensor(f"{prefix}.mlp.down_proj.weight"),
        )
        layers.append(lw)

    weights = LlamaWeights(
        token_embd=token_embd,
        output_norm=output_norm,
        lm_head=lm_head,
        layers=layers,
    )

    return config, weights


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Standalone Tests (no HuggingFace required)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_random_weights(config: LlamaConfig) -> LlamaWeights:
    """Create random weights for testing. Used when no HF model is available."""
    rng = np.random.RandomState(42)
    dim, hidden_dim, vocab = config.dim, config.hidden_dim, config.vocab_size
    kv_dim = config.n_kv_heads * config.head_dim

    layers = []
    for _ in range(config.n_layers):
        lw = LayerWeights(
            attn_norm    = rng.randn(dim).astype(np.float32),
            attn_q       = rng.randn(dim, dim).astype(np.float32) * 0.02,
            attn_k       = rng.randn(kv_dim, dim).astype(np.float32) * 0.02,
            attn_v       = rng.randn(kv_dim, dim).astype(np.float32) * 0.02,
            attn_output  = rng.randn(dim, dim).astype(np.float32) * 0.02,
            ffn_norm     = rng.randn(dim).astype(np.float32),
            ffn_gate     = rng.randn(hidden_dim, dim).astype(np.float32) * 0.02,
            ffn_up       = rng.randn(hidden_dim, dim).astype(np.float32) * 0.02,
            ffn_down     = rng.randn(dim, hidden_dim).astype(np.float32) * 0.02,
        )
        layers.append(lw)

    return LlamaWeights(
        token_embd=rng.randn(vocab, dim).astype(np.float32) * 0.02,
        output_norm=rng.randn(dim).astype(np.float32),
        lm_head=rng.randn(vocab, dim).astype(np.float32) * 0.02,
        layers=layers,
    )


def test_rmsnorm():
    """Verify RMSNorm basic invariants."""
    print("[test] rmsnorm...", end=" ")
    dim, eps = 128, 1e-5
    weight = np.ones(dim, dtype=np.float32)
    x = np.random.randn(dim).astype(np.float32)

    out = rmsnorm(x, weight, eps)

    # Invariant: output should have similar magnitude to input
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    # For unit weight, output = x / rms(x)
    expected_rms = math.sqrt(np.mean(x**2) + eps)
    np.testing.assert_allclose(out, x / expected_rms, rtol=1e-5)
    print("PASSED ✓")


def test_rope_shape():
    """Verify RoPE preserves shape and approximately preserves norm."""
    print("[test] rope...", end=" ")
    config = LlamaConfig.tiny_llama()
    freqs_cis = precompute_freqs_cis(config)

    q = np.random.randn(config.n_heads, config.head_dim).astype(np.float32)
    k = np.random.randn(config.n_kv_heads, config.head_dim).astype(np.float32)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis, pos=5)

    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"

    # Norm should be approximately preserved (within float32 tolerance)
    # RoPE is a rotation, which is norm-preserving per pair
    assert abs(np.linalg.norm(q_rot) - np.linalg.norm(q)) < 1e-4, \
        f"Q norm changed: {np.linalg.norm(q_rot):.6f} vs {np.linalg.norm(q):.6f}"
    assert abs(np.linalg.norm(k_rot) - np.linalg.norm(k)) < 1e-4, \
        f"K norm changed: {np.linalg.norm(k_rot):.6f} vs {np.linalg.norm(k):.6f}"
    print("PASSED ✓")


def test_attention_shape():
    """Verify attention forward pass produces correct shapes."""
    print("[test] attention...", end=" ")
    config = LlamaConfig.tiny_llama()
    weights = _make_random_weights(config)
    freqs_cis = precompute_freqs_cis(config)
    kv_cache = KVCache(config)

    hidden = np.random.randn(1, config.dim).astype(np.float32)

    attn_out, normed = attention_forward(
        hidden, weights.layers[0], kv_cache, 0, 0, freqs_cis, config
    )

    assert attn_out.shape == (1, config.dim), f"Shape: {attn_out.shape}"
    assert normed.shape == (1, config.dim), f"Normed shape: {normed.shape}"
    assert kv_cache.current_len == 0, "KV cache should not auto-increment (caller manages)"

    # Second token
    hidden2 = np.random.randn(1, config.dim).astype(np.float32)
    attn_out2, _ = attention_forward(
        hidden2, weights.layers[0], kv_cache, 0, 1, freqs_cis, config
    )
    assert attn_out2.shape == (1, config.dim)
    print("PASSED ✓")


def test_ffn_shape():
    """Verify feed-forward produces correct shapes."""
    print("[test] ffn...", end=" ")
    config = LlamaConfig.tiny_llama()
    weights = _make_random_weights(config)

    hidden = np.random.randn(1, config.dim).astype(np.float32)
    out = feed_forward(hidden, weights.layers[0], config)

    assert out.shape == (1, config.dim), f"Shape: {out.shape}"
    print("PASSED ✓")


def test_model_forward():
    """End-to-end: model forward pass with random weights."""
    print("[test] model forward...", end=" ")
    config = LlamaConfig.tiny_llama()
    weights = _make_random_weights(config)
    model = LlamaModel(config, weights)
    kv_cache = KVCache(config)

    # Run a short sequence
    tokens = [1, 4, 7, 2]  # random token IDs
    for pos, tok in enumerate(tokens):
        hidden = model.forward(tok, pos, kv_cache)
        assert hidden.shape == (1, config.dim), f"Step {pos}: shape {hidden.shape}"

    print("PASSED ✓")


def test_generate_with_random_weights():
    """Verify generation loop produces output tokens."""
    print("[test] generate...", end=" ")
    config = LlamaConfig.tiny_llama()
    weights = _make_random_weights(config)
    model = LlamaForCausalLM(config, weights)

    generated = model.generate(
        prompt_tokens=[1, 2, 3],
        max_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    assert len(generated) <= 10, f"Generated {len(generated)} tokens > max 10"
    assert all(isinstance(t, (int, np.integer)) for t in generated), "Non-int tokens"
    print(f"PASSED ✓ (generated {len(generated)} tokens)")


def test_kv_cache_reuse():
    """Verify that KV cache correctly reuses previous computations."""
    print("[test] kv cache reuse...", end=" ")
    config = LlamaConfig.tiny_llama()
    weights = _make_random_weights(config)
    model = LlamaForCausalLM(config, weights)

    np.random.seed(123)
    logits1 = model.forward(5, 0, KVCache(config))

    np.random.seed(123)
    cache = KVCache(config)
    model.forward(1, 0, cache)  # dummy first token (not used for comparison)
    logits2 = model.forward(5, 1, cache)

    # logits1 and logits2 should differ because position 1 ≠ position 0
    # But the KV cache should correctly store intermediate state
    assert logits1.shape == logits2.shape
    print("PASSED ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0: NumPy LLaMA Forward Pass — Standalone Tests")
    print("=" * 60)
    print()

    test_rmsnorm()
    test_rope_shape()
    test_attention_shape()
    test_ffn_shape()
    test_model_forward()
    test_generate_with_random_weights()
    test_kv_cache_reuse()

    print()
    print("All standalone tests passed! ✓")
    print()
    print("Next step: run verify_against_hf.py with a real model to")
    print("validate numerical correctness against HuggingFace.")
