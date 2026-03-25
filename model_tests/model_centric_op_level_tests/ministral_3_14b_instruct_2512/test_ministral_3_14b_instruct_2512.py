# test_ministral_3_14b_instruct_2512.py
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CPU vs Spyre comparison tests for:
  • scaled_dot_product_attention (SDPA) — Ministral-3-14B-Instruct-2512
  • torch.cat                           — Ministral-3-14B-Instruct-2512
  • torch.zeros                         — Ministral-3-14B-Instruct-2512

Model Architecture (Ministral-3-14B-Instruct-2512):
  hidden_size         : 4096
  num_attention_heads : 32   (query heads)
  num_key_value_heads : 8    (GQA - key/value heads)
  head_dim            : 128  (4096 // 32)
  num_hidden_layers   : 40
  attention           : Grouped Query Attention (GQA) + Sliding Window

Target tensor shape : [1, 32, 1, 128]
Default dtype       : torch.float16

All SDPA tests run in eager mode (no torch.compile).
torch.cat tests run in both eager and compiled mode.

Running tests
-------------
All tests:
    pytest test_ministral_3_14b_instruct_2512.py

--- By op (umbrella marks) ---
All SDPA tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa

All torch.cat tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_cat

All torch.zeros tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_zeros

All torch.add tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_add

All torch.reshape tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_reshape

All F.silu tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_nn_functional_silu

All _log_api_usage_once tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_log_api_usage_once
--- SDPA sub-groups (fine-grained marks) ---
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_decode
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_prefill_causal
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_sliding_window
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_causal_flag_vs_mask
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_weights_sum_to_one
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_gqa_shape
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_batch_consistency
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_gradient_flow
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_determinism
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_padding_mask

--- Combining marks with -k keyword filters ---
All eager cat tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_cat -k "eager"

All compiled cat tests:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_cat -k "compiled"

A specific cat pattern:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_cat -k "pattern_003"

SDPA decode tests for a specific batch/kv size:
    pytest test_ministral_3_14b_instruct_2512.py -m torch_sdpa_decode -k "bs1_kv512"

--- Boolean mark expressions ---
Both ops together (same as running all):
    pytest test_ministral_3_14b_instruct_2512.py -m "torch_sdpa or torch_cat or torch_zeros"

sys.path: pytest runs from the repo root so this file's directory is not
on sys.path.  One insertion (model dir) is all that is needed —
utils.py is fully self-contained.
"""

import math
import os
import sys
import unittest

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add this file's own directory so utils (same dir) is importable.
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from utils_ministral_3_14b_instruct_2512 import (
    # Core config
    ParameterizedTestMeta,
    DEVICE,
    TOLERANCES,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    GQA_GROUPS,
    SCALE,
    SLIDING_WINDOW,
    NUM_LAYERS,
    DEFAULT_DTYPE,
    VOCAB_SIZE,
    ROPE_THETA,
    HIDDEN_SIZE,       
    INTERMEDIATE_SIZE,
    NUM_ATTENTION_HEADS,
    F16, I64, EMBED_DIM, _W, S, _,

    # FFN architecture constants
    INTERMEDIATE_SIZE,
    HIDDEN_SIZE,

    # SDPA pre-built param dicts
    PREFILL_PARAMS,
    DECODE_PARAMS,
    DTYPE_PARAMS,
    NUMERIC_COVERAGE_PARAMS,
    GROWING_KV_PARAMS,
    SLIDING_WINDOW_PARAMS,

    # Log-API key registries
    MINISTRAL_MODEL_KEYS,
    HF_KEYS,
    TORCH_INTERNAL_KEYS,
    RUNTIME_KEYS,

    # Tensor factories
    make_qkv,
    make_tensor,
    _t,

    # SDPA helpers
    expand_kv,
    causal_mask,
    sliding_window_mask,
    sdpa_fn,
    make_strided_tensor,
    # Comparison helpers
    compare_sdpa,
    compare_with_cpu,
)


# ─────────────────────────────────────────────────────────────────────────────
# SDPA parameterized test inputs
# ─────────────────────────────────────────────────────────────────────────────

_PREFILL_PARAMS          = PREFILL_PARAMS
_DECODE_PARAMS           = DECODE_PARAMS
_DTYPE_PARAMS            = DTYPE_PARAMS
_NUMERIC_COVERAGE_PARAMS = NUMERIC_COVERAGE_PARAMS
_GROWING_KV_PARAMS       = GROWING_KV_PARAMS
_SLIDING_WINDOW_PARAMS   = SLIDING_WINDOW_PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# TestScaledDotProductAttention
# ─────────────────────────────────────────────────────────────────────────────

_KV_CACHE_LEN = 2048   # full KV-cache allocation
 
# Prefill Q stride  [1, 32, S, 128]
_Q_PREFILL_STRIDES  = (57344,  128, 4096,   1)
# Decode Q stride   [1, 32, 1, 128]
_Q_DECODE_STRIDES   = ( 4096,  128,  128,   1)
# KV stride         [1, 32, 2048, 128]
_KV_STRIDES         = (8388608, 262144, 128, 1)
 
 
def _make_kv(batch: int = 1, dtype: torch.dtype = DEFAULT_DTYPE):
    """Return (k, v) with full KV-cache layout [B, 32, 2048, 128]."""
    k = make_strided_tensor(
        (batch, NUM_Q_HEADS, _KV_CACHE_LEN, HEAD_DIM),
        _KV_STRIDES,
        dtype=dtype,
    )
    v = make_strided_tensor(
        (batch, NUM_Q_HEADS, _KV_CACHE_LEN, HEAD_DIM),
        _KV_STRIDES,
        dtype=dtype,
    )
    return k, v
 
 
def _make_q_prefill(batch: int = 1, seq: int = 14,
                    dtype: torch.dtype = DEFAULT_DTYPE):
    """Return Q with prefill layout [B, 32, S, 128], stride [57344,128,4096,1]."""
    # The stride formula for arbitrary S:  batch_stride = 32*S*128 is impractical
    # to cache; we always use seq=14 (from the trace).  For other seq lengths we
    # scale batch stride proportionally so the tensor is internally consistent.
    batch_stride = NUM_Q_HEADS * seq * HEAD_DIM   # = 32 * seq * 128
    strides = (batch_stride, HEAD_DIM, NUM_Q_HEADS * HEAD_DIM, 1)
    return make_strided_tensor(
        (batch, NUM_Q_HEADS, seq, HEAD_DIM),
        strides,
        dtype=dtype,
    )
 
 
def _make_q_decode(batch: int = 1, dtype: torch.dtype = DEFAULT_DTYPE):
    """Return Q with decode layout [B, 32, 1, 128], stride [4096,128,128,1]."""
    return make_strided_tensor(
        (batch, NUM_Q_HEADS, 1, HEAD_DIM),
        _Q_DECODE_STRIDES,
        dtype=dtype,
    )
 
 
def _sdpa_fn_no_expand(q, k, v, attn_mask=None, is_causal=False):
    """
    SDPA wrapper that does NOT do GQA expansion.
 
    Since the KV-cache tensors already have NUM_Q_HEADS heads (32), we pass
    q/k/v directly to F.scaled_dot_product_attention.  The k/v seq-dim is
    sliced to match q's seq-dim for prefill, and left as-is for decode
    (SDPA handles seq_q != seq_kv natively).
    """
    seq_q = q.shape[2]
    seq_kv = k.shape[2]
    # For SDPA compatibility we need k/v seq >= q seq.
    # In practice seq_kv == _KV_CACHE_LEN (2048) >= seq_q.
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=SCALE,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Prefill param dict  – Q [1,32,14,128] x KV [1,32,2048,128]
# ─────────────────────────────────────────────────────────────────────────────
 
_SEQ_Q = 14   # prefill sequence length from the trace
 
_STRIDED_PREFILL_PARAMS = {
    "bs1_seq14_fp16":  (_make_q_prefill(1, _SEQ_Q, F16), *_make_kv(1, F16)),
    "bs1_seq14_fp32":  (_make_q_prefill(1, _SEQ_Q, torch.float32),
                        *_make_kv(1, torch.float32)),
    "bs1_seq14_bf16":  (_make_q_prefill(1, _SEQ_Q, torch.bfloat16),
                        *_make_kv(1, torch.bfloat16)),
}
 
# ─────────────────────────────────────────────────────────────────────────────
# Decode param dict   – Q [1,32,1,128] x KV [1,32,2048,128]
# ─────────────────────────────────────────────────────────────────────────────
 
_STRIDED_DECODE_PARAMS = {
    "bs1_kv2048_fp16": (_make_q_decode(1, F16), *_make_kv(1, F16)),
}
 
# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window param dict
# ─────────────────────────────────────────────────────────────────────────────
 
_STRIDED_SLIDING_WINDOW_PARAMS = {
    # Use a shorter seq so the test runs in reasonable time on CI;
    # still exercises the sliding-window code path.
    "seq14_kv2048": (_make_q_prefill(1, _SEQ_Q, F16), *_make_kv(1, F16)),
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TestSDPA
# ─────────────────────────────────────────────────────────────────────────────
 
class TestSDPA(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Eager and compiled CPU vs Spyre SDPA comparison tests for
    Ministral-3-14B-Instruct-2512.
 
    Tensor layouts match the yaml trace exactly:
      Q  [1, 32,   14, 128]  stride [57344,   128, 4096,    1]  (prefill)
      Q  [1, 32,    1, 128]  stride [4096,    128,  128,    1]  (decode)
      K  [1, 32, 2048, 128]  stride [8388608, 262144, 128,  1]  (KV-cache)
      V  [1, 32, 2048, 128]  stride [8388608, 262144, 128,  1]  (KV-cache)
 
    All tests use compare_with_cpu (no compare_sdpa).
    """
 
    pytestmark = pytest.mark.torch_sdpa
 
    torch.manual_seed(0xAFFE)
 
    PARAMS = {
        # ── Target shape: exact [1,32,1,128] decode ───────────────────────
        ("test_target_shape_decode", "test_sdpa_decode"): {
            "param_sets": {
                "bs1_kv2048_fp16_eager":    (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], False),
                "bs1_kv2048_fp16_compiled": (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], True),
            },
        },
 
        # ── Prefill (causal, float16) ─────────────────────────────────────
        ("test_prefill_causal_fp16", "test_sdpa_prefill_causal"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], True),
            },
        },
 
        # ── Decode (no mask, float16) ─────────────────────────────────────
        ("test_decode_fp16", "test_sdpa_decode"): {
            "param_sets": {
                "bs1_kv2048_fp16_eager":    (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], False),
                "bs1_kv2048_fp16_compiled": (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], True),
            },
        },
 
        # ── Multi-dtype prefill (causal) ──────────────────────────────────
        ("test_prefill_causal_multidtype", "test_sdpa_prefill_causal"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"],  False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"],  True),
                "bs1_seq14_fp32_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp32"],  False),
                "bs1_seq14_fp32_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp32"],  True),
                "bs1_seq14_bf16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_bf16"],  False),
                "bs1_seq14_bf16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_bf16"],  True),
            },
        },
 
        # ── Sliding-window mask ───────────────────────────────────────────
        ("test_sliding_window", "test_sdpa_sliding_window"): {
            "param_sets": {
                "seq14_kv2048_eager":    (*_STRIDED_SLIDING_WINDOW_PARAMS["seq14_kv2048"], False),
                "seq14_kv2048_compiled": (*_STRIDED_SLIDING_WINDOW_PARAMS["seq14_kv2048"], True),
            },
        },
 
        # ── is_causal=True ≡ explicit causal mask ─────────────────────────
        # For causal_flag_vs_mask the seq lengths must match (prefill-only test).
        ("test_causal_flag_vs_mask", "test_sdpa_causal_flag_vs_mask"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], True),
            },
        },
 
        # ── Attention-weight rows sum to 1 ────────────────────────────────
        ("test_attn_weights_sum_to_one", "test_sdpa_weights_sum_to_one"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], True),
            },
        },
 
        # ── GQA shape check (K/V already expanded to 32 heads) ────────────
        ("test_gqa_expansion_shape", "test_sdpa_gqa_shape"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], True),
            },
        },
 
        # ── Batch consistency ─────────────────────────────────────────────
        # Batch > 1 requires different stride patterns – use a single-batch
        # tensor and expand it inside the test body.
        ("test_batch_consistency", "test_sdpa_batch_consistency"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], True),
            },
        },
 
        # ── Gradient flow (eager only) ────────────────────────────────────
        ("test_gradient_flow", "test_sdpa_gradient_flow"): {
            "param_sets": {
                "bs1_seq14_fp16_eager": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
            },
        },
 
        # ── Determinism ───────────────────────────────────────────────────
        ("test_determinism", "test_sdpa_determinism"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], True),
                "bs1_decode_fp16_eager":   (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], False),
                "bs1_decode_fp16_compiled":(*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], True),
            },
        },
 
        # ── Padding mask ──────────────────────────────────────────────────
        ("test_padding_mask", "test_sdpa_padding_mask"): {
            "param_sets": {
                "bs1_seq14_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], False),
                "bs1_seq14_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"], True),
            },
        },
    }
 
    # ── Base test methods ──────────────────────────────────────────────────
 
    def test_sdpa_prefill_causal(self, q, k, v, compiled):
        """Prefill: causal SDPA — CPU vs Spyre via compare_with_cpu."""
        fn = lambda q, k, v: _sdpa_fn_no_expand(q, k, v, is_causal=False)
        # For prefill with seq_q < seq_kv we use an explicit causal mask
        # restricted to the first seq_q rows so padding rows don't affect
        # the causal pattern.  is_causal=True requires seq_q == seq_kv in
        # SDPA when the kv-cache is longer, so we build an explicit mask.
        seq_q  = q.shape[2]
        seq_kv = k.shape[2]
 
        def fn(q, k, v):
            # Build a [1, 1, seq_q, seq_kv] causal additive mask.
            # Each query position i may attend to key positions 0..i
            # (within the cache prefix), and not to positions > i.
            mask = torch.full(
                (1, 1, seq_q, seq_kv),
                float("-inf"),
                dtype=q.dtype,
                device=q.device,
            )
            for i in range(seq_q):
                mask[:, :, i, : i + 1] = 0.0
            return _sdpa_fn_no_expand(q, k, v, attn_mask=mask, is_causal=False)
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_decode(self, q, k, v, compiled):
        """Decode (seq_q=1): no mask — CPU vs Spyre."""
        def fn(q, k, v):
            return _sdpa_fn_no_expand(q, k, v, is_causal=False)
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_sliding_window(self, q, k, v, compiled):
        """Sliding-window mask over the KV-cache — CPU vs Spyre."""
        seq_q  = q.shape[2]
        seq_kv = k.shape[2]
 
        def fn(q, k, v):
            # Build a [1, 1, seq_q, seq_kv] mask combining causal +
            # sliding-window constraints.
            mask = torch.zeros(1, 1, seq_q, seq_kv,
                               dtype=q.dtype, device=q.device)
            for i in range(seq_q):
                # Causal: future positions masked.
                if i + 1 < seq_kv:
                    mask[:, :, i, i + 1 :] = float("-inf")
                # Sliding-window: positions older than SLIDING_WINDOW.
                start = max(0, i - SLIDING_WINDOW + 1)
                if start > 0:
                    mask[:, :, i, :start] = float("-inf")
            return _sdpa_fn_no_expand(q, k, v, attn_mask=mask, is_causal=False)
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_causal_flag_vs_mask(self, q, k, v, compiled):
        """
        Explicit causal mask and is_causal=True must agree.
 
        Both q and k/v must have the same seq_len for is_causal=True to be
        valid.  We slice the kv cache to seq_q to satisfy this.
        """
        seq_q = q.shape[2]
 
        def fn(q, k, v):
            # Slice KV to match Q seq_len.
            k_s = k[:, :, :seq_q, :].contiguous()
            v_s = v[:, :, :seq_q, :].contiguous()
 
            out_flag = F.scaled_dot_product_attention(
                q, k_s, v_s, is_causal=True, scale=SCALE,
            )
            mask = causal_mask(seq_q, q.dtype, q.device)
            out_mask = F.scaled_dot_product_attention(
                q, k_s, v_s, attn_mask=mask, is_causal=False, scale=SCALE,
            )
            tol = TOLERANCES[q.dtype]
            torch.testing.assert_close(
                out_flag, out_mask,
                atol=tol["atol"], rtol=tol["rtol"],
                msg="is_causal flag vs explicit mask differ",
            )
            return out_flag
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_weights_sum_to_one(self, q, k, v, compiled):
        """Softmax rows must sum to 1 (fp32 accumulation)."""
        seq_q  = q.shape[2]
        seq_kv = k.shape[2]
 
        def fn(q, k, v):
            # Build the causal attention scores manually.
            scores = torch.matmul(q, k.transpose(-2, -1)) * SCALE
            # Causal mask: upper-right triangle in [seq_q x seq_kv].
            mask = torch.full(
                (1, 1, seq_q, seq_kv),
                float("-inf"),
                dtype=q.dtype,
                device=q.device,
            )
            for i in range(seq_q):
                mask[:, :, i, : i + 1] = 0.0
            scores = scores + mask
            weights = torch.softmax(scores.float(), dim=-1)
            row_sums = weights.sum(dim=-1)
            return row_sums
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_gqa_shape(self, q, k, v, compiled):
        """K/V already have 32 heads — shape must equal Q shape."""
        def fn(q, k, v):
            assert k.shape == q.shape, (
                f"K shape {tuple(k.shape)} != Q shape {tuple(q.shape)}"
            )
            assert v.shape == q.shape, (
                f"V shape {tuple(v.shape)} != Q shape {tuple(q.shape)}"
            )
            # Run the actual SDPA so the function has a tensor output.
            seq_q  = q.shape[2]
            seq_kv = k.shape[2]
            mask = torch.full(
                (1, 1, seq_q, seq_kv),
                float("-inf"),
                dtype=q.dtype,
                device=q.device,
            )
            for i in range(seq_q):
                mask[:, :, i, : i + 1] = 0.0
            return _sdpa_fn_no_expand(q, k, v, attn_mask=mask)
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_batch_consistency(self, q, k, v, compiled):
        """All batch items from the same tensor must produce identical outputs."""
        tol = TOLERANCES[q.dtype]
        B = 2   # expand to batch=2 using .expand for consistency check
 
        def fn(q, k, v):
            # Expand the single-batch tensors to B=2.
            q_b = q.expand(B, -1, -1, -1)
            k_b = k.expand(B, -1, -1, -1)
            v_b = v.expand(B, -1, -1, -1)
 
            seq_q  = q.shape[2]
            seq_kv = k.shape[2]
            mask = torch.full(
                (1, 1, seq_q, seq_kv),
                float("-inf"),
                dtype=q.dtype,
                device=q.device,
            )
            for i in range(seq_q):
                mask[:, :, i, : i + 1] = 0.0
 
            out = _sdpa_fn_no_expand(q_b, k_b, v_b, attn_mask=mask)
            torch.testing.assert_close(
                out[0], out[1],
                atol=tol["atol"], rtol=tol["rtol"],
                msg="Batch item 1 differs from item 0",
            )
            return out
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_gradient_flow(self, q, k, v, compiled):
        """Q gradient back-propagates through SDPA without NaNs (eager only)."""
        seq_q  = q.shape[2]
        seq_kv = k.shape[2]
 
        def fn(q, k, v):
            q_g = q.detach().requires_grad_(True)
            mask = torch.full(
                (1, 1, seq_q, seq_kv),
                float("-inf"),
                dtype=q_g.dtype,
                device=q_g.device,
            )
            for i in range(seq_q):
                mask[:, :, i, : i + 1] = 0.0
            out = _sdpa_fn_no_expand(q_g, k.detach(), v.detach(), attn_mask=mask)
            out.sum().backward()
            assert q_g.grad is not None, "Gradient for Q is None"
            assert not torch.isnan(q_g.grad).any(), (
                f"NaN in Q gradient: "
                f"{torch.isnan(q_g.grad).sum().item()} / {q_g.grad.numel()}"
            )
            return out
 
        compare_with_cpu(fn, q, k, v, compiled=False)   # gradient test: eager only
 
    def test_sdpa_determinism(self, q, k, v, compiled):
        """Two consecutive identical SDPA calls must return the same output."""
        seq_q  = q.shape[2]
        seq_kv = k.shape[2]
 
        def fn(q, k, v):
            mask = torch.full(
                (1, 1, seq_q, seq_kv),
                float("-inf"),
                dtype=q.dtype,
                device=q.device,
            )
            for i in range(seq_q):
                mask[:, :, i, : i + 1] = 0.0
            out1 = _sdpa_fn_no_expand(q, k, v, attn_mask=mask)
            out2 = _sdpa_fn_no_expand(q, k, v, attn_mask=mask)
            torch.testing.assert_close(
                out1, out2,
                msg=(
                    f"Two identical SDPA calls returned different results. "
                    f"Max |Δ|: {(out1 - out2).abs().max().item():.6f}"
                ),
            )
            return out1
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    def test_sdpa_padding_mask(self, q, k, v, compiled):
        """Positions masked with -inf must receive near-zero attention weight."""
        seq_q  = q.shape[2]
        seq_kv = k.shape[2]
        half   = seq_kv // 2
 
        def fn(q, k, v):
            pad_mask = torch.zeros(
                1, 1, seq_q, seq_kv, dtype=q.dtype, device=q.device
            )
            # Mask out the second half of the KV-cache positions.
            pad_mask[:, :, :, half:] = float("-inf")
            scores  = torch.matmul(q, k.transpose(-2, -1)) * SCALE
            scores  = scores + pad_mask
            weights = torch.softmax(scores.float(), dim=-1)
            max_masked = weights[:, :, :, half:].abs().max()
            assert max_masked < 1e-5, (
                f"Masked positions got non-zero attention: "
                f"{max_masked.item():.2e} (threshold 1e-5)"
            )
            return weights
 
        compare_with_cpu(fn, q, k, v, compiled=compiled)
 
    # ── Non-parameterized sanity checks ───────────────────────────────────
 
    def test_model_config_constants(self):
        """Sanity-check architecture constants from utils_ministral."""
        assert NUM_Q_HEADS    == 32
        assert NUM_KV_HEADS   == 8
        assert GQA_GROUPS     == 4
        assert HEAD_DIM       == 128,   f"HEAD_DIM should be 128, got {HEAD_DIM}"
        assert NUM_LAYERS     == 40
        assert SLIDING_WINDOW == 4096
        assert abs(SCALE - 1.0 / math.sqrt(128)) < 1e-9
        assert INTERMEDIATE_SIZE == 16384, (
            f"INTERMEDIATE_SIZE should be 16384, got {INTERMEDIATE_SIZE}"
        )
 
    def test_target_tensor_shape_and_dtype(self):
        """
        Strided tensors must have the exact yaml-trace shapes and strides.
 
          Q  [1, 32,   14, 128]  stride [57344,   128, 4096,    1]
          K  [1, 32, 2048, 128]  stride [8388608, 262144, 128,  1]
          V  [1, 32, 2048, 128]  stride [8388608, 262144, 128,  1]
        """
        q, k, v = _STRIDED_PREFILL_PARAMS["bs1_seq14_fp16"]
 
        assert q.shape  == (1, NUM_Q_HEADS, _SEQ_Q,          HEAD_DIM), \
            f"Q shape: {tuple(q.shape)}"
        assert k.shape  == (1, NUM_Q_HEADS, _KV_CACHE_LEN,   HEAD_DIM), \
            f"K shape: {tuple(k.shape)}"
        assert v.shape  == (1, NUM_Q_HEADS, _KV_CACHE_LEN,   HEAD_DIM), \
            f"V shape: {tuple(v.shape)}"
 
        assert q.stride() == (NUM_Q_HEADS * _SEQ_Q * HEAD_DIM,
                               HEAD_DIM,
                               NUM_Q_HEADS * HEAD_DIM,
                               1), \
            f"Q stride: {q.stride()}"
        assert k.stride() == _KV_STRIDES, f"K stride: {k.stride()}"
        assert v.stride() == _KV_STRIDES, f"V stride: {v.stride()}"
 
        assert q.dtype == torch.float16
        assert k.dtype == torch.float16
        assert v.dtype == torch.float16
 
    def test_decode_tensor_shape_and_stride(self):
        """
        Decode Q must have shape [1,32,1,128] and stride [4096,128,128,1].
        """
        q, k, v = _STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"]
 
        assert q.shape  == (1, NUM_Q_HEADS, 1,             HEAD_DIM), \
            f"Q shape: {tuple(q.shape)}"
        assert k.shape  == (1, NUM_Q_HEADS, _KV_CACHE_LEN, HEAD_DIM), \
            f"K shape: {tuple(k.shape)}"
        assert q.stride() == _Q_DECODE_STRIDES, f"Q stride: {q.stride()}"
        assert k.stride() == _KV_STRIDES,       f"K stride: {k.stride()}"
 
# ═════════════════════════════════════════════════════════════════════════════
#  TestTorchReshape
# ═════════════════════════════════════════════════════════════════════════════

class TestReshape(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.reshape — Ministral-3-14B-Instruct-2512.

    Source ops from model:
        - attn_output.reshape(*input_shape, -1).contiguous()  # Line 173
        - Grouped KV reshape: (1, 8, 4, seq_kv, 128) → (1, 32, seq_kv, 128)
        - FFN intermediate reshape: (1, 14, 16384) → (14, -1)
    """

    pytestmark = pytest.mark.torch_reshape
    torch.manual_seed(0)

    PARAMS = {
    
        # ── GROUP A: model output shapes [B,S,32,128] → [B,S,4096] ──────────
        ("test_torch_reshape_A000", "_run_reshape_test"): {
            "param_sets": {
                "decode_1x1x32x128_eager":    (_t((1,  1, 32, 128),torch.float16,(4096, 128, 128, 1)), (1,  1, -1), False),
                "decode_1x1x32x128_compiled": (_t((1,  1, 32, 128),torch.float16,(4096, 128, 128, 1)), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_A001", "_run_reshape_test"): {
            "param_sets": {
                "prefill_1x14x32x128_eager":    (_t((1, 8, 4, 2048, 128),torch.float16,(2097152, 262144, 0, 128, 1)), (1, 32, 2048, 128), False),
                "prefill_1x14x32x128_compiled": (_t((1, 8, 4, 2048, 128),torch.float16,(2097152, 262144, 0, 128, 1)), (1, 32, 2048, 128), True),
            }
        },
        ("test_torch_reshape_A002", "_run_reshape_test"): {
            "param_sets": {
                "prefill_1x14x32x128_eager":    (_t((1, 14, 32, 128),torch.float16,(57344, 4096, 128, 1)), (1, 14, -1), False),
                "prefill_1x14x32x128_compiled": (_t((1, 14, 32, 128),torch.float16,(57344, 4096, 128, 1)), (1, 14, -1), True),
            }
        },
        


        # ── GROUP L: non-contiguous (transpose → reshape, exact model path) ──
        ("test_torch_reshape_L000", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_q_decode_eager":    (_t((1, 32,  1, 128)), 1, 2, (1,  1, -1), False),
                "noncontig_q_decode_compiled": (_t((1, 32,  1, 128)), 1, 2, (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_L001", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_q_prefill14_eager":    (_t((1, 32, 14, 128)), 1, 2, (1, 14, -1), False),
                "noncontig_q_prefill14_compiled": (_t((1, 32, 14, 128)), 1, 2, (1, 14, -1), True),
            }
        },

        # ── GROUP M: .reshape().contiguous() — full model op chain ───────────
        ("test_torch_reshape_M000", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_decode_eager":    (_t((1,  1, 32, 128),torch.float16,(4096, 128, 128, 1)), (1,  1, -1), False),
                "chain_decode_compiled": (_t((1,  1, 32, 128),torch.float16,(4096, 128, 128, 1)), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_M001", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_prefill_eager":    (_t((1, 14, 32, 128),torch.float16,(57344, 4096, 128, 1)), (1, 14, -1), False),
                "chain_prefill_compiled": (_t((1, 14, 32, 128),torch.float16,(57344, 4096, 128, 1)), (1, 14, -1), True),
            }
        },
        ("test_torch_reshape_M002", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_prefill_eager":    (_t((1, 8, 4, 2048, 128),torch.float16,(2097152, 262144, 0, 128, 1)), (1, 32, 2048, 128), False),
                "chain_prefill_compiled": (_t((1, 8, 4, 2048, 128),torch.float16,(2097152, 262144, 0, 128, 1)), (1, 32, 2048, 128), True),
            }
        },
        # ── GROUP S: CPU-only contiguity structural assertion ─────────────────
        ("test_torch_reshape_S000", "_run_reshape_contiguity_test"): {
            "param_sets": {
                "contiguity_decode":  (_t((1,  1, 32, 128),torch.float16,(4096, 128, 128, 1)), (1,  1, -1)),
                "contiguity_prefill": (_t((1, 14, 32, 128),torch.float16,(57344, 4096, 128, 1)), (1, 14, -1)),
            }
        },
    }

    def _run_reshape_test(self, tensor, target_shape, compiled):
        """tensor.reshape(*target_shape) — eager or compiled."""
        compare_with_cpu(lambda t: t.reshape(*target_shape), tensor, compiled=compiled)

    def _run_reshape_contiguous_test(self, tensor, target_shape, compiled):
        """.reshape().contiguous() — full model op chain."""
        compare_with_cpu(
            lambda t: t.reshape(*target_shape).contiguous(),
            tensor, compiled=compiled,
        )

    def _run_reshape_after_transpose_test(self, tensor, d0, d1, target_shape, compiled):
        """tensor.transpose(d0,d1).reshape(*target_shape) — non-contiguous input."""
        compare_with_cpu(
            lambda t: t.transpose(d0, d1).reshape(*target_shape),
            tensor, compiled=compiled,
        )

    def _run_reshape_contiguity_test(self, tensor, target_shape):
        """CPU-only: contiguous input → contiguous output, same numel."""
        t = tensor.cpu()
        assert t.is_contiguous()
        result = t.reshape(*target_shape)
        assert result.is_contiguous(), (
            f"reshape {tuple(t.shape)} → {target_shape} is not contiguous"
        )
        assert result.numel() == t.numel()


# ═════════════════════════════════════════════════════════════════════════════
#  TestFunctionalSilu
# ═════════════════════════════════════════════════════════════════════════════

class TestFunctionalSilu(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.nn.functional.silu —
    Ministral-3-14B-Instruct-2512.
    SiLU is used in every FFN block as the SwiGLU gate:
        F.silu(gate_proj(x)) * up_proj(x)
    intermediate_size: 16384 (14B)
    """

    pytestmark = pytest.mark.torch_nn_functional_silu
    torch.manual_seed(0xCAFE)

    _INTERMEDIATE_14B = INTERMEDIATE_SIZE  # 16384
    _INTERMEDIATE_3B  = 8192  # kept for legacy
    _HIDDEN           = HIDDEN_SIZE

    PARAMS = {

        # ── GROUP A: FFN gate decode [B,1,intermediate] ───────────────────────
        ("test_silu_A000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "decode_1x1x16384_eager":    (_t((1, 1, 16384),torch.float16, (16384, 16384, 1)), False),
                "decode_1x1x16384_compiled": (_t((1, 1, 16384),torch.float16, (16384, 16384, 1)), True),
            }
        },


        # ── GROUP B: FFN gate prefill [B,S,intermediate] ─────────────────────
        ("test_silu_B000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "prefill_1x14x16384_eager":    (_t((1,  14, 16384),torch.float16, (229376, 16384, 1)), False),
                "prefill_1x14x16384_compiled": (_t((1,  14, 16384),torch.float16, (229376, 16384, 1)), True),
            }
        },


        # ── GROUP D: SwiGLU product F.silu(gate) * up ────────────────────────
        ("test_silu_D000", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_decode_1x1x16384_eager":    (_t((1, 1, 16384),torch.float16, (16384, 16384, 1)), _t((1,  1, 16384),torch.float16, (16384, 16384, 1)), False),
                "swiglu_decode_1x1x16384_compiled": (_t((1,  1, 16384),torch.float16, (16384, 16384, 1)), _t((1,  1, 16384),torch.float16, (16384, 16384, 1)), True),
            }
        },
        ("test_silu_D001", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_prefill_1x14x16384_eager":    (_t((1,  14, 16384),torch.float16, (229376, 16384, 1)),_t((1,  14, 16384),torch.float16, (229376, 16384, 1)), False),
                "swiglu_prefill_1x14x16384_compiled": (_t((1, 14, 16384),torch.float16, (229376, 16384, 1)), _t((1, 14, 16384),torch.float16, (229376, 16384, 1)), True),
            }
        },

        # ── GROUP F: non-contiguous input (transpose → silu) ─────────────────
        ("test_silu_F000", "_run_silu_noncontig_test"): {
            "param_sets": {
                "noncontig_gate_decode_eager":    (_t((1, 16384, 1)), 0, 1, False),
                "noncontig_gate_decode_compiled": (_t((1, 16384, 1)), 0, 1, True),
            }
        },


        # ── GROUP H: CPU-only numerical identity F.silu(x) == x*sigmoid(x) ───
        ("test_silu_H000", "_run_silu_identity_check_test"): {
            "param_sets": {
                "identity_gate_decode_fp32":  (_t((1, 1, 16384),torch.float16, (16384, 16384, 1)),),
                "identity_gate_prefill_fp32": (_t((1,  14, 16384),torch.float16, (229376, 16384, 1)),),
            }
        },
    }

    def _run_silu_ffn_gate_test(self, tensor, compiled):
        """F.silu(gate) — FFN gate activation, eager or compiled."""
        compare_with_cpu(
            lambda t: torch.nn.functional.silu(t),
            tensor, compiled=compiled,
        )

    def _run_silu_swiglu_test(self, gate, up, compiled):
        """F.silu(gate) * up — full SwiGLU product, eager or compiled."""
        compare_with_cpu(
            lambda g, u: torch.nn.functional.silu(g) * u,
            gate, up, compiled=compiled,
        )
    def _run_silu_noncontig_test(self, tensor, d0, d1, compiled):
        """F.silu on a non-contiguous view produced by transpose — made contiguous for Spyre."""
        compare_with_cpu(
            lambda t: torch.nn.functional.silu(t.transpose(d0, d1).contiguous()),
            tensor, compiled=compiled,
        )

    def _run_silu_special_values_test(self, tensor):
        """CPU-only: IEEE 754 special-value behaviour of F.silu."""
        t      = tensor.cpu().float()
        result = torch.nn.functional.silu(t).float()
        for idx in range(t.numel()):
            raw = t.view(-1)[idx].item()
            got = result.view(-1)[idx].item()
            if math.isnan(raw):
                assert math.isnan(got), f"silu(NaN) should be NaN, got {got}"
            elif raw == float("inf"):
                assert got == float("inf"), f"silu(+inf) should be +inf, got {got}"
            elif raw == float("-inf"):
                assert math.isnan(got), f"silu(-inf) should be NaN (IEEE 754), got {got}"
            elif raw == 0.0:
                assert got == 0.0, f"silu(0) should be 0.0, got {got}"
            else:
                # For inputs whose sigmoid underflows to ±0 (e.g. -65504), the
                # product saturates to -0.0.  Both ±0.0 are acceptable outputs
                # for inputs whose magnitude is large enough to saturate; skip
                # the sign check when the result is zero.
                if got == 0.0:
                    pass  # underflow to signed-zero is valid IEEE 754 behaviour
                else:
                    assert (raw >= 0) == (got >= 0), \
                        f"silu({raw}) sign wrong: got {got}"

    def _run_silu_identity_check_test(self, tensor):
        """CPU-only: F.silu(x) == x * sigmoid(x) element-wise (fp32)."""
        t = tensor.cpu().float()
        torch.testing.assert_close(
            torch.nn.functional.silu(t),
            t * torch.sigmoid(t),
            atol=1e-5, rtol=1e-5,
            msg=lambda msg: f"F.silu(x) != x*sigmoid(x) on {tuple(t.shape)}\n\n{msg}\n",
        )

    def test_silu_zero_fixed_point(self):
        """silu(0) == 0 for fp16, bf16, fp32."""
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            assert torch.nn.functional.silu(torch.zeros(1, dtype=dtype)).item() == 0.0

    def test_silu_shape_preserved_model_shapes(self):
        """F.silu must not alter shape for canonical model shapes."""
        for shape in [
            (1, 1, 16384), (1, 14, 16384), (1, 128, 16384),
            (1, 1, 8192),  (1, 1,  4096),  (1, 14,  4096),
        ]:
            t = _t(shape)
            assert torch.nn.functional.silu(t).shape == t.shape

    def test_silu_swiglu_matches_decomposition(self):
        """F.silu(gate)*up == (gate*sigmoid(gate))*up in fp32."""
        gate = _t((1, 14, 16384), torch.float32)
        up   = _t((1, 14, 16384), torch.float32)
        torch.testing.assert_close(
            torch.nn.functional.silu(gate) * up,
            (gate * torch.sigmoid(gate)) * up,
            atol=1e-5, rtol=1e-5,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TestLogApiUsageOnce
# ═════════════════════════════════════════════════════════════════════════════

class TestLogApiUsageOnce(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch._C._log_api_usage_once — Ministral-3-14B-Instruct-2512.

    Void C-extension telemetry hook: always returns None, idempotent,
    thread-safe, process-global.

    Now runs on both CPU and Spyre devices with tensor operations to verify
    the call works in compiled mode as well.
    """

    pytestmark = pytest.mark.torch_log_api_usage_once

    _MINISTRAL_MODEL_KEYS = MINISTRAL_MODEL_KEYS
    _HF_KEYS              = HF_KEYS
    _TORCH_INTERNAL_KEYS  = TORCH_INTERNAL_KEYS
    _RUNTIME_KEYS         = RUNTIME_KEYS

    PARAMS = {

        # ── GROUP A: Ministral model keys ────────────────────────────────────
        ("test_log_api_A000", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "model_forward_eager":    ("__A000__ministral.MistralModel.forward",          _t((1, 14, 4096)), False),
                "model_forward_compiled": ("__A000__ministral.MistralModel.forward",          _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_A001", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "attn_forward_eager":    ("__A001__ministral.MistralAttention.forward",       _t((1, 32,  1, 128)), False),
                "attn_forward_compiled": ("__A001__ministral.MistralAttention.forward",       _t((1, 32,  1, 128)), True),
            }
        },
        ("test_log_api_A002", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "mlp_forward_eager":    ("__A002__ministral.MistralMLP.forward",              _t((1,  1, 16384)), False),
                "mlp_forward_compiled": ("__A002__ministral.MistralMLP.forward",              _t((1,  1, 16384)), True),
            }
        },
        ("test_log_api_A003", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "rmsnorm_forward_eager":    ("__A003__ministral.MistralRMSNorm.forward",      _t((1, 14, 4096)), False),
                "rmsnorm_forward_compiled": ("__A003__ministral.MistralRMSNorm.forward",      _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_A004", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "rotary_forward_eager":    ("__A004__ministral.MistralRotaryEmbedding.forward", _t((1, 14, 64)), False),
                "rotary_forward_compiled": ("__A004__ministral.MistralRotaryEmbedding.forward", _t((1, 14, 64)), True),
            }
        },
        ("test_log_api_A005", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "decoder_layer_eager":    ("__A005__ministral.MistralDecoderLayer.forward",   _t((1, 14, 4096)), False),
                "decoder_layer_compiled": ("__A005__ministral.MistralDecoderLayer.forward",   _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_A006", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "causal_lm_eager":    ("__A006__ministral.MistralForCausalLM.forward",        _t((1, 14, 4096)), False),
                "causal_lm_compiled": ("__A006__ministral.MistralForCausalLM.forward",        _t((1, 14, 4096)), True),
            }
        },

        # ── GROUP B: HuggingFace + PyTorch internal keys ──────────────────────
        ("test_log_api_B000", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "hf_pretrained_eager":    ("__B000__transformers.PreTrainedModel.forward",    _t((1, 14, 4096)), False),
                "hf_pretrained_compiled": ("__B000__transformers.PreTrainedModel.forward",    _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_B001", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "hf_generate_eager":    ("__B001__transformers.generation.GenerationMixin.generate", _t((1, 1, 4096)), False),
                "hf_generate_compiled": ("__B001__transformers.generation.GenerationMixin.generate", _t((1, 1, 4096)), True),
            }
        },
        ("test_log_api_B002", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "torch_silu_eager":    ("__B002__torch.nn.functional.silu",                   _t((1, 1, 16384)), False),
                "torch_silu_compiled": ("__B002__torch.nn.functional.silu",                   _t((1, 1, 16384)), True),
            }
        },
        ("test_log_api_B003", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "torch_sdpa_eager":    ("__B003__torch.nn.functional.scaled_dot_product_attention", _t((1, 32, 1, 128)), False),
                "torch_sdpa_compiled": ("__B003__torch.nn.functional.scaled_dot_product_attention", _t((1, 32, 1, 128)), True),
            }
        },
        ("test_log_api_B004", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "torch_compile_eager":    ("__B004__torch.compile",                           _t((1, 14, 4096)), False),
                "torch_compile_compiled": ("__B004__torch.compile",                           _t((1, 14, 4096)), True),
            }
        },

        # ── GROUP C: runtime / inference-stack keys ───────────────────────────
        ("test_log_api_C000", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "spyre_step_eager":    ("__C000__spyre.engine.LLMEngine.step",                _t((1, 1, 4096)), False),
                "spyre_step_compiled": ("__C000__spyre.engine.LLMEngine.step",                _t((1, 1, 4096)), True),
            }
        },
        ("test_log_api_C001", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "vllm_forward_eager":    ("__C001__vllm.MistralForCausalLM",                  _t((1, 1, 4096)), False),
                "vllm_forward_compiled": ("__C001__vllm.MistralForCausalLM",                  _t((1, 1, 4096)), True),
            }
        },

        # ── GROUP D: dynamic layer-index keys ────────────────────────────────
        ("test_log_api_D000", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "layer0_eager":    ("__D000__ministral.layer.0.forward",  _t((1, 14, 4096)), False),
                "layer0_compiled": ("__D000__ministral.layer.0.forward",  _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_D001", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "layer19_eager":    ("__D001__ministral.layer.19.forward", _t((1, 14, 4096)), False),
                "layer19_compiled": ("__D001__ministral.layer.19.forward", _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_D002", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "layer39_eager":    ("__D002__ministral.layer.39.forward", _t((1, 14, 4096)), False),
                "layer39_compiled": ("__D002__ministral.layer.39.forward", _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_D003", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "import_eager":    ("__D003__ministral.modeling_ministral3.import", _t((1, 14, 4096)), False),
                "import_compiled": ("__D003__ministral.modeling_ministral3.import", _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_D004", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "tgi_forward_eager":    ("__D004__tgi.MistralModel.forward", _t((1, 14, 4096)), False),
                "tgi_forward_compiled": ("__D004__tgi.MistralModel.forward", _t((1, 14, 4096)), True),
            }
        },
        ("test_log_api_D005", "_run_api_log_with_tensor_test"): {
            "param_sets": {
                "python_torch_eager":    ("__D005__python.torch", _t((1, 14, 4096)), False),
                "python_torch_compiled": ("__D005__python.torch", _t((1, 14, 4096)), True),
            }
        },

        # ── GROUP E: return-value contract (CPU-only, no mode) ───────────────
        ("test_log_api_E000", "_run_returns_none_test"): {
            "param_sets": {
                "ministral_model":  ("__E000__ministral.MistralModel.forward",),
                "hf_pretrained":    ("__E000__transformers.PreTrainedModel.forward",),
                "torch_silu":       ("__E000__torch.nn.functional.silu",),
                "spyre_step":       ("__E000__spyre.engine.LLMEngine.step",),
            }
        },

        # ── GROUP F: idempotency (CPU-only, no mode) ──────────────────────────
        ("test_log_api_F000", "_run_idempotent_test"): {
            "param_sets": {
                "model_forward_10x": ("ministral.MistralModel.forward",  10),
                "silu_10x":          ("torch.nn.functional.silu",        10),
                "mlp_50x":           ("ministral.MistralMLP.forward",    50),
            }
        },

        # ── GROUP G: threading safety (CPU-only, no mode) ─────────────────────
        ("test_log_api_G000", "_run_thread_safety_test"): {
            "param_sets": {
                "threads_8_keys_25":  (8,  25, "__G000__"),
                "threads_16_keys_50": (16, 50, "__G000b__"),
            }
        },

        # ── GROUP H: non-string type guard (CPU-only, no mode) ────────────────
        ("test_log_api_H000", "_run_type_error_test"): {
            "param_sets": {
                "non_str_int":    (42,),
                "non_str_none":   (None,),
                "non_str_list":   (["ministral.forward"],),
                "non_str_tensor": (torch.tensor(1),),
                "non_str_float":  (3.14,),
            }
        },
    }

    def _run_api_log_with_tensor_test(self, key, tensor, compiled):
        """Fire _log_api_usage_once inside a tensor clone — eager or compiled."""
        def fn(t):
            out = t.clone()
            torch._C._log_api_usage_once(key)
            return out
        compare_with_cpu(fn, tensor, compiled=compiled)

    def _run_returns_none_test(self, key):
        """_log_api_usage_once must always return None."""
        result = torch._C._log_api_usage_once(key)
        assert result is None, (
            f"_log_api_usage_once({key!r}) returned {type(result).__name__}"
        )

    def _run_idempotent_test(self, key, repeat_count):
        """Calling N times with the same key must never raise."""
        errors = []
        for i in range(repeat_count):
            try:
                torch._C._log_api_usage_once(key)
            except Exception as exc:
                errors.append(f"  call {i}: {type(exc).__name__}: {exc}")
        assert not errors, (
            f"_log_api_usage_once({key!r}) raised on repeated calls:\n"
            + "\n".join(errors)
        )

    def _run_thread_safety_test(self, num_threads, keys_per_thread, prefix):
        """Concurrent calls from multiple threads must not raise."""
        shared_keys = [f"{prefix}shared_{i}" for i in range(min(10, keys_per_thread))]
        exceptions  = []
        lock        = threading.Lock()

        def worker(tid):
            private = [f"{prefix}thread{tid}_key{i}"
                       for i in range(keys_per_thread - len(shared_keys))]
            for key in shared_keys + private:
                try:
                    torch._C._log_api_usage_once(key)
                except Exception as exc:
                    with lock:
                        exceptions.append(f"  thread={tid} key={key!r}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not exceptions, "Thread-safety failures:\n" + "\n".join(exceptions)

    def _run_type_error_test(self, bad_arg):
        """Non-string argument must raise TypeError."""
        with self.assertRaises(TypeError):
            torch._C._log_api_usage_once(bad_arg)

    def test_log_api_usage_once_is_callable(self):
        """_log_api_usage_once must be a callable C-extension."""
        assert callable(torch._C._log_api_usage_once)

    def test_log_api_usage_once_all_layer_keys_accepted(self):
        """All 40 decoder-layer keys must be accepted without error."""
        for layer_idx in range(NUM_LAYERS):
            key = f"__layer_check__ministral.layer.{layer_idx}.forward"
            try:
                torch._C._log_api_usage_once(key)
            except Exception as exc:
                self.fail(f"layer {layer_idx}: {exc}")

    def test_log_api_usage_once_empty_string_does_not_crash(self):
        """Empty-string key must not segfault or hang."""
        try:
            result = torch._C._log_api_usage_once("")
            assert result is None
        except (ValueError, RuntimeError, TypeError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# TestCat
# ─────────────────────────────────────────────────────────────────────────────

class TestCat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.cat patterns observed in Ministral-3-14B-Instruct-2512.
    """

    pytestmark = pytest.mark.torch_cat

    torch.manual_seed(0)

    PARAMS = {
        ("test_torch_cat_pattern_000", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    # [torch.randn(1, 14, 64), torch.randn(1, 14, 64)], -1, False,
                    [make_strided_tensor((1, 14, 64),(896,1,14),torch.float16), make_strided_tensor((1, 14, 64),(896,1,14),torch.float16)], -1, False,
                ),
                "long_64_compiled": (
                    [make_strided_tensor((1, 14, 64),(896,1,14),torch.float16), make_strided_tensor((1, 14, 64),(896,1,14),torch.float16)], -1, True,
                ),
            }
        },
        ("test_torch_cat_pattern_001", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    [make_strided_tensor((1, 32, 14, 64),(28672, 64, 2048, 1),torch.float16), make_strided_tensor((1, 32, 14, 64),(57344, 128, 4096, 1),torch.float16)], -1, False,
                ),
                "long_64_compiled": (
                    [make_strided_tensor((1, 32, 14, 64),(28672, 64, 2048, 1),torch.float16), make_strided_tensor((1, 32, 14, 64),(57344, 128, 4096, 1),torch.float16)], -1, True,
                ),
            }
        },
        ("test_torch_cat_pattern_002", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    [make_strided_tensor((1, 8, 14, 64),(7168, 64, 512, 1),torch.float16), make_strided_tensor((1, 8, 14, 64),(14336, 128, 1024, 1),torch.float16)], -1, False,
                ),
                "long_64_compiled": (
                    [make_strided_tensor((1, 8, 14, 64),(7168, 64, 512, 1),torch.float16), make_strided_tensor((1, 8, 14, 64),(14336, 128, 1024, 1),torch.float16)], -1, True,
                ),
            }
        },
        ("test_torch_cat_pattern_003", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    [torch.zeros(0), make_strided_tensor((1, 8, 14, 64),(7168, 64, 512, 1),torch.float16)], -2, False,
                ),
                "long_64_compiled": (
                    [torch.zeros(0), make_strided_tensor((1, 8, 14, 64),(7168, 64, 512, 1),torch.float16)], -2, True,
                ),
            }
        },
        
        
        
        ("test_torch_cat_pattern_004", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    # [torch.randn(1, 14, 64), torch.randn(1, 14, 64)], -1, False,
                    [make_strided_tensor((1, 1, 64),(64,1,1),torch.float16), make_strided_tensor((1, 1, 64),(64,1,1),torch.float16)], -1, False,
                ),
                "long_64_compiled": (
                    [make_strided_tensor((1, 1, 64),(64,1,1),torch.float16), make_strided_tensor((1, 1, 64),(64,1,1),torch.float16)], -1, True,
                ),
            }
        },
        ("test_torch_cat_pattern_005", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    [make_strided_tensor((1, 32, 1, 64),(2048, 64, 64, 1),torch.float16), make_strided_tensor((1, 32, 1, 64),(4096, 128, 4096, 1),torch.float16)], -1, False,
                ),
                "long_64_compiled": (
                    [make_strided_tensor((1, 32, 1, 64),(2048, 64, 64, 1),torch.float16), make_strided_tensor((1, 32, 1, 64),(4096, 128, 4096, 1),torch.float16)], -1, True,
                ),
            }
        },
        ("test_torch_cat_pattern_006", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    [make_strided_tensor((1, 8, 1, 64),(512, 64, 64, 1),torch.float16), make_strided_tensor((1, 8, 1, 64),(1024, 128, 1024, 1),torch.float16)], -1, False,
                ),
                "long_64_compiled": (
                    [make_strided_tensor((1, 8, 1, 64),(512, 64, 64, 1),torch.float16), make_strided_tensor((1, 8, 1, 64),(1024, 128, 1024, 1),torch.float16)], -1, True,
                ),
            }
        },
        ("test_torch_cat_pattern_007", "_run_cat_test"): {
            "param_sets": {
                "long_64_eager": (
                    [torch.zeros(0), make_strided_tensor((1, 8, 1, 64),(512, 64, 64, 1),torch.float16)], -2, False,
                ),
                "long_64_compiled": (
                    [torch.zeros(0), make_strided_tensor((1, 8, 1, 64),(512, 64, 64, 1),torch.float16)], -2, True,
                ),
            }
        },
    }

    def _run_cat_test(self, tensors, dim, compiled):
        fp16_tensors = [t.to(torch.float16) for t in tensors]

        def cat_fn(*ts):
            return torch.cat(list(ts), dim=dim)

        compare_with_cpu(cat_fn, *fp16_tensors, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestZeros
# ─────────────────────────────────────────────────────────────────────────────

class TestZeros(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.zeros patterns observed in Ministral-3-14B-Instruct-2512.

    No ops_dict is used here — torch.zeros is hardcoded inside each base
    method.  ops_dict is only appropriate when the op itself varies across
    cases (e.g. testing torch.add vs torch.mul with the same body).

    pytestmark stamps every generated method with @pytest.mark.torch_zeros
    regardless of base method name, avoiding spurious marks like
    torch_zeros_list_shape that would otherwise be derived from base names
    such as _run_zeros_list_shape_test.
    """

    pytestmark = pytest.mark.torch_zeros

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  float16
        # ------------------------------------------------------------------
        ("test_torch_zeros_pattern_000", "_run_zeros_test"): {
            "param_sets": {
                "zeros_4d_fp16_eager":    ((1, 8, 2048, 128), torch.float16, False),
                "zeros_4d_fp16_compiled": ((1, 8, 2048, 128), torch.float16, True),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  list shape input
        # ------------------------------------------------------------------
        ("test_torch_zeros_pattern_001", "_run_zeros_list_shape_test"): {
            "param_sets": {
                "zeros_4d_list_eager":    ((1, 8, 2048, 128), False),
                "zeros_4d_list_compiled": ((1, 8, 2048, 128), True),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  tuple shape input
        # ------------------------------------------------------------------
        ("test_torch_zeros_pattern_002", "_run_zeros_tuple_shape_test"): {
            "param_sets": {
                "zeros_4d_tuple_eager":    ((1, 8, 2048, 128), False),
                "zeros_4d_tuple_compiled": ((1, 8, 2048, 128), True),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  requires_grad
        # ------------------------------------------------------------------
        ("test_torch_zeros_pattern_003", "_run_zeros_requires_grad_test"): {
            "param_sets": {
                "zeros_4d_grad_eager":    ((1, 8, 2048, 128), False),
                "zeros_4d_grad_compiled": ((1, 8, 2048, 128), True),
            },
        },
        # ------------------------------------------------------------------
        # pattern_004  out= parameter
        # ------------------------------------------------------------------
        ("test_torch_zeros_pattern_004", "_run_zeros_out_test"): {
            "param_sets": {
                "zeros_4d_out_eager":    ((1, 8, 2048, 128), False),
                "zeros_4d_out_compiled": ((1, 8, 2048, 128), True),
            },
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — no ops_dict, op is hardcoded inside each method.
    # ------------------------------------------------------------------

    def _run_zeros_test(self, shape, dtype, compiled):
        """
        torch.zeros with a fixed dtype — runs both eager and compiled paths.

        Args:
            shape: tuple of ints.
            dtype: torch.dtype or None (None = PyTorch default float32).
        """
        def zeros_fn(*size, device=None):
            kwargs = {"device": device}
            if dtype is not None:
                kwargs["dtype"] = dtype
            return torch.zeros(*size, **kwargs)

        compare_with_cpu(zeros_fn, *shape, compiled=compiled,  needs_device=True)

    def _run_zeros_list_shape_test(self, shape, compiled):
        """torch.zeros with shape passed as a Python list."""
        def zeros_list_fn(device=None):
            return torch.zeros(list(shape), device=device)

        compare_with_cpu(zeros_list_fn, compiled=compiled, needs_device=True)

    def _run_zeros_tuple_shape_test(self, shape, compiled):
        """torch.zeros with shape passed as a single tuple."""
        def zeros_tuple_fn(device=None):
            return torch.zeros(shape, device=device)

        compare_with_cpu(zeros_tuple_fn, compiled=compiled, needs_device=True)

    def _run_zeros_requires_grad_test(self, shape, compiled):
        """torch.zeros with requires_grad=True — values stay zero, grad enabled."""
        def zeros_grad_fn(*size, device=None):
            return torch.zeros(*size, device=device, requires_grad=True)

        compare_with_cpu(zeros_grad_fn, *shape, compiled=compiled, needs_device=True)
    
    def _run_zeros_out_test(self, shape, compiled):
        """
        torch.zeros with out= parameter.

        out= changes the call signature so compare_with_cpu cannot be used
        directly.  Instead we run the op on each device manually and compare.
        """
        cpu_out    = torch.empty(*shape, device="cpu")
        cpu_result = torch.zeros(*shape, out=cpu_out)
        assert cpu_result is cpu_out, "CPU: out= did not return the same tensor"

        spyre_out    = torch.empty(*shape, device=DEVICE)
        spyre_result = torch.zeros(*shape, out=spyre_out)
        assert spyre_result is spyre_out, f"{DEVICE}: out= did not return the same tensor"

        torch.testing.assert_close(
            spyre_result.to("cpu"),
            cpu_result,
            msg=(
                f"\nCPU vs {DEVICE} mismatch for torch.zeros with out= parameter.\n"
                f"  Shape: {shape}"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestAdd
# ─────────────────────────────────────────────────────────────────────────────

class TestAdd(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.add patterns observed in Ministral-3-14B-Instruct-2512.

    Shapes are sourced directly from Ministral-3-14B-Instruct-2512_spyre.yaml.
    Three call signatures appear in the model:

      binary      torch.add(tensor, tensor)
      scalar      torch.add(tensor, scalar)  or  torch.add(scalar, tensor)
      alpha       torch.add(tensor, tensor, alpha=value)
      inplace     tensor.add_(tensor)

    pytestmark stamps every generated method with @pytest.mark.torch_add.
    """

    pytestmark = pytest.mark.torch_add

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # Binary tensor + tensor patterns
        # ------------------------------------------------------------------

        # add.2  q_embed rotary prefill: [1,32,14,128] + [1,32,14,128]
        ("test_torch_add_pattern_000", "_run_add_binary"): {
            "param_sets": {
                "binary_1x32x14x128_eager": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                     make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x32x14x128_compiled": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # add.3  k_embed rotary prefill: [1,8,14,128] + [1,8,14,128]
        ("test_torch_add_pattern_001", "_run_add_binary"): {
            "param_sets": {
                "binary_1x8x14x128_eager": (
                    make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 8, 14, 128), (14336, 1792, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x8x14x128_compiled": (
                    make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 8, 14, 128), (14336, 1792, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # add.5  residual + hidden prefill: [1,14,5120] + [1,14,5120]
        ("test_torch_add_pattern_002", "_run_add_binary"): {
            "param_sets": {
                "binary_1x14x5120_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    False,
                ),
                "binary_1x14x5120_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    True,
                ),
            }
        },
        # add.7  q_embed rotary decode: [1,32,1,128] + [1,32,1,128]
        ("test_torch_add_pattern_003", "_run_add_binary"): {
            "param_sets": {
                "binary_1x32x1x128_eager": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x32x1x128_compiled": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # add.8  k_embed rotary decode: [1,8,1,128] + [1,8,1,128]
        ("test_torch_add_pattern_004", "_run_add_binary"): {
            "param_sets": {
                "binary_1x8x1x128_eager": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x8x1x128_compiled": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # add.10  residual + hidden decode: [1,1,5120] + [1,1,5120]
        ("test_torch_add_pattern_005", "_run_add_binary"): {
            "param_sets": {
                "binary_1x1x5120_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    False,
                ),
                "binary_1x1x5120_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # Scalar patterns  (tensor + scalar  or  scalar + tensor)
        # ------------------------------------------------------------------

        # add.1  variance epsilon prefill: [1,14,1] + 1e-5 
        ("test_torch_add_pattern_006", "_run_add_scalar"): {
            "param_sets": {
                "scalar_1x14x1_eps_eager": (
                    make_strided_tensor((1, 14, 1), (14, 1, 1), torch.float16),
                    1e-5,
                    False,
                ),
                "scalar_1x14x1_eps_compiled": (
                    make_strided_tensor((1, 14, 1), (14, 1, 1), torch.float16),
                    1e-5,
                    True,
                ),
            }
        },
        # add.6  variance epsilon decode: [1,1,1] + 1e-5
        ("test_torch_add_pattern_007", "_run_add_scalar"): {
            "param_sets": {
                "scalar_1x1x1_eps_eager": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16),       
                    1e-5,
                    False,
                ),
                "scalar_1x1x1_eps_compiled": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16),
                    1e-5,
                    True,
                ),
            }
        },
        # add.4  attn scale prefill: 1 + tensor[14]  (scalar on left)
        ("test_torch_add_pattern_008", "_run_add_scalar_left"): {
            "param_sets": {
                "scalar_left_1_plus_14_eager": (
                    1,
                    make_strided_tensor((14,), (1,), torch.float16),
                    False,
                ),
                "scalar_left_1_plus_14_compiled": (
                    1,
                   make_strided_tensor((14,), (1,), torch.float16),
                    True,
                ),
            }
        },
        # add.9  attn scale decode: 1 + tensor[1]  (scalar on left)
        ("test_torch_add_pattern_009", "_run_add_scalar_left"): {
            "param_sets": {
                "scalar_left_1_plus_1_eager": (
                    1,
                    make_strided_tensor((1,), (1,), torch.float16),
                    False,
                ),
                "scalar_left_1_plus_1_compiled": (
                    1,
                    make_strided_tensor((1,), (1,), torch.float16),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # Alpha  torch.add(a, b, alpha=value)
        # ------------------------------------------------------------------

        # Representative shape from add.2 with non-unit alpha
        ("test_torch_add_pattern_010", "_run_add_alpha"): {
            "param_sets": {
                "alpha_2_1x32x14x128_eager": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16),
                    2.0,
                    False,
                ),
                "alpha_2_1x32x14x128_compiled": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16),
                    2.0,
                    True,
                ),
                "alpha_0_1x32x14x128_eager": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16),
                    0.0,
                    False,
                ),
            }
        },

        # ------------------------------------------------------------------
        # In-place  tensor.add_(tensor)
        # ------------------------------------------------------------------

        # Representative shape from add.5 (residual update)
        ("test_torch_add_pattern_011", "_run_add_inplace"): {
            "param_sets": {
                "inplace_1x14x5120_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16,fill="zeros"),
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    False,
                ),
                "inplace_1x14x5120_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16,fill="zeros"),
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    True,
                ),
                # decode shape
                "inplace_1x1x5120_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16,fill="zeros"),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    False,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_add_binary(self, a, b, compiled):
        """torch.add(tensor, tensor) — both operands are tensors."""
        compare_with_cpu(torch.add, a, b, compiled=compiled)

    def _run_add_scalar(self, a, scalar, compiled):
        """torch.add(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.add(x, scalar),
            a,
            compiled=compiled,
        )

    def _run_add_scalar_left(self, scalar, b, compiled):
        """torch.add(scalar, tensor) — scalar on left, tensor on right."""
        compare_with_cpu(
            lambda x: torch.add(scalar, x),
            b,
            compiled=compiled,
        )

    def _run_add_alpha(self, a, b, alpha, compiled):
        """torch.add(tensor, tensor, alpha=value) — scaled second operand."""
        compare_with_cpu(
            lambda x, y: torch.add(x, y, alpha=alpha),
            a, b,
            compiled=compiled,
        )

    def _run_add_inplace(self, dst, src, compiled):
        """tensor.add_(tensor) — in-place addition; return value must be same object."""
        def fn(d, s):
            d = d.clone()
            out = d.add_(s)
            assert out.data_ptr() == d.data_ptr(), (
                "add_: return value is not the same tensor as dst"
            )
            return out

        compare_with_cpu(fn, dst, src, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestMul
# ─────────────────────────────────────────────────────────────────────────────

class TestMul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.mul patterns observed in Ministral-3-14B-Instruct-2512.

    Shapes sourced from Ministral-3-14B-Instruct-2512_spyre.yaml.
    Three call signatures appear in the model:

      binary       torch.mul(tensor, tensor)   — same or broadcast shapes
      scalar_right torch.mul(tensor, scalar)
      scalar_left  torch.mul(scalar, tensor)

    yaml entries covered:
      mul.1   [1,14,128]   * 1.0          scalar_right  attention_scaling
      mul.2   [1,14,5120]  * [1,14,1]     binary        rsqrt normalisation
      mul.3   [5120]       * [1,14,5120]  binary        weight * hidden (broadcast)
      mul.4   [1,32,14,128]* [1,1,14,128] binary        q * cos (broadcast)
      mul.5   [1,8,14,128] * [1,1,14,128] binary        k * cos (broadcast)
      mul.6   0.1          * [14]         scalar_left   beta scaling
      mul.7   [1,32,14,128]* [14,1]       binary        query * attn_scale (broadcast)
      mul.8   [1,14,16384] * [1,14,16384] binary        gate * up (elementwise)
      mul.9   [1,1,128]    * 1.0          scalar_right  attention_scaling decode
      mul.10  [1,1,5120]   * [1,1,1]      binary        rsqrt normalisation decode
      mul.11  [5120]       * [1,1,5120]   binary        weight * hidden decode
      mul.12  [1,32,1,128] * [1,1,1,128]  binary        q * cos decode (broadcast)
      mul.13  [1,8,1,128]  * [1,1,1,128]  binary        k * cos decode (broadcast)
      mul.14  0.1          * [1]          scalar_left   beta scaling decode
      mul.15  [1,32,1,128] * [1,1]        binary        query * attn_scale decode
      mul.16  [1,1,16384]  * [1,1,16384]  binary        gate * up decode
    """

    pytestmark = pytest.mark.torch_mul

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # mul.1  [1,14,128] * scalar 1.0  — attention_scaling prefill
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_000", "_run_mul_scalar_right"): {
            "param_sets": {
                "scalar_right_1x14x128_eager": (
                    make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), 1.0, False,
                ),
                "scalar_right_1x14x128_compiled": (
                    make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), 1.0, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.2  [1,14,5120] * [1,14,1]  — rsqrt normalisation prefill
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_001", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x14x5120_1x14x1_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1, 14, 1), (14, 1, 1), torch.float16),
                    False,
                ),
                "binary_1x14x5120_1x14x1_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1, 14, 1), (14, 1, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.3  [5120] * [1,14,5120]  — weight * hidden prefill (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_002", "_run_mul_binary"): {
            "param_sets": {
                "binary_5120_1x14x5120_eager": (
                    make_strided_tensor((5120,), (1,), torch.float16),
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    False,
                ),
                "binary_5120_1x14x5120_compiled": (
                    make_strided_tensor((5120,), (1,), torch.float16),
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.4  [1,32,14,128] * [1,1,14,128]  — q * cos prefill (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_003", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x14x128_1x1x14x128_eager": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 14, 128), (1792, 1792, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x32x14x128_1x1x14x128_compiled": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 14, 128), (1792, 1792, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.5  [1,8,14,128] * [1,1,14,128]  — k * cos prefill (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_004", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x8x14x128_1x1x14x128_eager": (
                    make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 14, 128), (1792, 1792, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x8x14x128_1x1x14x128_compiled": (
                    make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 14, 128), (1792, 1792, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.6  scalar 0.1 * [14]  — beta scaling prefill (scalar left)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_005", "_run_mul_scalar_left"): {
            "param_sets": {
                "scalar_left_0p1_x_14_eager": (
                    0.1, make_strided_tensor((14,), (1,), torch.float16), False,
                ),
                "scalar_left_0p1_x_14_compiled": (
                    0.1, make_strided_tensor((14,), (1,), torch.float16), True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.7  [1,32,14,128] * [14,1]  — query * attn_scale prefill (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_006", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x14x128_14x1_eager": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((14, 1), (1, 1), torch.float16),
                    False,
                ),
                "binary_1x32x14x128_14x1_compiled": (
                    make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16),
                    make_strided_tensor((14, 1), (1, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.8  [1,14,16384] * [1,14,16384]  — gate * up prefill (elementwise)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_007", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x14x16384_eager": (
                    make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    False,
                ),
                "binary_1x14x16384_compiled": (
                    make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.9  [1,1,128] * scalar 1.0  — attention_scaling decode
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_008", "_run_mul_scalar_right"): {
            "param_sets": {
                "scalar_right_1x1x128_eager": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), 1.0, False,
                ),
                "scalar_right_1x1x128_compiled": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), 1.0, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.10  [1,1,5120] * [1,1,1]  — rsqrt normalisation decode
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_009", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x1x5120_1x1x1_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16),
                    False,
                ),
                "binary_1x1x5120_1x1x1_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.11  [5120] * [1,1,5120]  — weight * hidden decode (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_010", "_run_mul_binary"): {
            "param_sets": {
                "binary_5120_1x1x5120_eager": (
                    make_strided_tensor((5120,), (1,), torch.float16),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    False,
                ),
                "binary_5120_1x1x5120_compiled": (
                    make_strided_tensor((5120,), (1,), torch.float16),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.12  [1,32,1,128] * [1,1,1,128]  — q * cos decode (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_011", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x1x128_1x1x1x128_eager": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x32x1x128_1x1x1x128_compiled": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.13  [1,8,1,128] * [1,1,1,128]  — k * cos decode (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_012", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x8x1x128_1x1x1x128_eager": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x8x1x128_1x1x1x128_compiled": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.14  scalar 0.1 * [1]  — beta scaling decode (scalar left)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_013", "_run_mul_scalar_left"): {
            "param_sets": {
                "scalar_left_0p1_x_1_eager": (
                    0.1, make_strided_tensor((1,), (1,), torch.float16), False,
                ),
                "scalar_left_0p1_x_1_compiled": (
                    0.1, make_strided_tensor((1,), (1,), torch.float16), True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.15  [1,32,1,128] * [1,1]  — query * attn_scale decode (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_014", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x1x128_1x1_eager": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16),
                    make_strided_tensor((1, 1), (1,1), torch.float16),
                    False,
                ),
                "binary_1x32x1x128_1x1_compiled": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16),
                    make_strided_tensor((1, 1), (1,1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.16  [1,1,16384] * [1,1,16384]  — gate * up decode (elementwise)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_015", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x1x16384_eager": (
                    make_strided_tensor((1, 1, 16384), (16384,16384, 1), torch.float16),
                    make_strided_tensor((1, 1, 16384), (16384,16384, 1), torch.float16),
                    False,
                ),
                "binary_1x1x16384_compiled": (
                    make_strided_tensor((1, 1, 16384), (16384,16384, 1), torch.float16),
                    make_strided_tensor((1, 1, 16384), (16384,16384, 1), torch.float16),
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_mul_binary(self, a, b, compiled):
        """torch.mul(tensor, tensor) — both operands are tensors."""
        compare_with_cpu(torch.mul, a, b, compiled=compiled)

    def _run_mul_scalar_right(self, a, scalar, compiled):
        """torch.mul(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.mul(x, scalar),
            a,
            compiled=compiled,
        )

    def _run_mul_scalar_left(self, scalar, b, compiled):
        """torch.mul(scalar, tensor) — scalar on left, tensor on right."""
        compare_with_cpu(
            lambda x: torch.mul(scalar, x),
            b,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestNeg
# ─────────────────────────────────────────────────────────────────────────────

class TestNeg(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.neg patterns observed in Ministral-3-14B-Instruct-2512.

    Shapes sourced from Ministral-3-14B-Instruct-2512_spyre.yaml.

    yaml entries covered:
      neg.1  [1,32,14,64]  rotate_half q prefill
      neg.2  [1,8,14,64]   rotate_half k prefill
      neg.3  [1,32,1,64]   rotate_half q decode
      neg.4  [1,8,1,64]    rotate_half k decode
    """

    pytestmark = pytest.mark.torch_neg

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # neg.1  [1,32,14,64]  — rotate_half q prefill
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_000", "_run_neg_test"): {
            "param_sets": {
                "neg_1x32x14x64_eager": (
                    make_strided_tensor((1, 32, 14, 64), (57344, 128, 4096, 1), torch.float16), False,
                    
                ),
                "neg_1x32x14x64_compiled": (
                    make_strided_tensor((1, 32, 14, 64), (57344, 128, 4096, 1), torch.float16), True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # neg.2  [1,8,14,64]  — rotate_half k prefill
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_001", "_run_neg_test"): {
            "param_sets": {
                "neg_1x8x14x64_eager": (
                     make_strided_tensor((1, 8, 14, 64), (14336, 128, 1024, 1), torch.float16), False,
                ),
                "neg_1x8x14x64_compiled": (
                    make_strided_tensor((1, 8, 14, 64), (14336, 128, 1024, 1), torch.float16), True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # neg.3  [1,32,1,64]  — rotate_half q decode
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_002", "_run_neg_test"): {
            "param_sets": {
                "neg_1x32x1x64_eager": (
                    make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), torch.float16), False,
                ),
                "neg_1x32x1x64_compiled": (
                     make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), torch.float16), True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # neg.4  [1,8,1,64]  — rotate_half k decode
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_003", "_run_neg_test"): {
            "param_sets": {
                "neg_1x8x1x64_eager": (
                     make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), torch.float16), False,
                ),
                "neg_1x8x1x64_compiled": (
                    make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), torch.float16), True,
                ),
            }
        },
    }

    def _run_neg_test(self, a, compiled):
        """torch.neg(tensor) — elementwise negation."""
        compare_with_cpu(torch.neg, a, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestPow
# ─────────────────────────────────────────────────────────────────────────────

class TestPow(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.pow patterns observed in Ministral-3-14B-Instruct-2512.

    Shapes sourced from Ministral-3-14B-Instruct-2512_spyre.yaml.
    The model always uses integer exponent 2 (variance computation).

    yaml entries covered:
      pow.1  [1,14,5120]  exponent=2   variance prefill
      pow.2  [1,1,5120]   exponent=2   variance decode
    """

    pytestmark = pytest.mark.torch_pow

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pow.1  [1,14,5120] ** 2  — variance prefill
        # ------------------------------------------------------------------
        ("test_torch_pow_pattern_000", "_run_pow_test"): {
            "param_sets": {
                "pow_1x14x5120_exp2_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), 2, False,
                ),
                "pow_1x14x5120_exp2_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), 2, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # pow.2  [1,1,5120] ** 2  — variance decode
        # ------------------------------------------------------------------
        ("test_torch_pow_pattern_001", "_run_pow_test"): {
            "param_sets": {
                "pow_1x1x5120_exp2_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 2, False,
                ),
                "pow_1x1x5120_exp2_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 2, False,
                ),
            }
        },
    }

    def _run_pow_test(self, a, exponent, compiled):
        """torch.pow(tensor, exponent) — elementwise power with scalar exponent."""
        compare_with_cpu(
            lambda x: torch.pow(x, exponent),
            a,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestRsqrt
# ─────────────────────────────────────────────────────────────────────────────

class TestRsqrt(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.rsqrt patterns observed in Ministral-3-14B-Instruct-2512.

    Shapes sourced from Ministral-3-14B-Instruct-2512_spyre.yaml.
    rsqrt is undefined for zero or negative values, so inputs are made
    strictly positive: torch.abs(x) + epsilon.

    yaml entries covered:
      rsqrt.1  [1,14,1]   variance + epsilon prefill
      rsqrt.2  [1,1,1]    variance + epsilon decode
    """

    pytestmark = pytest.mark.torch_rsqrt

    torch.manual_seed(0)

    # Small epsilon matching the model's variance_epsilon (1e-5),
    # added to abs(randn) to guarantee strictly positive inputs.
    _EPS = torch.finfo(torch.float16).eps  # 9.77e-4 — safely above zero

    PARAMS = {
        # ------------------------------------------------------------------
        # rsqrt.1  [1,14,1]  — variance prefill
        # ------------------------------------------------------------------
        ("test_torch_rsqrt_pattern_000", "_run_rsqrt_test"): {
            "param_sets": {
                "rsqrt_1x14x1_eager": (
                    torch.abs(make_strided_tensor((1, 14, 1), (14, 1, 1), torch.float16)) + _EPS,
                    False,
                ),
                "rsqrt_1x14x1_compiled": (
                   torch.abs(make_strided_tensor((1, 14, 1), (14, 1, 1), torch.float16)) + _EPS,
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # rsqrt.2  [1,1,1]  — variance decode
        # ------------------------------------------------------------------
        ("test_torch_rsqrt_pattern_001", "_run_rsqrt_test"): {
            "param_sets": {
                "rsqrt_1x1x1_eager": (
                  torch.abs(make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16)) + _EPS,
                    False,
                ),
                "rsqrt_1x1x1_compiled": (
                    torch.abs(make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16)) + _EPS,
                    True,
                ),
            }
        },
    }

    def _run_rsqrt_test(self, a, compiled):
        """torch.rsqrt(tensor) — elementwise reciprocal square root."""
        compare_with_cpu(torch.rsqrt, a, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestUnsqueeze
# ─────────────────────────────────────────────────────────────────────────────

class TestUnsqueeze(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.unsqueeze patterns observed in Ministral-3-14B-Instruct-2512.

    Shapes and dim values sourced from Ministral-3-14B-Instruct-2512_spyre.yaml.

    yaml entries covered:
      unsqueeze.1  [1,14,128]  dim=1   cos.unsqueeze(unsqueeze_dim) prefill
      unsqueeze.2  [14]        dim=-1  scaling.unsqueeze(-1) prefill
      unsqueeze.3  [1,1,128]   dim=1   cos.unsqueeze(unsqueeze_dim) decode
      unsqueeze.4  [1]         dim=-1  scaling.unsqueeze(-1) decode
    """

    pytestmark = pytest.mark.torch_unsqueeze

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # unsqueeze.1  [1,14,128]  dim=1  — cos prefill
        # ------------------------------------------------------------------
        ("test_torch_unsqueeze_pattern_000", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_1x14x128_dim1_eager": (
                    make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), 1, False,
                ),
                "unsqueeze_1x14x128_dim1_compiled": (
                   make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), 1, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # unsqueeze.2  [14]  dim=-1  — scaling prefill
        # ------------------------------------------------------------------
        ("test_torch_unsqueeze_pattern_001", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_14_dimneg1_eager": (
                    make_strided_tensor((14,), (1,), torch.float16), -1, False,
                ),
                "unsqueeze_14_dimneg1_compiled": (
                    make_strided_tensor((14,), (1,), torch.float16), -1, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # unsqueeze.3  [1,1,128]  dim=1  — cos decode
        # ------------------------------------------------------------------
        ("test_torch_unsqueeze_pattern_002", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_1x1x128_dim1_eager": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), 1, False,
                ),
                "unsqueeze_1x1x128_dim1_compiled": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), 1, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # unsqueeze.4  [1]  dim=-1  — scaling decode
        # ------------------------------------------------------------------
        ("test_torch_unsqueeze_pattern_003", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_1_dimneg1_eager": (
                    make_strided_tensor((1,), (1,), torch.float16), -1, False,
                ),
                "unsqueeze_1_dimneg1_compiled": (
                    make_strided_tensor((1,), (1,), torch.float16), -1, True,
                ),
            }
        },
    }

    def _run_unsqueeze_test(self, a, dim, compiled):
        """torch.unsqueeze(tensor, dim) — insert a dimension of size 1 at dim."""
        compare_with_cpu(
            lambda x: torch.unsqueeze(x, dim),
            a,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestFloor
# ─────────────────────────────────────────────────────────────────────────────

class TestFloor(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.floor patterns observed in Ministral-3-14B-Instruct-2512.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_floor_test``
    into one concrete test method per (pattern, eager/compiled) combination.
    - pytestmark = torch_floor selects the entire class with  pytest -m torch_floor.
    - Each generated method is also individually stamped with @pytest.mark.torch_floor
      by the metaclass (derived from _run_floor_test -> "floor" -> torch_floor).

    Input shapes : (14,) and (1,)
    dtype        : torch.float16

    Each param_set entry is a 3-tuple:
        (tensor: Tensor, op: callable, compiled: bool)
    Exception: _run_floor_special_values_test takes only (tensor,) — CPU-only.
    """

    pytestmark = pytest.mark.torch_floor

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  basic floor  len=14  t.floor()
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_000", "_run_floor_test"): {
            "param_sets": {
                "len14_eager":    (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.floor(),          False),
                "len14_compiled": (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.floor(),          True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_001  basic floor  len=1  t.floor()
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_001", "_run_floor_test"): {
            "param_sets": {
                "len1_eager":    (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.floor(),             False),
                "len1_compiled": (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.floor(),             True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_002  functional alias  len=14  torch.floor(t) == t.floor()
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_002", "_run_floor_test"): {
            "param_sets": {
                "len14_eager":    (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: torch.floor(t),      False),
                "len14_compiled": (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: torch.floor(t),      True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_003  functional alias  len=1  torch.floor(t) == t.floor()
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_003", "_run_floor_test"): {
            "param_sets": {
                "len1_eager":    (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: torch.floor(t),        False),
                "len1_compiled": (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: torch.floor(t),        True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_004  in-place variant  len=14  floor_() == floor()
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_004", "_run_floor_test"): {
            "param_sets": {
                "len14_eager":    (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.clone().floor_(),  False),
                "len14_compiled": (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.clone().floor_(),  True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_005  in-place variant  len=1  floor_() == floor()
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_005", "_run_floor_test"): {
            "param_sets": {
                "len1_eager":    (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.clone().floor_(),    False),
                "len1_compiled": (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.clone().floor_(),    True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_006  special values  len=14  ±inf, NaN, ±0, near-integers
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_006", "_run_floor_special_values_test"): {
            "param_sets": {
                "special_mixed_len14": (
                    torch.tensor(
                        [float("inf"), float("-inf"), float("nan"),
                         -0.0, 0.0, 0.5, -0.5, 1.9, -1.9,
                         65504.0, -65504.0, 0.9999, -0.9999, 1.0001],
                        dtype=torch.float16,
                    ),
                ),
            }
        },
        # ------------------------------------------------------------------
        # pattern_007  special values  len=1  single NaN passthrough
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_007", "_run_floor_special_values_test"): {
            "param_sets": {
                "special_nan_len1": (
                    torch.tensor([float("nan")], dtype=torch.float16),
                ),
            }
        },
        # ------------------------------------------------------------------
        # pattern_008  all-negative values  len=14
        # floor(-3.7) = -4 not -3 — explicit sign correctness check.
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_008", "_run_floor_test"): {
            "param_sets": {
                "len14_eager":    (torch.full((14,), -3.7, dtype=torch.float16), lambda t: t.floor(),        False),
                "len14_compiled": (torch.full((14,), -3.7, dtype=torch.float16), lambda t: t.floor(),        True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_009  all-negative values  len=1
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_009", "_run_floor_test"): {
            "param_sets": {
                "len1_eager":    (torch.full((1,), -3.7, dtype=torch.float16), lambda t: t.floor(),          False),
                "len1_compiled": (torch.full((1,), -3.7, dtype=torch.float16), lambda t: t.floor(),          True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_010  already-integer values  len=14
        # floor of an exact integer float must return itself unchanged.
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_010", "_run_floor_test"): {
            "param_sets": {
                "len14_eager":    (torch.arange(1, 15, dtype=torch.float16), lambda t: t.floor(),            False),
                "len14_compiled": (torch.arange(1, 15, dtype=torch.float16), lambda t: t.floor(),            True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_011  already-integer values  len=1
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_011", "_run_floor_test"): {
            "param_sets": {
                "len1_eager":    (torch.tensor([5.0], dtype=torch.float16), lambda t: t.floor(),             False),
                "len1_compiled": (torch.tensor([5.0], dtype=torch.float16), lambda t: t.floor(),             True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_012  floor -> int64 cast  len=14
        # Simulates position-index extraction: floor float -> integer id.
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_012", "_run_floor_test"): {
            "param_sets": {
                "len14_eager":    (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.floor().to(torch.int64), False),
                "len14_compiled": (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.floor().to(torch.int64), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_013  floor -> int64 cast  len=1
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_013", "_run_floor_test"): {
            "param_sets": {
                "len1_eager":    (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.floor().to(torch.int64),   False),
                "len1_compiled": (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.floor().to(torch.int64),   True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_014  floor -> clamp  len=14
        # Simulates integer-quantisation post-processing on floored activations.
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_014", "_run_floor_test"): {
            "param_sets": {
                "len14_eager":    (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.floor().clamp(min=-128, max=127), False),
                "len14_compiled": (make_strided_tensor((14,), (1,), torch.float16) * 10, lambda t: t.floor().clamp(min=-128, max=127), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_015  floor -> clamp  len=1
        # ------------------------------------------------------------------
        ("test_torch_floor_pattern_015", "_run_floor_test"): {
            "param_sets": {
                "len1_eager":    (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.floor().clamp(min=-128, max=127),   False),
                "len1_compiled": (make_strided_tensor((1,), (1,), torch.float16) * 10, lambda t: t.floor().clamp(min=-128, max=127),   True),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # ------------------------------------------------------------------

    def _run_floor_test(self, tensor, op, compiled):
        """
        Shared body for all torch.floor pattern tests.

        Wraps the given op in compare_with_cpu so that CPU and Spyre
        outputs are compared for both eager and compiled paths.

        Args:
            tensor:   float16 input tensor of shape (14,) or (1,).
            op:       callable applied to the tensor, e.g.:
                        lambda t: t.floor()
                        lambda t: torch.floor(t)
                        lambda t: t.clone().floor_()
                        lambda t: t.floor().to(torch.int64)
                        lambda t: t.floor().clamp(min=-128, max=127)
            compiled: True  -> torch.compile path
                      False -> eager path
        """
        compare_with_cpu(op, tensor, compiled=compiled)

    def _run_floor_special_values_test(self, tensor):
        """
        CPU-only structural check for IEEE 754 corner cases.

        Verifies that torch.floor agrees with math.floor on ±inf (unchanged),
        NaN (propagated), ±0, and near-integer boundary values.

        Args:
            tensor: float16 tensor containing special values.
        """
        t = tensor.cpu().float()
        result = t.floor()

        for idx in range(t.numel()):
            raw = t.view(-1)[idx].item()
            got = result.view(-1)[idx].item()

            if math.isnan(raw):
                assert math.isnan(got), (
                    f"floor(NaN) should be NaN, got {got} at index {idx}"
                )
            elif math.isinf(raw):
                assert got == raw, (
                    f"floor(±inf) should be ±inf, got {got} at index {idx}"
                )
            else:
                expected = math.floor(raw)
                assert got == expected, (
                    f"floor({raw}) expected {expected}, got {got} at index {idx}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TestLog
# ─────────────────────────────────────────────────────────────────────────────

class TestLog(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.log patterns observed in Ministral-3-14B-Instruct-2512.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_log_test``
    into one concrete test method per (pattern, eager/compiled) combination.
    - pytestmark = torch_log selects the entire class with  pytest -m torch_log.
    - Each generated method is also individually stamped with @pytest.mark.torch_log
      by the metaclass (derived from _run_log_test -> "log" -> torch_log).

    Input shapes : (14,) and (1,)
    dtype        : torch.float16

    Note: inputs are strictly positive (torch.abs + small epsilon) so that
    log(x) is real-valued and finite for all elements.

    Each param_set entry is a 3-tuple:
        (tensor: Tensor, op: callable, compiled: bool)
    Exception: _run_log_special_values_test takes only (tensor,) — CPU-only.
    """

    pytestmark = pytest.mark.torch_log

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  basic log  len=14  torch.log(t)
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_000", "_run_log_test"): {
            "param_sets": {
                "len14_eager":    (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t), False),
                "len14_compiled": (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_001  basic log  len=1  torch.log(t)
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_001", "_run_log_test"): {
            "param_sets": {
                "len1_eager":    (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t),   False),
                "len1_compiled": (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t),   True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_002  method alias  len=14  t.log() == torch.log(t)
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_002", "_run_log_test"): {
            "param_sets": {
                "len14_eager":    (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: t.log(),      False),
                "len14_compiled": (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: t.log(),      True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_003  method alias  len=1  t.log() == torch.log(t)
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_003", "_run_log_test"): {
            "param_sets": {
                "len1_eager":    (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: t.log(),        False),
                "len1_compiled": (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: t.log(),        True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_004  all-ones input  len=14  log(1.0) == 0.0 for all elements
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_004", "_run_log_test"): {
            "param_sets": {
                "len14_eager":    (make_strided_tensor((1,), (1,), torch.float16,fill="ones"), lambda t: torch.log(t),                    False),
                "len14_compiled": (make_strided_tensor((14,), (1,), torch.float16,fill="ones"), lambda t: torch.log(t),                    True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_005  all-ones input  len=1  log(1.0) == 0.0
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_005", "_run_log_test"): {
            "param_sets": {
                "len1_eager":    (make_strided_tensor((1,), (1,), torch.float16,fill="ones"), lambda t: torch.log(t),                      False),
                "len1_compiled": (make_strided_tensor((1,), (1,), torch.float16,fill="ones"), lambda t: torch.log(t),                      True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_006  large values  len=14
        # log of large positive floats — exercises upper fp16 range (max 65504).
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_006", "_run_log_test"): {
            "param_sets": {
                "len14_eager":    (torch.full((14,), 65504.0, dtype=torch.float16), lambda t: torch.log(t),        False),
                "len14_compiled": (torch.full((14,), 65504.0, dtype=torch.float16), lambda t: torch.log(t),        True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_007  large values  len=1
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_007", "_run_log_test"): {
            "param_sets": {
                "len1_eager":    (torch.full((1,), 65504.0, dtype=torch.float16), lambda t: torch.log(t),          False),
                "len1_compiled": (torch.full((1,), 65504.0, dtype=torch.float16), lambda t: torch.log(t),          True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_008  small positive values  len=14
        # log of values near zero — exercises lower fp16 range (min ~6e-5).
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_008", "_run_log_test"): {
            "param_sets": {
                "len14_eager":    (torch.full((14,), 1e-3, dtype=torch.float16), lambda t: torch.log(t),           False),
                "len14_compiled": (torch.full((14,), 1e-3, dtype=torch.float16), lambda t: torch.log(t),           True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_009  small positive values  len=1
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_009", "_run_log_test"): {
            "param_sets": {
                "len1_eager":    (torch.full((1,), 1e-3, dtype=torch.float16), lambda t: torch.log(t),             False),
                "len1_compiled": (torch.full((1,), 1e-3, dtype=torch.float16), lambda t: torch.log(t),             True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_010  log -> clamp  len=14
        # Simulates log-prob clamping: torch.log(t).clamp(min=-10).
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_010", "_run_log_test"): {
            "param_sets": {
                "len14_eager":    (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).clamp(min=-10), False),
                "len14_compiled": (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).clamp(min=-10), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_011  log -> clamp  len=1
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_011", "_run_log_test"): {
            "param_sets": {
                "len1_eager":    (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).clamp(min=-10),   False),
                "len1_compiled": (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).clamp(min=-10),   True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_012  log -> cast to float32  len=14
        # Simulates precision upcast after log for softmax / loss computation.
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_012", "_run_log_test"): {
            "param_sets": {
                "len14_eager":    (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).to(torch.float32), False),
                "len14_compiled": (torch.abs(make_strided_tensor((14,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).to(torch.float32), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_013  log -> cast to float32  len=1
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_013", "_run_log_test"): {
            "param_sets": {
                "len1_eager":    (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).to(torch.float32),   False),
                "len1_compiled": (torch.abs(make_strided_tensor((1,), (1,), torch.float16)) + 1e-3, lambda t: torch.log(t).to(torch.float32),   True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_014  special values  len=14  +inf, NaN, +0 (log boundary cases)
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_014", "_run_log_special_values_test"): {
            "param_sets": {
                "special_mixed_len14": (
                    torch.tensor(
                        [float("inf"), float("nan"), 0.0,
                         1e-3, 1.0, 2.0, 10.0, 100.0,
                         65504.0, 0.5, 0.25, 0.1, 1e-4, 1e-5],
                        dtype=torch.float16,
                    ),
                ),
            }
        },
        # ------------------------------------------------------------------
        # pattern_015  special values  len=1  log(0) == -inf
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_log_pattern_015", "_run_log_special_values_test"): {
            "param_sets": {
                "special_zero_len1": (
                    torch.tensor([0.0], dtype=torch.float16),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # ------------------------------------------------------------------

    def _run_log_test(self, tensor, op, compiled):
        """
        Shared body for all torch.log pattern tests.

        Wraps the given op in compare_with_cpu so that CPU and Spyre
        outputs are compared for both eager and compiled paths.

        Args:
            tensor:   float16 input tensor of shape (14,) or (1,).
            op:       callable applied to the tensor, e.g.:
                        lambda t: torch.log(t)
                        lambda t: t.log()
                        lambda t: torch.log(t).clamp(min=-10)
                        lambda t: torch.log(t).to(torch.float32)
            compiled: True  -> torch.compile path
                      False -> eager path
        """
        compare_with_cpu(op, tensor, compiled=compiled)

    def _run_log_special_values_test(self, tensor):
        """
        CPU-only structural check for log boundary cases.

        Verifies that torch.log agrees with math.log on representative
        values: +inf (unchanged), NaN (propagated), log(0) == -inf,
        log(1) == 0, and large/small positive values.

        Args:
            tensor: float16 tensor containing boundary values.
        """
        t = tensor.cpu().float()
        result = t.log()

        for idx in range(t.numel()):
            raw = t.view(-1)[idx].item()
            got = result.view(-1)[idx].item()

            if math.isnan(raw):
                assert math.isnan(got), (
                    f"log(NaN) should be NaN, got {got} at index {idx}"
                )
            elif math.isinf(raw) and raw > 0:
                assert math.isinf(got) and got > 0, (
                    f"log(+inf) should be +inf, got {got} at index {idx}"
                )
            elif raw == 0.0:
                assert math.isinf(got) and got < 0, (
                    f"log(0) should be -inf, got {got} at index {idx}"
                )
            elif raw > 0:
                expected = math.log(raw)
                assert math.isclose(got, expected, rel_tol=1e-2), (
                    f"log({raw}) expected {expected:.6f}, got {got:.6f} at index {idx}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TestMatmul
# ─────────────────────────────────────────────────────────────────────────────

class TestMatmul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.matmul patterns observed in Ministral-3-14B-Instruct-2512.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_matmul_test``
    into one concrete test method per (pattern, eager/compiled) combination.
    - pytestmark = torch_matmul selects the entire class with  pytest -m torch_matmul.
    - Each generated method is also individually stamped with @pytest.mark.torch_matmul
      by the metaclass (derived from _run_matmul_test -> "matmul" -> torch_matmul).

    Input shapes and output shapes:
        A: (1, 64, 1)  B: (1, 1,  1) -> out: (1, 64,  1)
        A: (1, 64, 1)  B: (1, 1, 14) -> out: (1, 64, 14)
        A: (1,  1, 1)  B: (1, 1,  1) -> out: (1,  1,  1)
        A: (1,  1, 1)  B: (1, 1, 14) -> out: (1,  1, 14)
    dtype : torch.float16

    Each param_set entry is a 4-tuple:
        (a: Tensor, b: Tensor, op: callable, compiled: bool)
    Exception: _run_matmul_special_values_test takes only (a, b) — CPU-only.
    """

    pytestmark = pytest.mark.torch_matmul

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  basic matmul  A:(1,64,1) @ B:(1,1,1) -> (1,64,1)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_000", "_run_matmul_test"): {
            "param_sets": {
                "1x64x1_1x1x1_eager":    (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16), lambda a, b: torch.matmul(a, b), False),
                "1x64x1_1x1x1_compiled": (make_strided_tensor((1, 64, 1), (1, 64, 1), torch.float16), make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16), lambda a, b: torch.matmul(a, b), True),
            }
        },


        # ------------------------------------------------------------------
        # pattern_004  method alias  A:(1,64,1) @ B:(1,1,14) -> (1,64,14)
        # a.matmul(b) == torch.matmul(a, b)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_004", "_run_matmul_test"): {
            "param_sets": {
                "method_1x64x1_1x1x14_eager": (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), lambda a, b: a.matmul(b), False),
                "method_1x64x1_1x1x14_compiled": (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), lambda a, b: a.matmul(b), True),
            }
        },


        # ------------------------------------------------------------------
        # pattern_008  all-zeros A  A:(1,64,1) @ B:(1,1,14) -> all-zeros out
        # matmul with zero matrix must produce zero output.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_008", "_run_matmul_test"): {
            "param_sets": {
                "zeros_a_1x64x1_1x1x14_eager":    (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16,fill="zeros"), make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), lambda a, b: torch.matmul(a, b), False),
                "zeros_a_1x64x1_1x1x14_compiled": (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16,fill="zeros"), make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), lambda a, b: torch.matmul(a, b), True),
            }
        },

        # ------------------------------------------------------------------
        # pattern_010  all-ones inputs  A:(1,64,1) @ B:(1,1,14) -> all-ones * 1
        # matmul of ones: each output element == 1.0 (inner dim = 1).
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_010", "_run_matmul_test"): {
            "param_sets": {
                "ones_1x64x1_1x1x14_eager":    (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16,fill="ones"), make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), lambda a, b: torch.matmul(a, b), False),
                "ones_1x64x1_1x1x14_compiled": (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16,fill="ones"), make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), lambda a, b: torch.matmul(a, b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_012  matmul -> add  A:(1,64,1) @ B:(1,1,14) + bias
        # Simulates linear projection with bias addition.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_012", "_run_matmul_test"): {
            "param_sets": {
                "add_bias_1x64x1_1x1x14_eager":    
                    (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), 
                     make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), 
                      lambda a, b: torch.matmul(a, b) + torch.ones(1, 64, 14, dtype=torch.float16), 
                      False),
                "add_bias_1x64x1_1x1x14_compiled": 
                    (make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), 
                     make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16), 
                     lambda a, b: torch.matmul(a, b) + torch.ones(1, 64, 14, dtype=torch.float16), 
                     True),
            }
        },

        # ------------------------------------------------------------------
        # pattern_014  special values  A:(1,64,1) @ B:(1,1,14)
        # CPU-only: +inf in A propagates to output; NaN in B poisons output.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_014", "_run_matmul_special_values_test"): {
            "param_sets": {
                "special_inf_a_1x64x1_1x1x14": (
                    torch.full((1, 64, 1),  float("inf"), dtype=torch.float16),
                   make_strided_tensor((1, 1, 14), (14, 14, 1), torch.float16,fill="ones"),
                ),
            }
        },
        # ------------------------------------------------------------------
        # pattern_015  special values  A:(1,1,1) @ B:(1,1,1)
        # CPU-only: NaN in A must propagate to all output elements.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_015", "_run_matmul_special_values_test"): {
            "param_sets": {
                "special_nan_a_1x1x1_1x1x1": (
                    torch.full((1, 1, 1), float("nan"), dtype=torch.float16),
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float16,fill="ones"),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # ------------------------------------------------------------------

    def _run_matmul_test(self, a, b, op, compiled):
        """
        Shared body for all torch.matmul pattern tests.

        Wraps the given op in compare_with_cpu so that CPU and Spyre
        outputs are compared for both eager and compiled paths.

        Args:
            a:        float16 tensor, first matmul operand.
            b:        float16 tensor, second matmul operand.
            op:       callable applied to (a, b), e.g.:
                        lambda a, b: torch.matmul(a, b)
                        lambda a, b: a.matmul(b)
                        lambda a, b: a @ b
                        lambda a, b: torch.matmul(a, b) + bias
            compiled: True  -> torch.compile path
                      False -> eager path
        """
        compare_with_cpu(op, a, b, compiled=compiled)

    def _run_matmul_special_values_test(self, a, b):
        """
        CPU-only structural check for matmul with special IEEE 754 values.

        Verifies that +inf in an input propagates to +inf in the output,
        and NaN in an input poisons all output elements to NaN.

        Args:
            a: float16 tensor, first matmul operand (may contain inf/NaN).
            b: float16 tensor, second matmul operand.
        """
        result = torch.matmul(a.cpu(), b.cpu())
        # True if any element in a or b is ±inf; inf * finite must stay inf
        # (NaN also acceptable when the other operand has a zero row/col).
        if torch.isinf(a).any() or torch.isinf(b).any():
            assert torch.isinf(result).any() or torch.isnan(result).any(), (
                f"Expected inf/nan in output when input contains inf, "
                f"got: {result}"
            )
        # NaN is contagious — any arithmetic with NaN must produce NaN.
        if torch.isnan(a).any() or torch.isnan(b).any():
            assert torch.isnan(result).any(), (
                f"Expected NaN in output when input contains NaN, "
                f"got: {result}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestMean
# ─────────────────────────────────────────────────────────────────────────────


class TestMean(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.mean patterns observed in Ministral-3-14B-Instruct-2512.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_mean_test``
    into one concrete test method per (pattern, eager/compiled) combination.
    - pytestmark = torch_mean selects the entire class with  pytest -m torch_mean.
    - Each generated method is also individually stamped with @pytest.mark.torch_mean
      by the metaclass (derived from _run_mean_test -> "mean" -> torch_mean).

    Input shapes and reduction outputs:
        (1,  1, 5120)  dim=-1 -> (1,  1,    1)   hidden-state mean (decode)
        (1, 14, 5120)  dim=-1 -> (1, 14,    1)   hidden-state mean (prefill)
        (1,  1, 5120)  dim=0  -> (1,  1, 5120)   batch mean (decode)
        (1, 14, 5120)  dim=0  -> (1, 14, 5120)   batch mean (prefill)
    dtype : torch.float16

    Each param_set entry is a 3-tuple:
        (tensor: Tensor, op: callable, compiled: bool)
    Exception: _run_mean_special_values_test takes only (tensor,) — CPU-only.
    """

    pytestmark = pytest.mark.torch_mean

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  basic mean  (1,1,5120)  no dim — global mean
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_000", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t), False),
                "1x1x5120_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_001  basic mean  (1,14,5120)  no dim — global mean
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_001", "_run_mean_test"): {
            "param_sets": {
                "1x14x5120_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t), False),
                "1x14x5120_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_002  mean over last dim  (1,1,5120)  dim=-1 -> (1,1,1)
        # Reduces hidden dimension — most common path in layer norm pre-step.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_002", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_dim-1_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1), False),
                "1x1x5120_dim-1_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_003  mean over last dim  (1,14,5120)  dim=-1 -> (1,14,1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_003", "_run_mean_test"): {
            "param_sets": {
                "1x14x5120_dim-1_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1), False),
                "1x14x5120_dim-1_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_004  mean over last dim keepdim  (1,1,5120)  -> (1,1,1)
        # keepdim=True preserves shape for downstream broadcast.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_004", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_keepdim_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1, keepdim=True), False),
                "1x1x5120_keepdim_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1, keepdim=True), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_005  mean over last dim keepdim  (1,14,5120)  -> (1,14,1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_005", "_run_mean_test"): {
            "param_sets": {
                "1x14x5120_keepdim_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1, keepdim=True), False),
                "1x14x5120_keepdim_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1, keepdim=True), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_006  mean over seq dim  (1,14,5120)  dim=1 -> (1,1,5120)
        # Reduces sequence dimension — pooling across tokens.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_006", "_run_mean_test"): {
            "param_sets": {
                "1x14x5120_dim1_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=1), False),
                "1x14x5120_dim1_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_007  mean over batch dim  (1,1,5120)  dim=0 -> (1,5120)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_007", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_dim0_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=0), False),
                "1x1x5120_dim0_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=0), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_008  method alias  (1,1,5120)  t.mean(dim=-1)
        # t.mean() == torch.mean(t) — both call sites appear in the model.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_008", "_run_mean_test"): {
            "param_sets": {
                "method_1x1x5120_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: t.mean(dim=-1), False),
                "method_1x1x5120_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: t.mean(dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_009  method alias  (1,14,5120)  t.mean(dim=-1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_009", "_run_mean_test"): {
            "param_sets": {
                "method_1x14x5120_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: t.mean(dim=-1), False),
                "method_1x14x5120_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: t.mean(dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_010  all-zeros input  (1,1,5120)  mean must be 0.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_010", "_run_mean_test"): {
            "param_sets": {
                "zeros_1x1x5120_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16, fill="zeros"), lambda t: torch.mean(t, dim=-1), False),
                "zeros_1x1x5120_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16, fill="zeros"), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_011  all-zeros input  (1,14,5120)  mean must be 0.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_011", "_run_mean_test"): {
            "param_sets": {
                "zeros_1x14x5120_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16, fill="zeros"), lambda t: torch.mean(t, dim=-1), False),
                "zeros_1x14x5120_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16, fill="zeros"), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_012  all-ones input  (1,1,5120)  mean must be 1.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_012", "_run_mean_test"): {
            "param_sets": {
                "ones_1x1x5120_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16, fill="ones"), lambda t: torch.mean(t, dim=-1), False),
                "ones_1x1x5120_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16, fill="ones"), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_013  all-ones input  (1,14,5120)  mean must be 1.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_013", "_run_mean_test"): {
            "param_sets": {
                "ones_1x14x5120_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16, fill="ones"), lambda t: torch.mean(t, dim=-1), False),
                "ones_1x14x5120_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16, fill="ones"), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_014  mean -> cast to float32  (1,1,5120)
        # Simulates precision upcast after mean for layer norm computation.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_014", "_run_mean_test"): {
            "param_sets": {
                "cast_fp32_1x1x5120_eager":    (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1).to(torch.float32), False),
                "cast_fp32_1x1x5120_compiled": (make_strided_tensor((1,  1, 5120), (5120, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1).to(torch.float32), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_015  mean -> cast to float32  (1,14,5120)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_015", "_run_mean_test"): {
            "param_sets": {
                "cast_fp32_1x14x5120_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1).to(torch.float32), False),
                "cast_fp32_1x14x5120_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), lambda t: torch.mean(t, dim=-1).to(torch.float32), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_016  special values  (1,1,5120)  NaN in input poisons mean
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_016", "_run_mean_special_values_test"): {
            "param_sets": {
                "special_nan_1x1x5120": (
                    torch.cat([
                        torch.tensor([float("nan")], dtype=torch.float16),
                        torch.ones(5119, dtype=torch.float16),
                    ]).reshape(1, 1, 5120),
                ),
            }
        },
        # ------------------------------------------------------------------
        # pattern_017  special values  (1,14,5120)  +inf in input -> +inf mean
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_017", "_run_mean_special_values_test"): {
            "param_sets": {
                "special_inf_1x14x5120": (
                    torch.cat([
                        torch.tensor([float("inf")], dtype=torch.float16),
                        torch.ones(14 * 5120 - 1, dtype=torch.float16),
                    ]).reshape(1, 14, 5120),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # ------------------------------------------------------------------

    def _run_mean_test(self, tensor, op, compiled):
        """
        Shared body for all torch.mean pattern tests.

        Wraps the given op in compare_with_cpu so that CPU and Spyre
        outputs are compared for both eager and compiled paths.

        Args:
            tensor:   float16 input tensor of shape (1,1,5120) or (1,14,5120).
            op:       callable applied to the tensor, e.g.:
                        lambda t: torch.mean(t)
                        lambda t: torch.mean(t, dim=-1)
                        lambda t: torch.mean(t, dim=-1, keepdim=True)
                        lambda t: t.mean(dim=-1)
                        lambda t: torch.mean(t, dim=-1).to(torch.float32)
            compiled: True  -> torch.compile path
                      False -> eager path
        """
        compare_with_cpu(op, tensor, compiled=compiled)

    def _run_mean_special_values_test(self, tensor):
        """
        CPU-only structural check for mean with special IEEE 754 values.

        Verifies that NaN in any element poisons the mean to NaN, and that
        +inf in any element produces +inf or NaN in the mean output.

        Args:
            tensor: float16 tensor of shape (1,1,5120) or (1,14,5120)
                    containing at least one special value.
        """
        result = torch.mean(tensor.cpu(), dim=-1)

        if torch.isnan(tensor).any():
            assert torch.isnan(result).any(), (
                f"Expected NaN in mean output when input contains NaN, "
                f"got: {result}"
            )
        if torch.isinf(tensor).any():
            assert torch.isinf(result).any() or torch.isnan(result).any(), (
                f"Expected inf/nan in mean output when input contains inf, "
                f"got: {result}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestTranspose
# ─────────────────────────────────────────────────────────────────────────────

class TestTranspose(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.transpose across all Ministral-3-14B-Instruct-2512 shapes.

    Input shapes (dtype=float16):
        3-D : [1, 64, 1],  [1, 64, 14]
        4-D : [1,  1, 32, 128], [1,  1, 8, 128]
              [1, 32,  1, 128], [1, 14, 32, 128]
              [1, 14,  8, 128], [1, 32, 14, 128]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_transpose_shape_test       → @pytest.mark.torch_transpose_shape
        _run_transpose_values_test      → @pytest.mark.torch_transpose_values
        _run_transpose_neg_dims_test    → @pytest.mark.torch_transpose_neg_dims
        _run_transpose_contiguity_test  → @pytest.mark.torch_transpose_contiguity
        _run_transpose_contig_copy_test → @pytest.mark.torch_transpose_contig_copy
        _run_transpose_dtype_test       → @pytest.mark.torch_transpose_dtype
    """

    pytestmark = pytest.mark.torch_transpose

    torch.manual_seed(0)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════

        # 3-D [1, 64, 1] ──────────────────────────────────────────────────
        # (0,1): [1,64,1] → [64,1,1]
        ("test_torch_transpose_shape_pattern_000", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x1_d01_eager":    (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), False),
                "s_1x64x1_d01_compiled": (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), True),
            },
        },
        # (1,2): [1,64,1] → [1,1,64]
        ("test_torch_transpose_shape_pattern_001", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x1_d12_eager":    (1, 2,  make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), False),
                "s_1x64x1_d12_compiled": (1, 2, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), True),
            },
        },

        # 3-D [1, 64, 14] ─────────────────────────────────────────────────
        # (0,1): [1,64,14] → [64,1,14]
        ("test_torch_transpose_shape_pattern_002", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x14_d01_eager":    (0, 1,  make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), False),
                "s_1x64x14_d01_compiled": (0, 1, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), True),
            },
        },
        # (1,2): [1,64,14] → [1,14,64]
        ("test_torch_transpose_shape_pattern_003", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x14_d12_eager":    (1, 2, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), False),
                "s_1x64x14_d12_compiled": (1, 2, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), True),
            },
        },

        # 4-D [1, 1, 32, 128] ─────────────────────────────────────────────
        # (1,3): [1,1,32,128] → [1,128,32,1]
        ("test_torch_transpose_shape_pattern_004", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_d13_eager":    (1, 2, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "s_1x1x32x128_d13_compiled": (1, 2, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,1,32,128] → [1,1,128,32]
        ("test_torch_transpose_shape_pattern_005", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "s_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },

        # 4-D [1, 1, 8, 128] ──────────────────────────────────────────────
        # (1,3): [1,1,8,128] → [1,128,8,1]
        ("test_torch_transpose_shape_pattern_006", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x8x128_d13_eager":    (1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "s_1x1x8x128_d13_compiled": (1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,1,8,128] → [1,1,128,8]
        ("test_torch_transpose_shape_pattern_007", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "s_1x1x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },

        # 4-D [1, 32, 1, 128] ─────────────────────────────────────────────
        # (1,3): [1,32,1,128] → [1,128,1,32]
        ("test_torch_transpose_shape_pattern_008", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_d13_eager":    (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "s_1x32x1x128_d13_compiled": (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,32,1,128] → [1,32,128,1]
        ("test_torch_transpose_shape_pattern_009", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_d23_eager":    (2, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "s_1x32x1x128_d23_compiled": (2, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },

        # 4-D [1, 14, 32, 128] ────────────────────────────────────────────
        # (1,2): [1,14,32,128] → [1,32,14,128]
        ("test_torch_transpose_shape_pattern_010", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x14x32x128_d12_eager":    (1, 2, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), False),
                "s_1x14x32x128_d12_compiled": (1, 2, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,14,32,128] → [1,14,128,32]
        ("test_torch_transpose_shape_pattern_011", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x14x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), False),
                "s_1x14x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), True),
            },
        },

        # 4-D [1, 14, 8, 128] ─────────────────────────────────────────────
        # (1,2): [1,14,8,128] → [1,8,14,128]
        ("test_torch_transpose_shape_pattern_012", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x14x8x128_d12_eager":    (1, 2, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), False),
                "s_1x14x8x128_d12_compiled": (1, 2, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,14,8,128] → [1,14,128,8]
        ("test_torch_transpose_shape_pattern_013", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x14x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), False),
                "s_1x14x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), True),
            },
        },

        # 4-D [1, 32, 14, 128] ────────────────────────────────────────────
        # (1,2): [1,32,14,128] → [1,14,32,128]
        ("test_torch_transpose_shape_pattern_014", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x14x128_d12_eager":    (1, 2, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), False),
                "s_1x32x14x128_d12_compiled": (1, 2, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,32,14,128] → [1,32,128,14]
        ("test_torch_transpose_shape_pattern_015", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x14x128_d23_eager":    (2, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), False),
                "s_1x32x14x128_d23_compiled": (2, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════

        # [1,64,14]  (1,2): t[b,r,c] == result[b,c,r]
        ("test_torch_transpose_values_pattern_000", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x64x14_d12_eager":    (1, 2, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), False),
                "v_1x64x14_d12_compiled": (1, 2, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), True),
            },
        },
        # [1,1,32,128]  (2,3): t[b,h,s,d] == result[b,h,d,s]
        ("test_torch_transpose_values_pattern_001", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "v_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,1,8,128]  (2,3)
        ("test_torch_transpose_values_pattern_002", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x1x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "v_1x1x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,1,128]  (1,3): t[b,h,s,d] == result[b,d,s,h]
        ("test_torch_transpose_values_pattern_003", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x1x128_d13_eager":    (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "v_1x32x1x128_d13_compiled": (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1,14,32,128]  (2,3)
        ("test_torch_transpose_values_pattern_004", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x14x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), False),
                "v_1x14x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,14,8,128]  (1,3)
        ("test_torch_transpose_values_pattern_005", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x14x8x128_d13_eager":    (1, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), False),
                "v_1x14x8x128_d13_compiled": (1, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,14,128]  (1,2)
        ("test_torch_transpose_values_pattern_006", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x14x128_d12_eager":    (1, 2, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), False),
                "v_1x32x14x128_d12_compiled": (1, 2, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), True),
            },
        },
        # [1,32,14,128]  (2,3)
        ("test_torch_transpose_values_pattern_007", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x14x128_d23_eager":    (2, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), False),
                "v_1x32x14x128_d23_compiled": (2, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # NEGATIVE DIMENSION INDEXING
        # ══════════════════════════════════════════════════════════════════

        # [1,64,14]  (-2,-1) == (1,2)
        ("test_torch_transpose_neg_dims_pattern_000", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x64x14_m2m1_vs_12_eager":    (-2, -1, 1, 2, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), False),
                "neg_1x64x14_m2m1_vs_12_compiled": (-2, -1, 1, 2, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), True),
            },
        },
        # [1,1,32,128]  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_001", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x1x32x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "neg_1x1x32x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,1,8,128]  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_002", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x1x8x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "neg_1x1x8x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,1,128]  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_003", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x1x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "neg_1x32x1x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1,14,32,128]  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_004", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x14x32x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), False),
                "neg_1x14x32x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,14,8,128]  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_005", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x14x8x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), False),
                "neg_1x14x8x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,14,128]  (-4,-1) == (0,3)
        ("test_torch_transpose_neg_dims_pattern_006", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x14x128_m4m1_vs_03_eager":    (-4, -1, 0, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), False),
                "neg_1x32x14x128_m4m1_vs_03_compiled": (-4, -1, 0, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), True),
            },
        },
        # [1,32,14,128]  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_007", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x14x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), False),
                "neg_1x32x14x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION
        # ══════════════════════════════════════════════════════════════════

        # [1,64,1]  (0,1)
        ("test_torch_transpose_dtype_pattern_000", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x1_d01_eager":    (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), False),
                "dtype_1x64x1_d01_compiled": (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float16), True),
            },
        },
        # [1,1,32,128]  (2,3)
        ("test_torch_transpose_dtype_pattern_001", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "dtype_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,1,8,128]  (2,3)
        ("test_torch_transpose_dtype_pattern_002", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x1x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "dtype_1x1x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,1,128]  (1,3)
        ("test_torch_transpose_dtype_pattern_003", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x32x1x128_d13_eager":    (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "dtype_1x32x1x128_d13_compiled": (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1,64,14]  (1,2)
        ("test_torch_transpose_dtype_pattern_004", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x14_d12_eager":    (1, 2,  make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), False),
                "dtype_1x64x14_d12_compiled": (1, 2, make_strided_tensor((1, 64, 14), (896, 14, 1), torch.float16), True),
            },
        },
        # [1,14,32,128]  (2,3)
        ("test_torch_transpose_dtype_pattern_005", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x14x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), False),
                "dtype_1x14x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,14,8,128]  (1,3)
        ("test_torch_transpose_dtype_pattern_006", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x14x8x128_d13_eager":    (1, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), False),
                "dtype_1x14x8x128_d13_compiled": (1, 3, make_strided_tensor((1, 14, 8, 128), (14336, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,14,128]  (1,2)
        ("test_torch_transpose_dtype_pattern_007", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x32x14x128_d12_eager":    (1, 2, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), False),
                "dtype_1x32x14x128_d12_compiled": (1, 2, make_strided_tensor((1, 32, 14, 128), (57344, 1792, 128, 1), torch.float16), True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_transpose_shape_test(self, dim0, dim1, x, compiled):
        expected = list(x.shape)
        d0, d1 = dim0 % x.ndim, dim1 % x.ndim
        expected[d0], expected[d1] = expected[d1], expected[d0]

        def fn(t):
            out = torch.transpose(t, dim0, dim1).contiguous()
            assert list(out.shape) == expected, (
                f"Shape mismatch: expected {expected}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_transpose_values_test(self, dim0, dim1, x, compiled):
        compare_with_cpu(
            lambda t: torch.transpose(t, dim0, dim1).contiguous(),
            x,
            compiled=compiled,
        )

    def _run_transpose_neg_dims_test(self, neg0, neg1, pos0, pos1, x, compiled):
        def fn(t):
            neg_result = torch.transpose(t, neg0, neg1).contiguous()
            pos_result = torch.transpose(t, pos0, pos1).contiguous()
            torch.testing.assert_close(
                neg_result, pos_result,
                msg=f"transpose({neg0},{neg1}) differs from transpose({pos0},{pos1})",
            )
            return neg_result

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_transpose_dtype_test(self, dim0, dim1, x, compiled):
        def fn(t):
            result = torch.transpose(t, dim0, dim1).contiguous()
            assert result.dtype == t.dtype, (
                f"dtype changed after transpose({dim0},{dim1}): "
                f"expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestCos
# ─────────────────────────────────────────────────────────────────────────────

class TestCos(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.cos across Ministral-3-14B-Instruct-2512 shapes.

    Op specification:
        name  : torch.cos.2_spyre
        op    : torch.cos
        dtype : torch.float16

    Input shapes:
        [1, 14, 128]
        [1,  1, 128]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_cos_shape_test  → @pytest.mark.torch_cos_shape
        _run_cos_values_test → @pytest.mark.torch_cos_values
        _run_cos_dtype_test  → @pytest.mark.torch_cos_dtype
    """

    pytestmark = pytest.mark.torch_cos

    torch.manual_seed(0)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — cos is pointwise: output shape == input shape
        # ══════════════════════════════════════════════════════════════════

        # [1, 14, 128]
        ("test_torch_cos_shape_pattern_000", "_run_cos_shape_test"): {
            "param_sets": {
                "s_1x14x128_eager":    (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), False),
                "s_1x14x128_compiled": (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 128]
        ("test_torch_cos_shape_pattern_001", "_run_cos_shape_test"): {
            "param_sets": {
                "s_1x1x128_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), False),
                "s_1x1x128_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — cos(x) CPU vs Spyre element-wise
        # ══════════════════════════════════════════════════════════════════

        # [1, 14, 128]  rand input
        ("test_torch_cos_values_pattern_000", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x14x128_rand_eager":    (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), False),
                "v_1x14x128_rand_compiled": (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), True),
            },
        },
        # [1, 14, 128]  zeros input: cos(0) == 1.0 for all elements
        ("test_torch_cos_values_pattern_001", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x14x128_zeros_eager":    (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16, fill="zeros"), False),
                "v_1x14x128_zeros_compiled": (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16, fill="zeros"), True),
            },
        },
        # [1, 1, 128]  rand input
        ("test_torch_cos_values_pattern_002", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x1x128_rand_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), False),
                "v_1x1x128_rand_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 128]  zeros input
        ("test_torch_cos_values_pattern_003", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x1x128_zeros_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16, fill="zeros"), False),
                "v_1x1x128_zeros_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16, fill="zeros"), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — float16 in must give float16 out
        # ══════════════════════════════════════════════════════════════════

        # [1, 14, 128]
        ("test_torch_cos_dtype_pattern_000", "_run_cos_dtype_test"): {
            "param_sets": {
                "dtype_1x14x128_eager":    (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), False),
                "dtype_1x14x128_compiled": (make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 128]
        ("test_torch_cos_dtype_pattern_001", "_run_cos_dtype_test"): {
            "param_sets": {
                "dtype_1x1x128_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), False),
                "dtype_1x1x128_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16), True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_cos_shape_test(self, x, compiled):
        def fn(t):
            out = torch.cos(t)
            assert list(out.shape) == list(t.shape), (
                f"Shape mismatch: expected {list(t.shape)}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_cos_values_test(self, x, compiled):
        compare_with_cpu(torch.cos, x, compiled=compiled)

    def _run_cos_dtype_test(self, x, compiled):
        def fn(t):
            result = torch.cos(t)
            assert result.dtype == t.dtype, (
                f"dtype changed after cos: expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestContiguous
# ─────────────────────────────────────────────────────────────────────────────

class TestContiguous(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.Tensor.contiguous across Ministral-3-14B-Instruct-2512 shapes.

    Input shapes (dtype=float16):
        4-D : [1, 14, 32, 128], [1, 1, 32, 128]
        3-D : [1, 14, 4096],    [1, 1, 4096]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_contiguous_shape_test    → @pytest.mark.torch_contiguous_shape
        _run_contiguous_values_test   → @pytest.mark.torch_contiguous_values
        _run_contiguous_noncontig_test → @pytest.mark.torch_contiguous_noncontig
        _run_contiguous_dtype_test    → @pytest.mark.torch_contiguous_dtype
    """

    pytestmark = pytest.mark.torch_contiguous

    torch.manual_seed(0)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — .contiguous() must not change shape
        # ══════════════════════════════════════════════════════════════════

        # [1, 14, 32, 128]
        ("test_torch_contiguous_shape_pattern_000", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x14x32x128_eager":    (make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), False),
                "s_1x14x32x128_compiled": (make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), True),
            },
        },
        # [1, 14, 4096]
        ("test_torch_contiguous_shape_pattern_001", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x14x4096_eager":    (make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), False),
                "s_1x14x4096_compiled": (make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]
        ("test_torch_contiguous_shape_pattern_002", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_eager":    (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "s_1x1x32x128_compiled": (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096]
        ("test_torch_contiguous_shape_pattern_003", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), False),
                "s_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — values are preserved after .contiguous()
        # ══════════════════════════════════════════════════════════════════

        # [1, 14, 32, 128]  already-contiguous input
        ("test_torch_contiguous_values_pattern_000", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x14x32x128_eager":    (make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), False),
                "v_1x14x32x128_compiled": (make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), True),
            },
        },
        # [1, 14, 4096]  already-contiguous input
        ("test_torch_contiguous_values_pattern_001", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x14x4096_eager":    (make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), False),
                "v_1x14x4096_compiled": (make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]  already-contiguous input
        ("test_torch_contiguous_values_pattern_002", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x1x32x128_eager":    (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "v_1x1x32x128_compiled": (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096]  already-contiguous input
        ("test_torch_contiguous_values_pattern_003", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), False),
                "v_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # NON-CONTIGUOUS INPUT — transpose first to get a non-contig view,
        # then .contiguous() must produce is_contiguous()==True with same values.
        # Only dim pairs where BOTH swapped sizes > 1 guarantee non-contiguity.
        # ══════════════════════════════════════════════════════════════════


        # [1, 14, 32, 128]  (1,3) sizes 14↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_001", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x14x32x128_d13_eager":    (1, 3, make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), False),
                "nc_1x14x32x128_d13_compiled": (1, 3, make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), True),
            },
        },
        # [1, 14, 32, 128]  (2,3) sizes 32↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_002", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x14x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), False),
                "nc_1x14x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), True),
            },
        },
        # [1, 14, 4096]  (1,2) sizes 14↔4096 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_003", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x14x4096_d12_eager":    (1, 2, make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), False),
                "nc_1x14x4096_d12_compiled": (1, 2, make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]  (2,3) sizes 32↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_004", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "nc_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096] has no pair with both dims > 1 — copy test only
        ("test_torch_contiguous_noncontig_pattern_005", "_run_contiguous_values_test"): {
            "param_sets": {
                "nc_copy_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), False),
                "nc_copy_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — float16 in must give float16 out
        # ══════════════════════════════════════════════════════════════════

        # [1, 14, 32, 128]
        ("test_torch_contiguous_dtype_pattern_000", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x14x32x128_eager":    (make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), False),
                "dtype_1x14x32x128_compiled": (make_strided_tensor((1, 14, 32, 128), (57344, 128, 1792, 1), torch.float16), True),
            },
        },
        # [1, 14, 4096]
        ("test_torch_contiguous_dtype_pattern_001", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x14x4096_eager":    (make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), False),
                "dtype_1x14x4096_compiled": (make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]
        ("test_torch_contiguous_dtype_pattern_002", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x1x32x128_eager":    (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "dtype_1x1x32x128_compiled": (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096]
        ("test_torch_contiguous_dtype_pattern_003", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), False),
                "dtype_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16), True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_contiguous_shape_test(self, x, compiled):
        def fn(t):
            out = t.contiguous()
            assert list(out.shape) == list(t.shape), (
                f"Shape mismatch: expected {list(t.shape)}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_contiguous_values_test(self, x, compiled):
        compare_with_cpu(lambda t: t.contiguous(), x, compiled=compiled)

    def _run_contiguous_noncontig_test(self, dim0, dim1, x, compiled):
        # Verify the transposed view is non-contiguous on CPU before device dispatch
        raw = torch.transpose(x, dim0, dim1)
        assert not raw.is_contiguous(), (
            f"transpose({dim0},{dim1}) on shape {list(x.shape)} should be non-contiguous"
        )

        def fn(t):
            view = torch.transpose(t, dim0, dim1)
            out = view.contiguous()
            assert out.is_contiguous(), (
                f"contiguous() result should be contiguous for shape {list(t.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_contiguous_dtype_test(self, x, compiled):
        def fn(t):
            result = t.contiguous()
            assert result.dtype == t.dtype, (
                f"dtype changed after contiguous(): expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedding(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.nn.functional.embedding across Ministral-3-14B-Instruct-2512 shapes.

    Op specification:
        op           : torch.nn.functional.embedding
        weight shape : [131072, 5120]   (vocab_size × embed_dim)
        index shapes : [1, 14]  (prefill),  [1, 1]  (decode)
        index dtype  : torch.int64
        weight dtype : torch.float16
        output shape : [*index.shape, embed_dim]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_embedding_shape_test  → @pytest.mark.torch_embedding_shape
        _run_embedding_values_test → @pytest.mark.torch_embedding_values
        _run_embedding_dtype_test  → @pytest.mark.torch_embedding_dtype
    """

    pytestmark = pytest.mark.torch_embedding

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — output shape: [*index.shape, embed_dim]
        # ══════════════════════════════════════════════════════════════════

        # [1, 14] + [131072, 5120] → [1, 14, 5120]
        ("test_torch_embedding_shape_pattern_000", "_run_embedding_shape_test"): {
            "param_sets": {
                "s_1x14_eager":    (make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, False),
                "s_1x14_compiled": (make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, True),
            },
        },
        # [1, 1] + [131072, 5120] → [1, 1, 5120]
        ("test_torch_embedding_shape_pattern_001", "_run_embedding_shape_test"): {
            "param_sets": {
                "s_1x1_eager":    (make_strided_tensor((1,1),(1,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, False),
                "s_1x1_compiled": (make_strided_tensor((1,1),(1,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — embedding(indices, weight) CPU vs Spyre
        # ══════════════════════════════════════════════════════════════════

        # [1, 14]  random indices — general prefill lookup
        ("test_torch_embedding_values_pattern_000", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x14_rand_eager":    (make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, False),
                "v_1x14_rand_compiled": (make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, True),
            },
        },
        # [1, 14]  all-zero indices — exercises first row of weight table
        ("test_torch_embedding_values_pattern_001", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x14_zero_eager":    (make_strided_tensor((1,14),(14,1),torch.int64,fill="zeros",min_val=0,max_val=VOCAB_SIZE), _W, False),
                "v_1x14_zero_compiled": (make_strided_tensor((1,14),(14,1),torch.int64,fill="zeros",min_val=0,max_val=VOCAB_SIZE), _W, True),
            },
        },
        # [1, 1]  random index — single-token decode step
        ("test_torch_embedding_values_pattern_002", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x1_rand_eager":    (make_strided_tensor((1,1),(1,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, False),
                "v_1x1_rand_compiled": (make_strided_tensor((1,1),(1,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, True),
            },
        },
        # [1, 1]  last vocab index — boundary check
        ("test_torch_embedding_values_pattern_003", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x1_last_eager":    (torch.full((1, 1), VOCAB_SIZE - 1, dtype=I64), _W, False),
                "v_1x1_last_compiled": (torch.full((1, 1), VOCAB_SIZE - 1, dtype=I64), _W, True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — output dtype must match weight dtype (float16)
        # ══════════════════════════════════════════════════════════════════

        # [1, 14]
        ("test_torch_embedding_dtype_pattern_000", "_run_embedding_dtype_test"): {
            "param_sets": {
                "dtype_1x14_eager":    (make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, False),
                "dtype_1x14_compiled": (make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, True),
            },
        },
        # [1, 1]
        ("test_torch_embedding_dtype_pattern_001", "_run_embedding_dtype_test"): {
            "param_sets": {
                "dtype_1x1_eager":    (make_strided_tensor((1,1),(1,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, False),
                "dtype_1x1_compiled": (make_strided_tensor((1,1),(1,1),torch.int64,min_val=0,max_val=VOCAB_SIZE), _W, True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_embedding_shape_test(self, indices, weight, compiled):
        expected = list(indices.shape) + [weight.shape[1]]

        def fn(idx, w):
            out = F.embedding(idx, w)
            assert list(out.shape) == expected, (
                f"Shape mismatch: expected {expected}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, indices, weight, compiled=compiled)

    def _run_embedding_values_test(self, indices, weight, compiled):
        compare_with_cpu(
            lambda idx, w: F.embedding(idx, w),
            indices, weight,
            compiled=compiled,
        )

    def _run_embedding_dtype_test(self, indices, weight, compiled):
        def fn(idx, w):
            result = F.embedding(idx, w)
            assert result.dtype == w.dtype, (
                f"dtype changed after embedding: expected {w.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, indices, weight, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestGetitem
# ─────────────────────────────────────────────────────────────────────────────

class TestGetitem(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.Tensor.__getitem__ across Ministral-3-14B-Instruct-2512 shapes.

    Index shapes  (int64) : [64], [1, 14], [1, 1]
    Data  shapes  (float16): [1, 32, 14, 128], [1, 8, 14, 128], [1, 8, 2048, 128]
                             [1, 14, 5120],     [1, 32, 1, 128], [1, 8, 1, 128]
                             [1, 1, 5120]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_getitem_shape_test  → @pytest.mark.torch_getitem_shape
        _run_getitem_values_test → @pytest.mark.torch_getitem_values
        _run_getitem_dtype_test  → @pytest.mark.torch_getitem_dtype

    Param tuple layout
    ------------------
    _run_getitem_shape_test  : (x, idx, expected_shape, compiled)
    _run_getitem_values_test : (x, idx, compiled)
    _run_getitem_dtype_test  : (x, idx, compiled)

    idx is a plain Python int, slice, or tuple of slices — not a Tensor.
    compare_with_cpu passes it through unchanged (only Tensor args are moved
    to the target device).
    """

    pytestmark = pytest.mark.torch_getitem

    torch.manual_seed(0)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — output shape is correct after indexing
        # ══════════════════════════════════════════════════════════════════

        # [64] int64  →  [:32]  →  [32]
        ("test_torch_getitem_shape_pattern_000", "_run_getitem_shape_test"): {
            "param_sets": {           
                "s_64_sl32_eager":    ( make_strided_tensor((64,),(1,),torch.int64,min_val=0,max_val=1000), S(None, 32), [32], False),
                "s_64_sl32_compiled": ( make_strided_tensor((64,),(1,),torch.int64,min_val=0,max_val=1000), S(None, 32), [32], True),
            },
        },
        # [1, 14] int64  →  [0]  →  [14]
        ("test_torch_getitem_shape_pattern_001", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x14_idx0_eager":    ( make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=1000), 0, [14], False),
                "s_1x14_idx0_compiled": ( make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=1000), 0, [14], True),
            },
        },
        # [1, 32, 14, 128] float16  →  [:, :, :1, :]  →  [1, 32, 1, 128]  (decode slice)
        ("test_torch_getitem_shape_pattern_002", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x32x14x128_d2sl1_eager":    (make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), [1, 32, 1, 128], False),
                "s_1x32x14x128_d2sl1_compiled": (make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), [1, 32, 1, 128], True),
            },
        },
        # [1, 8, 14, 128] float16  →  [:, :, :1, :]  →  [1, 8, 1, 128]
        ("test_torch_getitem_shape_pattern_003", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x14x128_d2sl1_eager":    (make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), [1, 8, 1, 128], False),
                "s_1x8x14x128_d2sl1_compiled": (make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), [1, 8, 1, 128], True),
            },
        },
        # [1, 8, 2048, 128] float16  →  [:, :, :14, :]  →  [1, 8, 14, 128]  (KV-cache prefill)
        ("test_torch_getitem_shape_pattern_004", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x2048x128_sl14_eager":    (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 14), _), [1, 8, 14, 128], False),
                "s_1x8x2048x128_sl14_compiled": (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 14), _), [1, 8, 14, 128], True),
            },
        },
        # [1, 14, 5120] float16  →  [:, 0, :]  →  [1, 5120]  (first token hidden state)
        ("test_torch_getitem_shape_pattern_005", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x14x5120_d1idx0_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), (_, 0, _), [1, 5120], False),
                "s_1x14x5120_d1idx0_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), (_, 0, _), [1, 5120], True),
            },
        },
        # [1, 1] int64  →  [0]  →  [1]
        ("test_torch_getitem_shape_pattern_006", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x1_idx0_eager":    (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, [1], False),
                "s_1x1_idx0_compiled": (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, [1], True),
            },
        },
        # [1, 32, 1, 128] float16  →  [0]  →  [32, 1, 128]
        ("test_torch_getitem_shape_pattern_007", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_idx0_eager":    (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, [32, 1, 128], False),
                "s_1x32x1x128_idx0_compiled": (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, [32, 1, 128], True),
            },
        },
        # [1, 8, 1, 128] float16  →  [0]  →  [8, 1, 128]
        ("test_torch_getitem_shape_pattern_008", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x1x128_idx0_eager":    (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, [8, 1, 128], False),
                "s_1x8x1x128_idx0_compiled": (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, [8, 1, 128], True),
            },
        },
        # [1, 1, 5120] float16  →  [0]  →  [1, 5120]
        ("test_torch_getitem_shape_pattern_009", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x1x5120_idx0_eager":    (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 0, [1, 5120], False),
                "s_1x1x5120_idx0_compiled": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 0, [1, 5120], True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — indexed values must match CPU reference
        # ══════════════════════════════════════════════════════════════════

        # [64] int64  →  [:32]  first half
        ("test_torch_getitem_values_pattern_000", "_run_getitem_values_test"): {
            "param_sets": {
                "v_64_sl32_eager":    ( make_strided_tensor((64,),(1,),torch.int64,min_val=0,max_val=1000), S(None, 32), False),
                "v_64_sl32_compiled": (make_strided_tensor((64,),(1,),torch.int64,min_val=0,max_val=1000), S(None, 32), True),
            },
        },
        # [64] int64  →  [32:]  second half
        ("test_torch_getitem_values_pattern_001", "_run_getitem_values_test"): {
            "param_sets": {
                "v_64_sl32end_eager":    (make_strided_tensor((64,),(1,),torch.int64,min_val=0,max_val=1000), S(32, None), False),
                "v_64_sl32end_compiled": (make_strided_tensor((64,),(1,),torch.int64,min_val=0,max_val=1000), S(32, None), True),
            },
        },
        # [1, 14] int64  →  [0]
        ("test_torch_getitem_values_pattern_002", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x14_idx0_eager":    ( make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=1000), 0, False),
                "v_1x14_idx0_compiled": (make_strided_tensor((1,14),(14,1),torch.int64,min_val=0,max_val=1000), 0, True),
            },
        },
        # [1, 32, 14, 128] float16  →  [:, :, :1, :]
        ("test_torch_getitem_values_pattern_003", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x32x14x128_d2sl1_eager":    (make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), False),
                "v_1x32x14x128_d2sl1_compiled": (make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 2048, 128] float16  →  [:, :, :14, :]  (KV-cache prefill slice)
        ("test_torch_getitem_values_pattern_004", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x2048x128_sl14_eager":    (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 14), _), False),
                "v_1x8x2048x128_sl14_compiled": (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 14), _), True),
            },
        },
        # [1, 8, 2048, 128] float16  →  [:, :, :1, :]  (KV-cache decode slice)
        ("test_torch_getitem_values_pattern_005", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x2048x128_sl1_eager":    (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 1), _), False),
                "v_1x8x2048x128_sl1_compiled": (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 14, 5120] float16  →  [:, 0, :]  (first token)
        ("test_torch_getitem_values_pattern_006", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x14x5120_d1idx0_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), (_, 0, _), False),
                "v_1x14x5120_d1idx0_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), (_, 0, _), True),
            },
        },
        # [1, 1] int64  →  [0]
        ("test_torch_getitem_values_pattern_007", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x1_idx0_eager":    (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, False),
                "v_1x1_idx0_compiled": (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, True),
            },
        },
        # [1, 32, 1, 128] float16  →  [:, :, 0, :]  →  [1, 32, 128]
        ("test_torch_getitem_values_pattern_008", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x32x1x128_d2idx0_eager":    (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), (_, _, 0, _), False),
                "v_1x32x1x128_d2idx0_compiled": (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), (_, _, 0, _), True),
            },
        },
        # [1, 8, 1, 128] float16  →  [:, :, 0, :]  →  [1, 8, 128]
        ("test_torch_getitem_values_pattern_009", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x1x128_d2idx0_eager":    (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), (_, _, 0, _), False),
                "v_1x8x1x128_d2idx0_compiled": (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), (_, _, 0, _), True),
            },
        },
        # [1, 1, 5120] float16  →  [0]  →  [1, 5120]
        ("test_torch_getitem_values_pattern_010", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x1x5120_idx0_eager":    (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 0, False),
                "v_1x1x5120_idx0_compiled": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 0, True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — dtype must not change after indexing
        # ══════════════════════════════════════════════════════════════════

        # [64] int64  →  [:32]
        ("test_torch_getitem_dtype_pattern_000", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_64_eager":    (make_strided_tensor((64,), (1,), torch.int64, min_val=0, max_val=1000), S(None, 32), False),
                "dtype_64_compiled": (make_strided_tensor((64,), (1,), torch.int64, min_val=0, max_val=1000), S(None, 32), True),
            },
        },
        # [1, 14] int64  →  [0]
        ("test_torch_getitem_dtype_pattern_001", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x14_eager":    (make_strided_tensor((1, 14), (14, 1), torch.int64, min_val=0, max_val=1000), 0, False),
                "dtype_1x14_compiled": (make_strided_tensor((1, 14), (14, 1), torch.int64, min_val=0, max_val=1000), 0, True),
            },
        },
        # [1, 32, 14, 128] float16  →  [:, :, :1, :]
        ("test_torch_getitem_dtype_pattern_002", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x32x14x128_eager":    (make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), False),
                "dtype_1x32x14x128_compiled": (make_strided_tensor((1, 32, 14, 128), (57344, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 14, 128] float16  →  [:, :, :1, :]
        ("test_torch_getitem_dtype_pattern_003", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x14x128_eager":    (make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), False),
                "dtype_1x8x14x128_compiled": (make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 2048, 128] float16  →  [:, :, :14, :]
        ("test_torch_getitem_dtype_pattern_004", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x2048x128_eager":    (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 14), _), False),
                "dtype_1x8x2048x128_compiled": (make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1), torch.float16), (_, _, S(None, 14), _), True),
            },
        },
        # [1, 14, 5120] float16  →  [:, 0, :]
        ("test_torch_getitem_dtype_pattern_005", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x14x5120_eager":    (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), (_, 0, _), False),
                "dtype_1x14x5120_compiled": (make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16), (_, 0, _), True),
            },
        },
        # [1, 1] int64  →  [0]
        ("test_torch_getitem_dtype_pattern_006", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x1_eager":    (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, False),
                "dtype_1x1_compiled": (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, True),
            },
        },
        # [1, 32, 1, 128] float16  →  [0]
        ("test_torch_getitem_dtype_pattern_007", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x32x1x128_eager":    (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, False),
                "dtype_1x32x1x128_compiled": (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, True),
            },
        },
        # [1, 8, 1, 128] float16  →  [0]
        ("test_torch_getitem_dtype_pattern_008", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x1x128_eager":    (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, False),
                "dtype_1x8x1x128_compiled": (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, True),
            },
        },
        # [1, 1, 5120] float16  →  [0]
        ("test_torch_getitem_dtype_pattern_009", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x1x5120_eager":    (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 0, False),
                "dtype_1x1x5120_compiled": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16), 0, True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_getitem_shape_test(self, x, idx, expected_shape, compiled):
        def fn(t):
            out = t[idx]
            assert list(out.shape) == expected_shape, (
                f"Shape mismatch: expected {expected_shape}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_getitem_values_test(self, x, idx, compiled):
        compare_with_cpu(lambda t: t[idx], x, compiled=compiled)

    def _run_getitem_dtype_test(self, x, idx, compiled):
        def fn(t):
            result = t[idx]
            assert result.dtype == t.dtype, (
                f"dtype changed after getitem: expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ═══════════════════════════════════════════════════════════════════════════
# TestSin  (unary element-wise)
# ═══════════════════════════════════════════════════════════════════════════

class TestSin(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.sin patterns observed in Ministral-3-14B-Instruct-2512.

    ``torch.sin`` is a unary element-wise op — it takes an input tensor and
    returns a new tensor with the sine of each element.

    Shapes from Ministral rotary embedding paths:
      [1, 14, 128]  — prefill
      [1, 1, 128]   — single-token decode
    """

    pytestmark = pytest.mark.torch_sin

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  prefill shape [1, 14, 128], float16
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_000", "_run_sin_test"): {
            "param_sets": {
                "1x14x128_fp16_eager": (
                    make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16),
                    False,
                ),
                "1x14x128_fp16_compiled": (
                    make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  decode shape [1, 1, 128], float16
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_001", "_run_sin_test"): {
            "param_sets": {
                "1x1x128_fp16_eager": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16),
                    False,
                ),
                "1x1x128_fp16_compiled": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  out= parameter, prefill shape [1, 14, 128]
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_002", "_run_sin_out_test"): {
            "param_sets": {
                "1x14x128_out_eager": (
                    make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16),
                    False,
                ),
                "1x14x128_out_compiled": (
                    make_strided_tensor((1, 14, 128), (1792, 128, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  out= parameter, decode shape [1, 1, 128]
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_003", "_run_sin_out_test"): {
            "param_sets": {
                "1x1x128_out_eager": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16),
                    False,
                ),
                "1x1x128_out_compiled": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float16),
                    True,
                ),
            },
        },
    }

    def _run_sin_test(self, input_tensor, compiled):
        """torch.sin(tensor) — unary element-wise sine."""
        compare_with_cpu(torch.sin, input_tensor, compiled=compiled)

    def _run_sin_out_test(self, input_tensor, compiled):
        """torch.sin with out= parameter — identity + numerical check."""
        shape = input_tensor.shape
        dtype = input_tensor.dtype

        def sin_out_fn(x):
            out = torch.empty(shape, dtype=dtype, device=x.device)
            result = torch.sin(x, out=out)
            assert result is out, f"{x.device}: out= did not return the same tensor"
            return result

        compare_with_cpu(sin_out_fn, input_tensor, compiled=compiled)


# ═══════════════════════════════════════════════════════════════════════════
# TestTruediv  (binary scalar division via torch.div rounding_mode=None)
# ═══════════════════════════════════════════════════════════════════════════

class TestTruediv(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.truediv (true division) patterns observed in
    Ministral-3-14B-Instruct-2512.

    In Ministral, truediv appears in the rotary embedding scaling formula:
      scaling = 1 + beta * log(1 + floor(position_ids / max_position_embeddings))

    ``torch.truediv`` was removed in recent PyTorch versions; we use the
    equivalent ``torch.div(x, divisor, rounding_mode=None)``.

    Shapes from Ministral rotary embedding scaling path:
      [14]  — prefill (position_ids for 14 tokens)
      [1]   — single-token decode
    """

    pytestmark = pytest.mark.torch_truediv

    torch.manual_seed(0)

    PARAMS = {
        # ==================================================================
        # Basic scalar division
        # ==================================================================

        # ------------------------------------------------------------------
        # pattern_000  prefill: [14] int64 / 16384
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_000", "_run_truediv_scalar_test"): {
            "param_sets": {
                "prefill_14_eager": (
                    make_strided_tensor((14,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    False,
                ),
                "prefill_14_compiled": (
                    make_strided_tensor((14,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  decode: [1] int64 / 16384
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_001", "_run_truediv_scalar_test"): {
            "param_sets": {
                "decode_1_eager": (
                    make_strided_tensor((1,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    False,
                ),
                "decode_1_compiled": (
                    make_strided_tensor((1,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    True,
                ),
            },
        },

        # ==================================================================
        # Exact value checks
        # ==================================================================

        # ------------------------------------------------------------------
        # pattern_002  exact: 16384 / 16384 = 1.0
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_002", "_run_truediv_scalar_test"): {
            "param_sets": {
                "exact_one_eager": (
                    torch.tensor([16384], dtype=torch.int64),
                    16384,
                    False,
                ),
                "exact_one_compiled": (
                    torch.tensor([16384], dtype=torch.int64),
                    16384,
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  zero: 0 / 16384 = 0.0
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_003", "_run_truediv_scalar_test"): {
            "param_sets": {
                "zero_input_eager": (
                    torch.tensor([0], dtype=torch.int64),
                    16384,
                    False,
                ),
                "zero_input_compiled": (
                    torch.tensor([0], dtype=torch.int64),
                    16384,
                    True,
                ),
            },
        },

        # ==================================================================
        # Correctness: truediv must NOT floor
        # ==================================================================

        # ------------------------------------------------------------------
        # pattern_004  7 / 2 = 3.5 (not 3)
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_004", "_run_truediv_no_floor_test"): {
            "param_sets": {
                "no_floor_eager": (
                    torch.tensor([7], dtype=torch.int64),
                    2,
                    False,
                ),
                "no_floor_compiled": (
                    torch.tensor([7], dtype=torch.int64),
                    2,
                    True,
                ),
            },
        },

        # ==================================================================
        # Dtype check: int64 input must produce float output
        # ==================================================================

        # ------------------------------------------------------------------
        # pattern_005  output dtype must be float, not int64
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_005", "_run_truediv_dtype_test"): {
            "param_sets": {
                "dtype_check_eager": (
                    make_strided_tensor((14,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    False,
                ),
                "dtype_check_compiled": (
                    make_strided_tensor((14,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    True,
                ),
            },
        },

        # ==================================================================
        # out= parameter
        # ==================================================================

        # ------------------------------------------------------------------
        # pattern_006  out= prefill: [14] int64 / 16384
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_006", "_run_truediv_out_test"): {
            "param_sets": {
                "prefill_out_eager": (
                    make_strided_tensor((14,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    False,
                ),
                "prefill_out_compiled": (
                    make_strided_tensor((14,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_007  out= decode: [1] int64 / 16384
        # ------------------------------------------------------------------
        ("test_torch_truediv_pattern_007", "_run_truediv_out_test"): {
            "param_sets": {
                "decode_out_eager": (
                    make_strided_tensor((1,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    False,
                ),
                "decode_out_compiled": (
                    make_strided_tensor((1,), (1,), torch.int64, min_val=0, max_val=1000),
                    16384,
                    True,
                ),
            },
        },
    }

    def _run_truediv_scalar_test(self, input_tensor, divisor, compiled):
        """torch.div(tensor, scalar, rounding_mode=None) — true division."""
        compare_with_cpu(
            lambda x: torch.true_divide(x, divisor),
            input_tensor,
            compiled=compiled,
        )

    def _run_truediv_no_floor_test(self, input_tensor, divisor, compiled):
        """Correctness: 7 / 2 must be 3.5, not 3."""
        expected = torch.tensor([3.5])

        def truediv_no_floor_fn(x):
            result = torch.true_divide(x, divisor)
            if x.device.type == "cpu":
                torch.testing.assert_close(
                    result, expected,
                    msg="truediv floored the result like div — expected 3.5, not 3",
                )
            return result

        compare_with_cpu(truediv_no_floor_fn, input_tensor, compiled=compiled)

    def _run_truediv_dtype_test(self, input_tensor, divisor, compiled):
        """Dtype check: int64 / int scalar must produce float output."""
        def truediv_dtype_fn(x):
            result = torch.true_divide(x, divisor)
            assert result.dtype.is_floating_point, (
                f"truediv output dtype should be float, got {result.dtype}"
            )
            return result

        compare_with_cpu(truediv_dtype_fn, input_tensor, compiled=compiled)

    def _run_truediv_out_test(self, input_tensor, divisor, compiled):
        """torch.div with out= parameter — identity + numerical check."""
        out_shape = input_tensor.shape

        def truediv_out_fn(x):
            out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
            result = torch.true_divide(x, divisor, out=out)
            assert result is out, f"{x.device}: out= did not return the same tensor"
            return result

        compare_with_cpu(truediv_out_fn, input_tensor, compiled=compiled)


# ═══════════════════════════════════════════════════════════════════════════
# Tensor.index_copy_  (in-place scatter)
# ═══════════════════════════════════════════════════════════════════════════

class TestTensorIndexCopy(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for Tensor.index_copy_ patterns observed in Ministral-3-14B-Instruct-2512.

    ``Tensor.index_copy_(dim, index, tensor)`` is an in-place op that
    copies rows/slices from ``tensor`` into ``self`` at positions given
    by ``index`` along dimension ``dim``.

    In Ministral this is used to write key/value states into the KV-cache:
      self.keys.index_copy_(2, cache_position, key_states)

    Shapes:
      self (cache):  [1, 8, 2048, 128]  float16
      dim:           2
      Prefill: cache_position [14] int64, key_states [1, 8, 14, 128] float16
      Decode:  cache_position [1]  int64, key_states [1, 8, 1, 128]  float16
    """

    pytestmark = pytest.mark.torch_index_copy

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  prefill: 14 tokens written into cache at positions 0-13
        # ------------------------------------------------------------------
        
        ("test_torch_index_copy_pattern_000", "_run_index_copy_test"): {
            "param_sets": {
                "prefill_14_eager": (
                    # torch.randn(1, 8, 2048, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1),torch.float16),
                    2,
                    # torch.arange(14, dtype=torch.int64),
                    make_strided_tensor((14,), (1,), torch.int64, fill="arange"),
                    # torch.randn(1, 8, 14, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16),
                    False,
                ),
                "prefill_14_compiled": (
                    # torch.randn(1, 8, 2048, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1),torch.float16),
                    2,
                    # torch.arange(14, dtype=torch.int64),
                    make_strided_tensor((14,), (1,), torch.int64, fill="arange"),
                    # torch.randn(1, 8, 14, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 14, 128), (14336, 128, 1024, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  decode: 1 token written into cache at position 14
        # ------------------------------------------------------------------
        ("test_torch_index_copy_pattern_001", "_run_index_copy_test"): {
            "param_sets": {
                "decode_1_eager": (
                    # torch.randn(1, 8, 2048, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1),torch.float16),
                    2,
                    # torch.arange(1, dtype=torch.int64),
                    make_strided_tensor((1,), (1,), torch.int64, fill="arange"),
                    # torch.randn(1, 8, 1, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 128, 1), torch.float16),
                    False,
                ),
                "decode_1_compiled": (
                    # torch.randn(1, 8, 2048, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 2048, 128), (2097152, 262144, 128, 1),torch.float16),
                    2,
                    # torch.arange(1, dtype=torch.int64),
                    make_strided_tensor((1,), (1,), torch.int64, fill="arange"),
                    # torch.randn(1, 8, 1, 128, dtype=torch.float16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 128, 1), torch.float16),
                    True,
                ),
            },
        },
    }

    def _run_index_copy_test(self, cache, dim, index, source, compiled):
        """
        Tensor.index_copy_(dim, index, tensor) — in-place scatter into cache.

        Clones the cache before the in-place op so the original is not
        mutated.  Asserts that the return value *is* the same tensor as
        the clone (in-place contract).
        """
        def index_copy_fn(c, idx, src):
            c = c.clone()
            result = c.index_copy_(dim, idx, src)
            assert result.data_ptr() == c.data_ptr(), (
                "index_copy_: return value is not the same tensor as self"
            )
            return result

        compare_with_cpu(
            index_copy_fn, cache, index, source, compiled=compiled,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestLinear (matrix multiply + optional bias)
# ═══════════════════════════════════════════════════════════════════════════

class TestLinear(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.nn.functional.linear patterns observed in
    Ministral-3-14B-Instruct-2512.

    ``F.linear(input, weight, bias=None)`` applies y = x A^T + b.
      - input:  (*, in_features)
      - weight: (out_features, in_features)
      - bias:   (out_features,) or None
      - output: (*, out_features)

    Shapes are sourced from the Ministral model's projection layers.
    Patterns 000-010: without bias
    Patterns 011-021: with bias
    """

    pytestmark = pytest.mark.torch_linear

    torch.manual_seed(0)

    PARAMS = {
        # ==================================================================
        # Without bias — patterns 000 through 010
        # ==================================================================

        # ------------------------------------------------------------------
        # pattern_000  prefill: [1,14,5120] x [4096,5120]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_000", "_run_linear_test"): {
            "param_sets": {
                "1x14x5120_4096x5120_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    False,
                ),
                "1x14x5120_4096x5120_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  prefill: [1,14,5120] x [1024,5120]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_001", "_run_linear_test"): {
            "param_sets": {
                "1x14x5120_1024x5120_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    False,
                ),
                "1x14x5120_1024x5120_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  prefill: [1,14,4096] x [5120,4096]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_002", "_run_linear_test"): {
            "param_sets": {
                "1x14x4096_5120x4096_eager": (
                    make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    False,
                ),
                "1x14x4096_5120x4096_compiled": (
                    make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  prefill: [1,14,5120] x [16384,5120]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_003", "_run_linear_test"): {
            "param_sets": {
                "1x14x5120_16384x5120_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    False,
                ),
                "1x14x5120_16384x5120_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_004  prefill: [1,14,16384] x [5120,16384]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_004", "_run_linear_test"): {
            "param_sets": {
                "1x14x16384_5120x16384_eager": (
                    make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    False,
                ),
                "1x14x16384_5120x16384_compiled": (
                     make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_005  decode: [1,1,5120] x [131072,5120]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_005", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_131072x5120_eager": (
                    make_strided_tensor((1, 1, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((131072, 5120), (5120, 1), torch.float16),
                    False,
                ),
                "1x1x5120_131072x5120_compiled": (
                    make_strided_tensor((1, 1, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((131072, 5120), (5120, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_006  decode: [1,1,5120] x [4096,5120]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_006", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_4096x5120_eager": (
                     make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    False,
                ),
                "1x1x5120_4096x5120_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_007  decode: [1,1,5120] x [1024,5120]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_007", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_1024x5120_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    False,
                ),
                "1x1x5120_1024x5120_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_008  decode: [1,1,4096] x [5120,4096]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_008", "_run_linear_test"): {
            "param_sets": {
                "1x1x4096_5120x4096_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    False,
                ),
                "1x1x4096_5120x4096_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_009  decode: [1,1,5120] x [16384,5120]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_009", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_16384x5120_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    False,
                ),
                "1x1x5120_16384x5120_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_010  decode: [1,1,16384] x [5120,16384]
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_010", "_run_linear_test"): {
            "param_sets": {
                "1x1x16384_5120x16384_eager": (
                    make_strided_tensor((1, 1, 16384), (16384, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    False,
                ),
                "1x1x16384_5120x16384_compiled": (
                    make_strided_tensor((1, 1, 16384), (16384, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    True,
                ),
            },
        },

        # ==================================================================
        # With bias — patterns 011 through 021
        # ==================================================================

        # ------------------------------------------------------------------
        # pattern_011  prefill: [1,14,5120] x [4096,5120] + bias(4096)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_011", "_run_linear_bias_test"): {
            "param_sets": {
                "1x14x5120_4096x5120_bias_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((4096,), (1,), torch.float16),
                    False,
                ),
                "1x14x5120_4096x5120_bias_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((4096,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_012  prefill: [1,14,5120] x [1024,5120] + bias(1024)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_012", "_run_linear_bias_test"): {
            "param_sets": {
                "1x14x5120_1024x5120_bias_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((1024,), (1,), torch.float16),
                    False,
                ),
                "1x14x5120_1024x5120_bias_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((1024,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_013  prefill: [1,14,4096] x [5120,4096] + bias(5120)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_013", "_run_linear_bias_test"): {
            "param_sets": {
                "1x14x4096_5120x4096_bias_eager": (
                    make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    False,
                ),
                "1x14x4096_5120x4096_bias_compiled": (
                    make_strided_tensor((1, 14, 4096), (57344, 4096, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_014  prefill: [1,14,5120] x [16384,5120] + bias(16384)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_014", "_run_linear_bias_test"): {
            "param_sets": {
                "1x14x5120_16384x5120_bias_eager": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((16384,), (1,), torch.float16),
                    False,
                ),
                "1x14x5120_16384x5120_bias_compiled": (
                    make_strided_tensor((1, 14, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((16384,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_015  prefill: [1,14,16384] x [5120,16384] + bias(5120)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_015", "_run_linear_bias_test"): {
            "param_sets": {
                "1x14x16384_5120x16384_bias_eager": (
                    make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    False,
                ),
                "1x14x16384_5120x16384_bias_compiled": (
                    make_strided_tensor((1, 14, 16384), (229376, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_016  decode: [1,1,5120] x [131072,5120] + bias(131072)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_016", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_131072x5120_bias_eager": (
                    make_strided_tensor((1, 1, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((131072, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((131072,), (1,), torch.float16),
                    False,
                ),
                "1x1x5120_131072x5120_bias_compiled": (
                    make_strided_tensor((1, 1, 5120), (71680, 5120, 1), torch.float16),
                    make_strided_tensor((131072, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((131072,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_017  decode: [1,1,5120] x [4096,5120] + bias(4096)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_017", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_4096x5120_bias_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((4096,), (1,), torch.float16),
                    False,
                ),
                "1x1x5120_4096x5120_bias_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((4096, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((4096,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_018  decode: [1,1,5120] x [1024,5120] + bias(1024)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_018", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_1024x5120_bias_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((1024,), (1,), torch.float16),
                    False,
                ),
                "1x1x5120_1024x5120_bias_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((1024, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((1024,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_019  decode: [1,1,4096] x [5120,4096] + bias(5120)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_019", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x4096_5120x4096_bias_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    False,
                ),
                "1x1x4096_5120x4096_bias_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 128, 1), torch.float16),
                    make_strided_tensor((5120, 4096), (4096, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_020  decode: [1,1,5120] x [16384,5120] + bias(16384)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_020", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_16384x5120_bias_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((16384,), (1,), torch.float16),
                    False,
                ),
                "1x1x5120_16384x5120_bias_compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), torch.float16),
                    make_strided_tensor((16384, 5120), (5120, 1), torch.float16),
                    make_strided_tensor((16384,), (1,), torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_021  decode: [1,1,16384] x [5120,16384] + bias(5120)
        # ------------------------------------------------------------------
        ("test_torch_linear_pattern_021", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x16384_5120x16384_bias_eager": (
                    make_strided_tensor((1, 1, 16384), (16384, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    False,
                ),
                "1x1x16384_5120x16384_bias_compiled": (
                    make_strided_tensor((1, 1, 16384), (16384, 16384, 1), torch.float16),
                    make_strided_tensor((5120, 16384), (16384, 1), torch.float16),
                    make_strided_tensor((5120,), (1,), torch.float16),
                    True,
                ),
            },
        },
    }

    def _run_linear_test(self, input_tensor, weight, compiled):
        """F.linear(input, weight) — without bias."""
        compare_with_cpu(
            lambda x, w: F.linear(x, w),
            input_tensor, weight,
            compiled=compiled,
            atol=2.0 ,rtol=0.05,
        )

    def _run_linear_bias_test(self, input_tensor, weight, bias, compiled):
        """F.linear(input, weight, bias) — with bias."""
        compare_with_cpu(
            lambda x, w, b: F.linear(x, w, b),
            input_tensor, weight, bias,
            compiled=compiled,
            atol =2.0 , rtol=5e-2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestFloat
# ─────────────────────────────────────────────────────────────────────────────

class TestFloat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for Tensor.float patterns observed in Ministral-3-14B-Instruct-2512.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_float_test``
    into one concrete test method per (pattern, param_set) combination.
    The metaclass also stamps each generated method with @pytest.mark.torch_float
    so the entire op can be selected with  pytest -m torch_float.

    ``Tensor.float()`` is equivalent to ``self.to(torch.float32)`` — it
    converts the input tensor to float32 dtype.  Input tensors are
    pre-constructed at param_set level with appropriate shapes and dtypes.

    Shapes are sourced from the Ministral model:
      [1, 64, 1]   — float16, attention mask
      [1, 1, 14]   — int64, position indices
      [1, 1, 14]   — float16, position embeddings
      [1, 1, 1]    — int64, scalar position
      [1, 1, 1]    — float16, scalar value
    """

    pytestmark = pytest.mark.torch_float

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  [1, 64, 1] float16 -> float32
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_000", "_run_float_test"): {
            "param_sets": {
                "1x64x1_fp16_eager": (
                    
                    make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    False,
                ),
                "1x64x1_fp16_compiled": (
                    make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  [1, 1, 14] int64 -> float32
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_001", "_run_float_test"): {
            "param_sets": {
                "1x1x14_int64_eager": (
                    make_strided_tensor((1,1,14),(14,14,1),torch.int64,min_val=0,max_val=1000),
                    False,
                ),
                "1x1x14_int64_compiled": (
                    make_strided_tensor((1,1,14),(14,14,1),torch.int64,min_val=0,max_val=1000),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  [1, 1, 14] float16 -> float32
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_002", "_run_float_test"): {
            "param_sets": {
                "1x1x14_fp16_eager": (
                    make_strided_tensor((1,1,14),(14,14,1),torch.float16),
                    False,
                ),
                "1x1x14_fp16_compiled": (
                    make_strided_tensor((1,1,14),(14,14,1),torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  [1, 1, 1] int64 -> float32
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_003", "_run_float_test"): {
            "param_sets": {
                "1x1x1_int64_eager": (
                    make_strided_tensor((1,1,1),(1,1,1),torch.int64,min_val=0,max_val=1000),
                    False,
                ),
                "1x1x1_int64_compiled": (
                    make_strided_tensor((1,1,1),(1,1,1),torch.int64,min_val=0,max_val=1000),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_004  [1, 1, 1] float16 -> float32
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_004", "_run_float_test"): {
            "param_sets": {
                "1x1x1_fp16_eager": (
                    make_strided_tensor((1,1,1),(1,1,1),torch.float16),
                    False,
                ),
                "1x1x1_fp16_compiled": (
                    make_strided_tensor((1,1,1),(1,1,1),torch.float16),
                    True,
                ),
            },
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # Never called directly; the metaclass replaces them with concrete
    # tests, each stamped with @pytest.mark.torch_float.
    # ------------------------------------------------------------------

    def _run_float_test(self, input_tensor, compiled):
        """
        Tensor.float() — converts tensor to float32 dtype.

        Wraps the method call in a function so compare_with_cpu can
        handle device placement.
        """
        def float_fn(x):
            return x.float()

        compare_with_cpu(float_fn, input_tensor, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestTo
# ─────────────────────────────────────────────────────────────────────────────
class TestTo(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for Tensor.to patterns observed in Ministral-3-14B-Instruct-2512.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_to_test``
    into one concrete test method per (pattern, param_set) combination.
    The metaclass also stamps each generated method with @pytest.mark.torch_to
    so the entire op can be selected with  pytest -m torch_to.

    ``Tensor.to(dtype)`` returns a tensor converted to the specified dtype.
    If the tensor already has the target dtype, self is returned.

    Each param_set contains:
      - input_tensor: tensor to convert
      - target_dtype: the desired dtype passed to .to()
      - compiled:     whether to run under torch.compile

    Shapes are sourced from the Ministral model:
      [1, 64, 1]   — attention mask conversion
      [1, 14, 128] — prefill rotary embedding conversion
      [1, 1, 128]  — decode rotary embedding conversion
    """

    pytestmark = pytest.mark.torch_to

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  [1, 64, 1] float16 -> float32
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_000", "_run_to_test"): {
            "param_sets": {
                "1x64x1_fp16_to_fp32_eager": (
                    make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    torch.float32,
                    False,
                ),
                "1x64x1_fp16_to_fp32_compiled": (
                    make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    torch.float32,
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  [1, 14, 128] float16 -> float32
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_001", "_run_to_test"): {
            "param_sets": {
                "1x14x128_fp16_to_fp32_eager": (
                    make_strided_tensor((1,14,128),(1792, 128, 1),torch.float16),
                    torch.float32,
                    False,
                ),
                "1x14x128_fp16_to_fp32_compiled": (
                    make_strided_tensor((1,14,128),(1792, 128, 1),torch.float16),
                    torch.float32,
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  [1, 1, 128] float16 -> float32
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_002", "_run_to_test"): {
            "param_sets": {
                "1x1x128_fp16_to_fp32_eager": (
                    make_strided_tensor((1,1,128),(128, 128, 1),torch.float16),
                    torch.float32,
                    False,
                ),
                "1x1x128_fp16_to_fp32_compiled": (
                    make_strided_tensor((1,1,128),(128, 128, 1),torch.float16),
                    torch.float32,
                    True,
                ),
            },
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # Never called directly; the metaclass replaces them with concrete
    # tests, each stamped with @pytest.mark.torch_to.
    # ------------------------------------------------------------------

    def _run_to_test(self, input_tensor, target_dtype, compiled):
        """
        Tensor.to(dtype) — converts tensor to the specified dtype.

        Wraps the method call in a function so compare_with_cpu can
        handle device placement.
        """
        def to_fn(x):
            return x.to(target_dtype)

        compare_with_cpu(to_fn, input_tensor, compiled=compiled)



# ─────────────────────────────────────────────────────────────────────────────
# TestView
# ─────────────────────────────────────────────────────────────────────────────
class TestView(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for Tensor.view patterns observed in Ministral-3-14B-Instruct-2512.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_view_test``
    into one concrete test method per (pattern, param_set) combination.
    The metaclass also stamps each generated method with @pytest.mark.torch_view
    so the entire op can be selected with  pytest -m torch_view.

    ``Tensor.view(*shape)`` returns a new tensor with the same data but a
    different shape.  The -1 dimension is inferred from the total element
    count and the other dimensions.

    Each param_set contains:
      - input_tensor: tensor to reshape
      - target_shape: the desired shape passed to .view()
      - compiled:     whether to run under torch.compile

    Shapes are sourced from the Ministral model:
      [1, 14, 4096]  -> (1, 14, -1, 128)  — prefill, multi-head reshape
      [1, 14, 1024]  -> (1, 14, -1, 128)  — prefill, grouped-query reshape
      [1, 1, 4096]   -> (1, 1, -1, 128)   — decode, multi-head reshape
      [1, 1, 1024]   -> (1, 1, -1, 128)   — decode, grouped-query reshape
    """

    pytestmark = pytest.mark.torch_view

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  [1, 14, 4096] -> (1, 14, -1, 128)  i.e. (1, 14, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_000", "_run_view_test"): {
            "param_sets": {
                "1x14x4096_to_1x14xNx128_eager": (
                    make_strided_tensor((1,14,4096),(57344, 4096, 1),torch.float16),
                    (1, 14, -1, 128),
                    False,
                ),
                "1x14x4096_to_1x14xNx128_compiled": (
                    make_strided_tensor((1,14,4096),(57344, 4096, 1),torch.float16),
                    (1, 14, -1, 128),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  [1, 14, 1024] -> (1, 14, -1, 128)  i.e. (1, 14, 8, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_001", "_run_view_test"): {
            "param_sets": {
                "1x14x1024_to_1x14xNx128_eager": (
                    make_strided_tensor((1,14,1024),(14336, 1024, 1),torch.float16),
                    (1, 14, -1, 128),
                    False,
                ),
                "1x14x1024_to_1x14xNx128_compiled": (
                   make_strided_tensor((1,14,1024),(14336, 1024, 1),torch.float16),
                    (1, 14, -1, 128),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  [1, 1, 4096] -> (1, 1, -1, 128)  i.e. (1, 1, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_002", "_run_view_test"): {
            "param_sets": {
                "1x1x4096_to_1x1xNx128_eager": (
                    make_strided_tensor((1,1,4096),(4096, 4096, 1),torch.float16),
                    (1, 1, -1, 128),
                    False,
                ),
                "1x1x4096_to_1x1xNx128_compiled": (
                    make_strided_tensor((1,1,4096),(4096, 4096, 1),torch.float16),
                    (1, 1, -1, 128),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  [1, 1, 1024] -> (1, 1, -1, 128)  i.e. (1, 1, 8, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_003", "_run_view_test"): {
            "param_sets": {
                "1x1x1024_to_1x1xNx128_eager": (
                    make_strided_tensor((1,1,1024),(1024, 1024, 1),torch.float16),
                    (1, 1, -1, 128),
                    False,
                ),
                "1x1x1024_to_1x1xNx128_compiled": (
                    make_strided_tensor((1,1,1024),(1024, 1024, 1),torch.float16),
                    (1, 1, -1, 128),
                    True,
                ),
            },
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # Never called directly; the metaclass replaces them with concrete
    # tests, each stamped with @pytest.mark.torch_view.
    # ------------------------------------------------------------------

    def _run_view_test(self, input_tensor, target_shape, compiled):
        """
        Tensor.view(*shape) — returns a new tensor with the same data
        but a different shape.

        Wraps the method call in a function so compare_with_cpu can
        handle device placement.
        """
        def view_fn(x):
            return x.view(*target_shape)

        compare_with_cpu(view_fn, input_tensor, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestExpand
# ─────────────────────────────────────────────────────────────────────────────
class TestExpand(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for Tensor.expand patterns observed in Ministral-3-14B-Instruct-2512.

    ``Tensor.expand`` is a view op — returns a new view with singleton
    dimensions expanded to a larger size.  No new memory is allocated.

    Shapes from Ministral:
      [1, 8, 1, 2048, 128]  — KV-cache broadcast expand
      [1, 64, 1]            — attention mask expand
    """

    pytestmark = pytest.mark.torch_expand

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  [1,8,1,2048,128] -> [1,8,4,2048,128]
        # ------------------------------------------------------------------
        ("test_torch_expand_pattern_000", "_run_expand_test"): {
            "param_sets": {
                "1x8x1x2048x128_to_1x8x4x2048x128_eager": (
                    
                    torch.randn(1, 8, 1, 2048, 128, dtype=torch.float16),
                    (1, 8, 4, 2048, 128),
                    False,
                ),
                "1x8x1x2048x128_to_1x8x4x2048x128_compiled": (
                    torch.randn(1, 8, 1, 2048, 128, dtype=torch.float16),
                    (1, 8, 4, 2048, 128),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  [1,8,1,2048,128] -> [1,8,4,2048,128] using -1
        # ------------------------------------------------------------------
        ("test_torch_expand_pattern_001", "_run_expand_test"): {
            "param_sets": {
                "1x8x1x2048x128_neg1_eager": (
                    make_strided_tensor((1, 8, 1, 2048, 128), (2097152, 262144, 262144, 128, 1), torch.float16),
                    (-1, -1, 4, -1, -1),
                    False,
                ),
                "1x8x1x2048x128_neg1_compiled": (
                    make_strided_tensor((1, 8, 1, 2048, 128), (2097152, 262144, 262144, 128, 1), torch.float16),
                    (-1, -1, 4, -1, -1),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  [1,64,1] -> [1,64,1]
        # ------------------------------------------------------------------
        ("test_torch_expand_pattern_002", "_run_expand_test"): {
            "param_sets": {
                "1x64x1_to_1x64x1_eager": (
                    make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    (1, -1, 1),
                    False,
                ),
                "1x64x1_to_1x64x1_compiled": (
                    make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    (1, -1, 1),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  [1,64,1] -> [1,64,1] using -1
        # ------------------------------------------------------------------
        ("test_torch_expand_pattern_003", "_run_expand_test"): {
            "param_sets": {
                "1x64x1_neg1_eager": (
                   make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    (-1, -1, -1),
                    False,
                ),
                "1x64x1_neg1_compiled": (
                   make_strided_tensor((1,64,1),(64,1,1),torch.float16),
                    (-1, -1, -1),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_004  shared memory + clone — [1,8,1,2048,128] -> [1,8,4,2048,128]
        # ------------------------------------------------------------------
        ("test_torch_expand_pattern_004", "_run_expand_clone_test"): {
            "param_sets": {
                "1x8x1x2048x128_clone_eager": (
                    
                    make_strided_tensor((1, 8, 1, 2048, 128), (2097152, 262144, 262144, 128, 1), torch.float16),
                    (1, 8, 4, 2048, 128),
                    False,
                ),
                "1x8x1x2048x128_clone_compiled": (
                    make_strided_tensor((1, 8, 1, 2048, 128), (2097152, 262144, 262144, 128, 1), torch.float16),
                    (1, 8, 4, 2048, 128),
                    True,
                ),
            },
        },
    }


    def _run_expand_test(self, input_tensor, target_size, compiled):
        def expand_fn(x):
            return x.expand(*target_size)

        compare_with_cpu(expand_fn, input_tensor.clone(), compiled=compiled)  # ← .clone() here


    def _run_expand_clone_test(self, input_tensor, target_size, compiled):
        def expand_clone_fn(x):
            expanded = x.expand(*target_size)
            assert any(s == 0 for s in expanded.stride()), (
                f"Expected stride 0 on expanded dim, got strides {expanded.stride()}"
            )
            cloned = expanded.clone()
            assert all(s != 0 for s in cloned.stride()), (
                f"Clone should have no stride-0 dims, got strides {cloned.stride()}"
            )
            return cloned

        compare_with_cpu(expand_clone_fn, input_tensor.clone(), compiled=compiled)  # ← .clone() here



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()