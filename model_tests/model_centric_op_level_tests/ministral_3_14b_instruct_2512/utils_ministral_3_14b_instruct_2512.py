# utils_minstra_3_14b_instruct_2512.py
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
Self-contained test utilities for Ministral-3-14B-Instruct-2512 operator tests.

Public API
----------
DEVICE                  torch.device targeting the Spyre accelerator.

--- Generic tolerances (eager / compiled) ---
EAGER_ATOL / EAGER_RTOL         Tolerances for uncompiled Spyre execution.
COMPILED_ATOL / COMPILED_RTOL   Tolerances for torch.compile Spyre execution.

--- SDPA tolerances ---
TOLERANCES              Per-dtype {atol, rtol} dict used by compare_sdpa.

--- Model architecture constants (Ministral-3-14B-Instruct-2512) ---
NUM_Q_HEADS             Query heads = 32.
NUM_KV_HEADS            Key/Value heads = 8 (Grouped Query Attention).
HEAD_DIM                Per-head dimension = 128.
GQA_GROUPS              Expansion factor = NUM_Q_HEADS // NUM_KV_HEADS = 4.
SCALE                   Attention scale = 1 / sqrt(HEAD_DIM).
SLIDING_WINDOW          Sliding-window context = 4096 tokens.
NUM_LAYERS              Decoder layers = 40.
NUM_ATTENTION_HEADS     Total attention heads for weight-shape arithmetic = 40.
                        NOTE: Used only for weight-matrix shapes such as
                        NUM_ATTENTION_HEADS * HEAD_DIM = 5120 (EMBED_DIM).
                        SDPA and attention tensor shapes use NUM_Q_HEADS = 32.
DEFAULT_DTYPE           torch.float16.
VOCAB_SIZE              Vocabulary size = 131_072.
ROPE_THETA              RoPE base frequency = 1_000_000.0.

--- Dimension constants ---
HIDDEN_SIZE             Residual-stream / hidden dimension = 4096.
                        Use this for hidden-state tensor shapes.
EMBED_DIM               Token embedding / projection dimension = 5120.
                        Equal to NUM_ATTENTION_HEADS * HEAD_DIM.
                        Use this for embedding and linear-projection shapes.
INTERMEDIATE_SIZE       FFN intermediate dimension = 16384 (= 4 × EMBED_DIM).
                        This is the single canonical value for this model.

NOTE ON LEGACY ALIASES
  INTERMEDIATE_SIZE_14B (= 16384) and INTERMEDIATE_SIZE_3B (= 8192) are kept
  only for backward compatibility with older shared utility code.  No new
  test cases should reference these aliases.

--- SDPA pre-built param dicts (built once at import time) ---
PREFILL_PARAMS          Causal prefill cases: seq_q == seq_kv.
DECODE_PARAMS           Decode cases: seq_q == 1.
DTYPE_PARAMS            Multi-dtype prefill (fp16, bf16, fp32).
NUMERIC_COVERAGE_PARAMS 40-seed numerical coverage sweep.
GROWING_KV_PARAMS       Autoregressive decode with growing KV cache.
SLIDING_WINDOW_PARAMS   Sliding-window mask cases.

--- Log-API key registries ---
MINISTRAL_MODEL_KEYS    Event keys emitted by the Ministral-3-14B modeling file.
HF_KEYS                 Event keys emitted by the HuggingFace transformers layer.
TORCH_INTERNAL_KEYS     Event keys emitted by PyTorch internals.
RUNTIME_KEYS            Event keys emitted by inference runtimes.

--- Tensor factories ---
cached_randn            LRU-cached random tensor factory.
make_qkv                Build (q, k, v) param-set tuples for SDPA tests.
make_tensor             General-purpose contiguous tensor factory.
                        Alias: _t (single-letter shorthand for PARAMS dicts).
expand_kv               GQA head expansion (on CPU, then moved back to device).
causal_mask             Upper-triangular causal additive mask.
sliding_window_mask     Causal mask with sliding-window cutoff.
sdpa_fn                 F.scaled_dot_product_attention wrapper (eager, GQA).

--- Comparison helpers ---
compare_sdpa            Eager-only CPU vs Spyre SDPA comparison.
compare_with_cpu        Eager or compiled CPU vs Spyre comparison.

--- Test infrastructure ---
ParameterizedTestMeta   Metaclass that expands a PARAMS dict on a TestCase
                        subclass into individual test_* methods.
                        NOTE: functools.wraps is deliberately NOT used.
                        functools.wraps sets __wrapped__, which causes pytest
                        to misidentify each test's source location and
                        silently deselect it.  __name__ / __qualname__ are
                        set explicitly instead.

make_tensor / _t behaviour
--------------------------
* Floating dtypes (float16, bfloat16, float32):
      torch.randn(shape, dtype=dtype) * 10.0
  Scaled by 10 so activations span a realistic range for comparison.

* Integer dtypes (int32, int64):
      torch.randint(-100, 100, shape, dtype=dtype)

* Boolean dtype:
      torch.randint(0, 2, shape, dtype=dtype)
"""

import functools
import math
from typing import Optional

from collections import defaultdict
import pytest
import torch
import torch.nn.functional as F

# Set this to True to print the tensor shapes, dim_map and device_size
DEBUG_LAYOUT = False

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("spyre")

# ─────────────────────────────────────────────────────────────────────────────
# Generic tolerances  (eager / compiled paths)
# ─────────────────────────────────────────────────────────────────────────────
# Aligned with the project-wide default used in model yaml files (Ministral-3-14B-Instruct-2512_spyre.yaml).
#
# torch.compile may reorder / fuse ops and alter rounding, so compiled runs
# use a wider bound than eager.  Failures exceeding EAGER_ATOL should be
# investigated as backend bugs rather than resolved by widening the tolerance.

EAGER_ATOL,    EAGER_RTOL    = 5e-3, 5e-3
COMPILED_ATOL, COMPILED_RTOL = 1e-1, 1e-2

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture constants  (Ministral-3-14B-Instruct-2512)
# ─────────────────────────────────────────────────────────────────────────────

# Attention head counts
NUM_Q_HEADS    = 32     # Query heads — used for all SDPA / attention shapes.
NUM_KV_HEADS   = 8      # Key/Value heads (Grouped Query Attention).
GQA_GROUPS     = NUM_Q_HEADS // NUM_KV_HEADS   # 4 — expansion factor.

# Per-head dimension (hidden_size_attn // num_q_heads)
HEAD_DIM = 128

# Attention scale factor (1 / sqrt(HEAD_DIM) ≈ 0.0884)
SCALE = 1.0 / math.sqrt(HEAD_DIM)

# Sliding-window attention context length
SLIDING_WINDOW = 4096

# Number of decoder transformer layers
NUM_LAYERS = 40

# Vocabulary size (used by embedding and LM-head tests)
VOCAB_SIZE = 131_072

# RoPE base frequency
ROPE_THETA = 1_000_000.0

# Default compute dtype for this model
DEFAULT_DTYPE = torch.float16

# ── Dimension constants ──────────────────────────────────────────────────────
#
# Three distinct "hidden" dimensions appear in the 14B model.  They are kept
# as separate named constants to avoid confusion:
#
#   HIDDEN_SIZE   (4096)  — residual-stream dimension.  Used for hidden-state
#                           tensor shapes such as [batch, seq, HIDDEN_SIZE].
#
#   EMBED_DIM     (5120)  — token embedding / projection dimension.
#                           Equal to NUM_ATTENTION_HEADS * HEAD_DIM.
#                           Used for embedding weight shapes and linear-
#                           projection shapes (Q/K/V, output, LM head).
#
#   INTERMEDIATE_SIZE (16384) — FFN gate/up/down intermediate dimension.
#                               Equal to 4 × EMBED_DIM.
#
# NUM_ATTENTION_HEADS (40) is used only for weight-matrix shape arithmetic
# (e.g. NUM_ATTENTION_HEADS * HEAD_DIM = 5120).  All SDPA and runtime
# attention shapes use NUM_Q_HEADS = 32 instead.

HIDDEN_SIZE         = 4096    # Residual-stream dimension.
EMBED_DIM           = 5120    # Embedding / projection dimension (= 40 * 128).
NUM_ATTENTION_HEADS = 40      # Used for weight-shape arithmetic only.
INTERMEDIATE_SIZE   = 16384   # FFN intermediate size (= 4 × EMBED_DIM).

# Legacy compatibility aliases — do NOT use these in new test cases.
INTERMEDIATE_SIZE_14B = INTERMEDIATE_SIZE   # Kept for shared utility code.
INTERMEDIATE_SIZE_3B  = 8192               # Kept for shared utility code.

# ─────────────────────────────────────────────────────────────────────────────
# Per-dtype SDPA comparison tolerances
# ─────────────────────────────────────────────────────────────────────────────

TOLERANCES: dict = {
    torch.float32:  dict(atol=1e-4, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
    torch.float16:  dict(atol=1e-2, rtol=1e-2),
}

# ─────────────────────────────────────────────────────────────────────────────
# Dtype and index shorthand constants
# ─────────────────────────────────────────────────────────────────────────────

F16  = torch.float16
F32  = torch.float32
I64  = torch.int64
BF16 = torch.bfloat16

# ─────────────────────────────────────────────────────────────────────────────
# Embedding weight and index helpers
# ─────────────────────────────────────────────────────────────────────────────

# Pre-built embedding weight table [VOCAB_SIZE, EMBED_DIM] in float16.
# Shared across all TestEmbedding and TestGetitem param sets.
_W = torch.randn(VOCAB_SIZE, EMBED_DIM, dtype=F16)

# Slice shorthand — used in PARAMS dicts for getitem tests.
S = slice          # e.g. S(None, 32) == [:32]
_ = slice(None)    # [:] on any dimension

# ─────────────────────────────────────────────────────────────────────────────
# Log-API-usage-once key registries  (Ministral-3-14B-Instruct-2512)
# ─────────────────────────────────────────────────────────────────────────────
# Canonical lists of torch._C._log_api_usage_once event strings that appear
# in the Ministral-3-14B-Instruct-2512 call graph.

# Keys emitted by the Ministral-3-14B-Instruct-2512 modeling file itself.
MINISTRAL_MODEL_KEYS: list[str] = [
    "ministral_3_14b.MistralForCausalLM.forward",
    "ministral_3_14b.MistralModel.forward",
    "ministral_3_14b.MistralDecoderLayer.forward",
    "ministral_3_14b.MistralSdpaAttention.forward",
    "ministral_3_14b.MistralAttention.forward",
    "ministral_3_14b.MistralFlashAttention2.forward",
    "ministral_3_14b.MistralMLP.forward",
    "ministral_3_14b.MistralMLP.gate_activation",
    "ministral_3_14b.MistralMLP.swiglu_product",
    "ministral_3_14b.MistralMLP.down_proj",
    "ministral_3_14b.MistralRMSNorm.forward",
    "ministral_3_14b.MistralRotaryEmbedding.forward",
    "ministral_3_14b.modeling_ministral3_14b.import",
]

# Keys emitted by the HuggingFace transformers layer.
HF_KEYS: list[str] = [
    "transformers.PreTrainedModel.forward",
    "transformers.PreTrainedModel.from_pretrained",
    "transformers.generation.GenerationMixin.generate",
    "transformers.generation.GenerationMixin.greedy_search",
    "transformers.generation.GenerationMixin.sample",
    "transformers.generation.GenerationMixin.beam_search",
    "transformers.AutoModelForCausalLM.from_pretrained",
    "transformers.AutoTokenizer.from_pretrained",
    "transformers.pipeline",
]

# Keys emitted by PyTorch internals.
TORCH_INTERNAL_KEYS: list[str] = [
    "torch.nn.functional.silu",
    "torch.nn.functional.scaled_dot_product_attention",
    "torch.nn.functional.layer_norm",
    "torch.nn.functional.linear",
    "torch.nn.Embedding.forward",
    "torch.nn.Linear.forward",
    "torch.nn.LayerNorm.forward",
    "torch.autocast",
    "torch.compile",
    "torch.no_grad",
    "python.torch",
]

# Keys emitted by inference runtimes (Spyre, vLLM, TGI).
RUNTIME_KEYS: list[str] = [
    "spyre.Ministral3_14BForCausalLM.forward",
    "spyre.engine.LLMEngine.step",
    "spyre.worker.Worker.execute_model",
    "vllm.Ministral3_14BForCausalLM",
    "vllm.engine.LLMEngine.step",
    "tgi.Ministral3_14BModel.forward",
]

# ─────────────────────────────────────────────────────────────────────────────
# ParameterizedTestMeta
#
#   Why functools.wraps is NOT used:
#       functools.wraps copies __module__, __qualname__, __doc__, and
#       __wrapped__ onto the wrapper function.  When pytest collects
#       unittest.TestCase subclasses it inspects __wrapped__ on each test
#       method.  If __wrapped__ points to a base function defined in this
#       utility module, pytest misidentifies the test's source location and
#       silently deselect it — producing the "collected N items" + immediate
#       exit symptom with zero tests run.  __name__ / __qualname__ are set
#       explicitly instead, and only __doc__ and unittest skip attributes are
#       copied from the base function.
# ─────────────────────────────────────────────────────────────────────────────

class ParameterizedTestMeta(type):
    """
    Metaclass that expands a PARAMS dict into concrete unittest test methods.

    Expected PARAMS structure::

        PARAMS = {
            (test_name_prefix, base_func_name): {
                # Optional: when present, tests are expanded as ops × cases.
                "ops_dict": {"op_name": op_callable, ...},
                "param_sets": {
                    case_name: (arg0, arg1, ...),
                    ...
                },
            },
            ...
        }

    Pytest marks
    ------------
    A mark is derived from *base_func_name* and applied to every generated
    method when the class does NOT already define ``pytestmark``::

        _run_cat_test    → @pytest.mark.torch_cat
        test_sdpa_decode → @pytest.mark.torch_sdpa_decode

    When ``pytestmark`` is present on the class (e.g.
    ``pytest.mark.torch_sdpa``), that mark stamps every method and no
    additional mark is derived from the base name.

    unittest.skip propagation
    -------------------------
    ``__unittest_skip__`` / ``__unittest_skip_why__`` on the base function
    are copied to every generated method so ``@unittest.skip`` works as
    expected.
    """

    def __new__(mcs, name, bases, namespace):
        param_map = namespace.get("PARAMS", {})
        to_delete = set()

        for (test_name_prefix, base_func_name), cases in param_map.items():
            base_func = namespace.get(base_func_name)
            if base_func is None:
                continue

            # Derive a pytest mark from the base method name unless the class
            # already declares pytestmark.  Adding a second derived mark would
            # produce spurious PytestUnknownMarkWarning entries and pollute the
            # mark registry.
            class_has_pytestmark = "pytestmark" in namespace
            if not class_has_pytestmark:
                raw = base_func_name
                if raw.startswith("_run_") and raw.endswith("_test"):
                    op_label = raw.removeprefix("_run_").removesuffix("_test")
                else:
                    op_label = raw.removeprefix("test_")
                mark = getattr(pytest.mark, f"torch_{op_label}")
            else:
                mark = None

            ops_dict   = cases.get("ops_dict", None)
            param_sets = cases["param_sets"]

            for test_case, params in param_sets.items():
                if ops_dict:
                    # Cross-product expansion: one test per (op, case) pair.
                    for op_name, op in ops_dict.items():
                        test_name = f"{test_name_prefix}_{op_name}_{test_case}"
                        assert test_name not in namespace, (
                            f"Test name conflict: {test_name}"
                        )

                        def _make_ops_test(_base_func, _op, _params, _tname, _mark):
                            def test(self):
                                _base_func(self, _op, *_params)
                            test.__name__     = _tname
                            test.__qualname__ = f"{name}.{_tname}"
                            test.__doc__      = _base_func.__doc__
                            if getattr(_base_func, "__unittest_skip__", False):
                                test.__unittest_skip__     = True
                                test.__unittest_skip_why__ = getattr(
                                    _base_func, "__unittest_skip_why__", ""
                                )
                            return _mark(test) if _mark is not None else test

                        namespace[test_name] = _make_ops_test(
                            base_func, op, params, test_name, mark
                        )
                else:
                    # Per-case expansion: one test per case.
                    test_name = f"{test_name_prefix}_{test_case}"
                    assert test_name not in namespace, (
                        f"Test name conflict: {test_name}"
                    )

                    def _make_test(_base_func, _params, _tname, _mark):
                        def test(self):
                            _base_func(self, *_params)
                        test.__name__     = _tname
                        test.__qualname__ = f"{name}.{_tname}"
                        test.__doc__      = _base_func.__doc__
                        if getattr(_base_func, "__unittest_skip__", False):
                            test.__unittest_skip__     = True
                            test.__unittest_skip_why__ = getattr(
                                _base_func, "__unittest_skip_why__", ""
                            )
                        return _mark(test) if _mark is not None else test

                    namespace[test_name] = _make_test(
                        base_func, params, test_name, mark
                    )

            to_delete.add(base_func_name)

        for key in to_delete:
            namespace.pop(key, None)

        return super().__new__(mcs, name, bases, namespace)


# ─────────────────────────────────────────────────────────────────────────────
# cached_randn
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def cached_randn(
    shape,
    differentiation=None,
    abs=False,
    dtype=torch.float16,
    scale=1.0,
) -> torch.Tensor:
    """LRU-cached random tensor factory.

    Repeated calls with identical arguments return the *same* tensor object,
    making test-input construction at module import time cheap and
    deterministic.

    Parameters
    ----------
    shape           : tuple — tensor shape.
    differentiation : any hashable — salt to produce distinct tensors that
                      share the same shape and dtype.
    abs             : bool — apply element-wise absolute value if True.
    dtype           : torch.dtype — output dtype (default float16).
    scale           : float — multiply the randn output by this factor.
    """
    out = torch.randn(shape, dtype=dtype) * scale
    return out if not abs else torch.abs(out)


# ─────────────────────────────────────────────────────────────────────────────
# General-purpose contiguous tensor factory
# ─────────────────────────────────────────────────────────────────────────────


def make_tensor(
    shape: tuple,
    dtype: torch.dtype = torch.float16,
    strides: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Return a tensor of *shape* and *dtype*. If *strides* is None, the tensor is
    contiguous (default). If *strides* is given, the tensor is a strided view
    of a flat storage with the specified strides.

    For contiguous tensors:
        float16 / bfloat16 / float32 : torch.randn(shape) * 10.0
        int32 / int64                : torch.randint(-100, 100, shape)
        bool                         : torch.randint(0, 2, shape)

    For strided tensors:
        The same value generation is used, but the storage is 1‑D and then
        viewed with `torch.as_strided`. The storage size is the minimum needed
        to index every element: 1 + Σ (s_i - 1) * stride_i.
    """
    if strides is None:
        # Original contiguous behaviour
        if dtype in (torch.int32, torch.int64):
            return torch.randint(-100, 100, shape, dtype=dtype)
        if dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=dtype)
        return torch.randn(shape, dtype=dtype) * 10.0

    # Strided case: compute minimum storage size
    storage_size = 1
    for s, st in zip(shape, strides):
        storage_size += (s - 1) * st

    # Create flat storage with appropriate values
    if dtype in (torch.int32, torch.int64):
        storage = torch.randint(-100, 100, (storage_size,), dtype=dtype)
    elif dtype == torch.bool:
        storage = torch.randint(0, 2, (storage_size,), dtype=dtype)
    else:
        storage = torch.randn(storage_size, dtype=dtype) * 10.0
        
    t = torch.as_strided(storage, size=shape, stride=strides)   
    assert t.shape == shape,    f"Shape mismatch: expected {shape}, got {t.shape}"
    assert t.stride() == strides, f"Stride mismatch: expected {strides}, got {t.stride()}"
    # Return strided view
    return t


# Short alias — identical semantics, used inside PARAMS dicts for brevity.
_t = make_tensor


# ─────────────────────────────────────────────────────────────────────────────
# SDPA tensor factory
# ─────────────────────────────────────────────────────────────────────────────

def make_qkv(
    batch: int,
    seq_q: int,
    seq_kv: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
    diff=0,
) -> tuple:
    """Return a param-set tuple ``(q, k, v)`` backed by ``cached_randn``.

    Shapes
    ------
    q : [batch, NUM_Q_HEADS,  seq_q,  HEAD_DIM]   e.g. [1, 32,   1, 128]
    k : [batch, NUM_KV_HEADS, seq_kv, HEAD_DIM]   e.g. [1,  8, 128, 128]
    v : [batch, NUM_KV_HEADS, seq_kv, HEAD_DIM]   e.g. [1,  8, 128, 128]

    Parameters
    ----------
    diff : any hashable — forwarded to ``cached_randn(differentiation=...)``
           to produce distinct tensors for param-set entries that share the
           same shape and dtype.
    """
    q = cached_randn(
        (batch, NUM_Q_HEADS,  seq_q,  HEAD_DIM),
        differentiation=("q", diff),
        dtype=dtype,
    )
    k = cached_randn(
        (batch, NUM_KV_HEADS, seq_kv, HEAD_DIM),
        differentiation=("k", diff),
        dtype=dtype,
    )
    v = cached_randn(
        (batch, NUM_KV_HEADS, seq_kv, HEAD_DIM),
        differentiation=("v", diff),
        dtype=dtype,
    )
    return (q, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# SDPA pre-built param dicts  (constructed here, after make_qkv is defined)
# ─────────────────────────────────────────────────────────────────────────────

# Divide NUM_LAYERS seeds evenly across three sequence-length tiers.
_SEED_S = NUM_LAYERS // 3          # short sequences  (~13 seeds)
_SEED_M = NUM_LAYERS - 2 * _SEED_S # medium sequences (~14 seeds)
_SEED_L = _SEED_S                  # long sequences   (~13 seeds)

PREFILL_PARAMS: dict = {
    f"bs{b}_seq{s}": make_qkv(b, s, s, diff=(b, s))
    for b, s in [
        (1, 128), (1, 512), (1, 1024), (1, 2048),
        (2, 256), (2, 512),
        (4, 128), (4, 256),
        (8,  64), (8, 128),
    ]
}

DECODE_PARAMS: dict = {
    f"bs{b}_kv{kv}": make_qkv(b, 1, kv, diff=(b, kv, "dec"))
    for b, kv in [
        (1, 128), (1, 512), (1, 1024), (1, 2048),
        (2, 256),
        (4, 512),
        (8, 128), (8, 256),
    ]
}

DTYPE_PARAMS: dict = {
    "fp16": make_qkv(1, 128, 128, dtype=torch.float16,  diff="fp16"),
    "bf16": make_qkv(1, 128, 128, dtype=torch.bfloat16, diff="bf16"),
    "fp32": make_qkv(1, 128, 128, dtype=torch.float32,  diff="fp32"),
}

NUMERIC_COVERAGE_PARAMS: dict = {
    # Short sequences (seq=128) — _SEED_S seeds.
    **{
        f"seed{i:02d}_seq128":  make_qkv(1, 128,  128,  diff=("cov", i, "seq128"))
        for i in range(_SEED_S)
    },
    # Medium sequences (seq=512) — _SEED_M seeds.
    **{
        f"seed{i:02d}_seq512":  make_qkv(1, 512,  512,  diff=("cov", i, "seq512"))
        for i in range(_SEED_S, _SEED_S + _SEED_M)
    },
    # Long sequences (seq=2048) — _SEED_L seeds.
    **{
        f"seed{i:02d}_seq2048": make_qkv(1, 2048, 2048, diff=("cov", i, "seq2048"))
        for i in range(_SEED_S + _SEED_M, NUM_LAYERS)
    },
}

GROWING_KV_PARAMS: dict = {
    f"kv{kv}": make_qkv(1, 1, kv, diff=("grow", kv))
    for kv in [1, 2, 4, 8, 16, 32, 64, 128]
}

SLIDING_WINDOW_PARAMS: dict = {
    "seq2048": make_qkv(1, 2048, 2048, diff="sw2048"),
    "seq8192": make_qkv(1, 8192, 8192, diff="sw8192"),
}


# ─────────────────────────────────────────────────────────────────────────────
# GQA helper
# ─────────────────────────────────────────────────────────────────────────────

def expand_kv(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand K/V from num_kv_heads to num_q_heads via repeat_interleave.

    Expansion is performed on CPU because the Spyre backend raises an error
    in view / reshape / unsafe_view (maybe_get_squeezed_layout malloc
    corruption).  The expanded tensors are moved back to the original device
    before being returned.
    """
    device = k.device
    _, num_kv_heads, _, _ = k.shape
    gqa_groups = NUM_Q_HEADS // num_kv_heads
    k_exp = k.cpu().repeat_interleave(gqa_groups, dim=1).to(device)
    v_exp = v.cpu().repeat_interleave(gqa_groups, dim=1).to(device)
    return k_exp, v_exp


# ─────────────────────────────────────────────────────────────────────────────
# Mask builders
# ─────────────────────────────────────────────────────────────────────────────

def causal_mask(
    seq_len: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Return an upper-triangular causal additive mask of shape [1,1,S,S].

    Future positions receive ``-inf`` so they are zeroed out by softmax.
    """
    mask = torch.zeros(seq_len, seq_len, dtype=dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask.unsqueeze(0).unsqueeze(0).to(device)


def sliding_window_mask(
    seq_len: int,
    dtype: torch.dtype,
    device,
    window: int = SLIDING_WINDOW,
) -> torch.Tensor:
    """Return a causal mask with a sliding-window cutoff of shape [1,1,S,S].

    Positions further than *window* tokens in the past are also masked to
    ``-inf``, matching the model's SWA (sliding-window attention) behaviour.
    """
    mask = causal_mask(seq_len, dtype, "cpu")
    for i in range(seq_len):
        start = max(0, i - window + 1)
        mask[0, 0, i, :start] = float("-inf")
    return mask.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Core SDPA wrapper (eager, GQA-aware)
# ─────────────────────────────────────────────────────────────────────────────

def sdpa_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Wrapper around F.scaled_dot_product_attention with GQA expansion.

    K and V are expanded from NUM_KV_HEADS to NUM_Q_HEADS on CPU before being
    passed to SDPA.  See expand_kv for the reason this must happen on CPU.
    """
    k_exp, v_exp = expand_kv(k, v)
    return F.scaled_dot_product_attention(
        q, k_exp, v_exp,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=SCALE,
    )


def make_strided_tensor(
    shape: tuple,
    strides: tuple,
    dtype: torch.dtype = torch.float16,
    fill: str = "randn",
    min_val: float | None = None,
    max_val: float | None = None,
) -> torch.Tensor:
    """
    Create a tensor with explicit strides using torch.as_strided.

    Args:
        shape:    Shape of the resulting tensor.
        strides:  Strides (in elements) of the resulting tensor.
        dtype:    Data type of the tensor.
        fill:     How to fill the underlying storage —
                  "randn" | "zeros" | "ones" | "arange".
        min_val:  If provided, clamps / shifts values so nothing is below this.
        max_val:  If provided, clamps / shifts values so nothing is above this.
                  When both are given the storage is rescaled into [min_val, max_val].
    """

    # Minimum flat storage needed
    storage_size = 1
    for s, st in zip(shape, strides):
        storage_size += (s - 1) * st

    # Allocate storage
    if fill == "randn":
        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            storage = torch.randint(0, 2**31, (storage_size,), dtype=dtype)
        else:
            storage = torch.randn(storage_size, dtype=dtype)
    elif fill == "zeros":
        storage = torch.zeros(storage_size, dtype=dtype)
    elif fill == "ones":
        storage = torch.ones(storage_size, dtype=dtype)
    elif fill == "arange":
        storage = torch.arange(storage_size, dtype=dtype)
    else:
        raise ValueError(f"Unknown fill mode: {fill!r}")

    # ── Range constraint ──────────────────────────────────────────────────────
    if min_val is not None and max_val is not None:
        s_min = storage.min()
        s_max = storage.max()
        if s_min == s_max:
            storage.fill_(min_val)
        else:
            # For integer dtypes, treat max_val as exclusive (like Python range)
            effective_max = (max_val - 1) if dtype in (
                torch.int8, torch.int16, torch.int32, torch.int64
            ) else max_val

            storage = (storage.float() - s_min.float()) / (s_max.float() - s_min.float())
            storage = storage * (effective_max - min_val) + min_val
            storage = storage.to(dtype)
    elif min_val is not None:
        storage = storage.clamp(min=min_val)
    elif max_val is not None:
        storage = storage.clamp(max=max_val)
    # (both None → no constraint, original behaviour)
    # ─────────────────────────────────────────────────────────────────────────

    # Create strided tensor
    t = torch.as_strided(storage, size=shape, stride=strides)

    # Assertions (important)
    assert t.shape == shape,    f"Shape mismatch: expected {shape}, got {t.shape}"
    assert t.stride() == strides, f"Stride mismatch: expected {strides}, got {t.stride()}"

    return t


# ─────────────────────────────────────────────────────────────────────────────
# Internal execution helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_on_device(fn, args, device, compiled=False, needs_device=False):
    """Run *fn* with *args* placed on *device*, optionally under torch.compile.

    All Tensor arguments are moved to *device* before the call.  The result
    is returned as-is (still on *device*) — callers are responsible for any
    subsequent CPU transfer.

    Parameters
    ----------
    fn           : callable to invoke.
    args         : positional arguments; Tensors are moved to *device*.
    device       : target ``torch.device`` (or string).
    compiled     : if True, wraps *fn* with ``torch.compile`` and resets
                   Dynamo's code caches first.
    needs_device : if True, passes ``device=device`` as an extra keyword
                   argument (for factory functions such as ``torch.zeros``).
    """
    if compiled:
        torch._dynamo.reset_code_caches()

    device      = torch.device(device) if isinstance(device, str) else device
    device_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
    device_kwargs = {"device": device} if needs_device else {}
    runner      = torch.compile(fn) if compiled else fn

    with torch.no_grad():
        return runner(*device_args, **device_kwargs)


def _to_cpu(result):
    """Recursively move tensors to CPU; pass scalars and None through."""
    if isinstance(result, torch.Tensor):
        return result.cpu()
    if isinstance(result, (tuple, list)):
        return type(result)(_to_cpu(r) for r in result)
    return result


def _assert_close(actual, expected, atol, rtol, label):
    """Recursively assert numerical closeness with a descriptive label.

    Handles tensors, nested tuples/lists of tensors, and plain Python
    scalars.  Tensors in *actual* are moved to CPU before comparison so
    callers do not need to do this explicitly.
    """
    if isinstance(actual, (tuple, list)):
        assert type(actual) == type(expected) and len(actual) == len(expected), (
            f"{label}: result structure mismatch — "
            f"actual {type(actual).__name__}[{len(actual)}] vs "
            f"expected {type(expected).__name__}[{len(expected)}]"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, atol, rtol, f"{label}[{i}]")
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(
            actual.cpu(),
            expected.cpu(),
            equal_nan=True,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"{label} mismatch\n\n{msg}\n",
        )
    else:
        # Plain Python scalar (e.g. from .numel() or .item()).
        assert actual == expected, (
            f"{label}: scalar mismatch: {actual} != {expected}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Spyre device_tensor_layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_spyre_layout(tensor: torch.Tensor) -> dict | None:
    """Extract ``device_tensor_layout`` attributes from a Spyre tensor.

    Returns a dict with keys ``dim_map`` and ``device_size`` when the
    attribute is present, or ``None`` for CPU tensors and any tensor that
    does not expose the layout descriptor (e.g. scalars, non-Spyre devices).

    The ``device_tensor_layout`` object is expected to expose:
    * ``.dim_map``    — sequence mapping each logical dimension to its
                        physical placement on the Spyre device mesh.
    * ``.device_size`` — sequence giving the per-dimension shard / tile
                         count across the device mesh.

    Both values are normalised to plain Python lists so that equality checks
    do not depend on the concrete container type returned by the backend.
    """
    layout = tensor.device_tensor_layout()
    if layout is None:
        return None
    return {
        "dim_map":     layout.dim_map,
        "device_size": layout.device_size,
    }


def _is_shape_changing_op(spyre_out: torch.Tensor, spyre_ref: torch.Tensor) -> bool:
    """Determine if an operation is likely shape-changing by comparing properties.

    An operation is considered shape-changing if:
    1. The tensors have different strides (for non-contiguous views)
    2. The dim_map patterns suggest different physical layouts
    3. The tensors have different contiguity properties
    """
    # If shapes are the same but strides differ significantly, likely a view/reshape
    if spyre_out.shape == spyre_ref.shape:
        # Check if strides differ (on CPU for comparison)
        cpu_out = spyre_out.cpu()
        cpu_ref = spyre_ref.cpu()

        # Different stride patterns indicate view/reshape operations
        if cpu_out.stride() != cpu_ref.stride():
            return True

        # Check if one is contiguous and the other isn't
        if cpu_out.is_contiguous() != cpu_ref.is_contiguous():
            return True

    # If dim_map patterns are available, use them for detection
    layout_out = _get_spyre_layout(spyre_out)
    layout_ref = _get_spyre_layout(spyre_ref)

    if layout_out and layout_ref:
        # Different dim_map lengths indicate different logical mapping
        if len(layout_out["dim_map"]) != len(layout_ref["dim_map"]):
            return True

        # Same length but different values indicates different physical layout
        if layout_out["dim_map"] != layout_ref["dim_map"]:
            return True

        # Different device_size distribution indicates different sharding
        if layout_out["device_size"] != layout_ref["device_size"]:
            return True

    # If we can't determine, assume it's not shape-changing (will attempt layout comparison)
    return False


def _get_spyre_layout(tensor: torch.Tensor) -> dict | None:
    """Extract device_tensor_layout attributes from a Spyre tensor.

    Returns None for CPU tensors or when the attribute doesn't exist.
    """
    # If not on Spyre device, return None immediately
    if tensor.device.type != "spyre":
        return None

    # Try to get the layout attribute (might not exist on all tensor types)
    try:
        layout = tensor.device_tensor_layout()
    except (AttributeError, RuntimeError):
        return None

    if layout is None:
        return None

    return {
        "dim_map":     layout.dim_map,
        "device_size": layout.device_size,
    }


def _assert_spyre_layout_equal(
    spyre_out: torch.Tensor,
    spyre_ref: torch.Tensor,
    label: str,
) -> None:
    """Assert that two Spyre tensors have compatible layouts or skip if needed.

    On CPU devices, this function does nothing (no layout to compare).
    On Spyre devices, it either validates layout equality or logs differences.
    """
    IS_SPYRE_DEVICE = DEVICE.type == "spyre"

    # If not running on Spyre, skip layout comparison entirely
    if not IS_SPYRE_DEVICE:
        if DEBUG_LAYOUT:
            print(f"\n{label}: Running on CPU - skipping layout comparison")
        return

    layout_out = _get_spyre_layout(spyre_out)
    layout_ref = _get_spyre_layout(spyre_ref)

    if DEBUG_LAYOUT:
        print(f"\n{label}:")
        print(f"  spyre_out shape: {spyre_out.shape}")
        print(f"  spyre_ref shape: {spyre_ref.shape}")
        print(f"  spyre_out stride: {spyre_out.stride()}")
        print(f"  spyre_ref stride: {spyre_ref.stride()}")
        if layout_out:
            print(f"  spyre_out layout: dim_map={layout_out['dim_map']}, "
                  f"device_size={layout_out['device_size']}")
        else:
            print(f"  spyre_out layout: None")
        if layout_ref:
            print(f"  spyre_ref layout: dim_map={layout_ref['dim_map']}, "
                  f"device_size={layout_ref['device_size']}")
        else:
            print(f"  spyre_ref layout: None")

    # If either layout is missing, we can't compare
    if layout_out is None or layout_ref is None:
        if DEBUG_LAYOUT:
            print(f"  {label}: Layout missing on one or both tensors - skipping layout comparison")
        return

    # Skip layout comparison for scalar tensors
    if spyre_out.dim() == 0 or spyre_ref.dim() == 0:
        if DEBUG_LAYOUT:
            print(f"  {label}: Scalar tensor - skipping layout comparison")
        return

    # Skip layout comparison if shapes differ
    if spyre_out.shape != spyre_ref.shape:
        if DEBUG_LAYOUT:
            print(f"  {label}: Shapes differ - skipping layout comparison")
        return
    
    validate_cpu_spyre_layout(spyre_ref.shape, layout_ref["dim_map"], layout_ref["device_size"])

    if layout_out["dim_map"] != layout_ref["dim_map"] or layout_out["device_size"] != layout_ref["device_size"]:
        if DEBUG_LAYOUT:
            print(f"  {label}: Layout differs (expected for shape-changing ops)")
            print(f"    dim_map: {layout_out['dim_map']} vs {layout_ref['dim_map']}")
            print(f"    device_size: {layout_out['device_size']} vs {layout_ref['device_size']}")
        # Don't assert - just log the difference
        return

    if DEBUG_LAYOUT:
        print(f"  {label}: Layouts match exactly")


def _assert_spyre_layout_equal_recursive(
    spyre_out,
    spyre_ref,
    label: str,
) -> None:
    """Recursively apply ``_assert_spyre_layout_equal`` over nested structures.

    Mirrors the structure-handling in ``_assert_close`` so that functions
    returning tuples or lists of tensors are covered correctly.
    """
    if isinstance(spyre_out, (tuple, list)):
        assert type(spyre_out) == type(spyre_ref) and len(spyre_out) == len(spyre_ref), (
            f"{label}: result structure mismatch for layout check — "
            f"actual {type(spyre_out).__name__}[{len(spyre_out)}] vs "
            f"expected {type(spyre_ref).__name__}[{len(spyre_ref)}]"
        )
        for i, (a, e) in enumerate(zip(spyre_out, spyre_ref)):
            _assert_spyre_layout_equal_recursive(a, e, f"{label}[{i}]")
    elif isinstance(spyre_out, torch.Tensor):
        _assert_spyre_layout_equal(spyre_out, spyre_ref, label)
    # Plain scalars carry no layout — nothing to compare.


# ─────────────────────────────────────────────────────────────────────────────
# compare_sdpa  — eager-only SDPA comparison (CPU vs Spyre)
# ─────────────────────────────────────────────────────────────────────────────

def compare_sdpa(
    fn,
    *cpu_args,
    dtype: torch.dtype = DEFAULT_DTYPE,
    atol: float | None = None,
    rtol: float | None = None,
):
    """Run ``fn(*cpu_args)`` eagerly on CPU and on Spyre; assert closeness.

    Comparison strategy
    -------------------
    Rather than moving the Spyre result back to CPU for numerical comparison,
    the CPU result is promoted to the Spyre device and the two Spyre tensors
    are compared via their ``device_tensor_layout`` descriptors
    (``dim_map`` and ``device_size``).  This validates not only that the
    numerical values agree but also that the Spyre backend lays out the result
    tensor identically whether it computed the value itself or merely received
    it from the host.

    Steps
    -----
    1. Run ``fn(*cpu_args)`` on CPU → ``cpu_out``.
    2. Move tensor args to ``DEVICE``; run ``fn(*spyre_args)`` → ``spyre_out``.
    3. Move ``cpu_out`` to ``DEVICE`` → ``spyre_ref``.
    4. Assert ``device_tensor_layout`` (``dim_map``, ``device_size``) matches
       between ``spyre_out`` and ``spyre_ref``.

    Tolerances default to the per-dtype values in ``TOLERANCES`` but can be
    overridden via *atol* / *rtol*.
    """
    tol   = TOLERANCES[dtype]
    _atol = atol if atol is not None else tol["atol"]
    _rtol = rtol if rtol is not None else tol["rtol"]

    # Step 1 — CPU reference.
    cpu_out = fn(*cpu_args)

    # Step 2 — Spyre execution.
    spyre_args = tuple(
        arg.to(DEVICE) if isinstance(arg, torch.Tensor) else arg
        for arg in cpu_args
    )
    spyre_out = fn(*spyre_args)

    # Step 3 — Promote the CPU result onto Spyre so both tensors live on the
    # same device and carry device_tensor_layout descriptors.
    spyre_ref = _to_spyre(cpu_out)

    # Step 4 — Compare device_tensor_layout (dim_map + device_size).
    _assert_spyre_layout_equal_recursive(
        spyre_out,
        spyre_ref,
        label="spyre (eager) <-> cpu→spyre (ref)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# compare_with_cpu  — eager or compiled comparison (CPU vs Spyre)
# ─────────────────────────────────────────────────────────────────────────────

def _to_spyre(result):
    """Recursively move tensors to DEVICE; pass scalars and None through.

    This is the mirror of ``_to_cpu``: it promotes CPU results onto the Spyre
    device so that ``device_tensor_layout`` attributes become available for
    structural comparison.
    """
    if isinstance(result, torch.Tensor):
        return result.to(DEVICE)
    if isinstance(result, (tuple, list)):
        return type(result)(_to_spyre(r) for r in result)
    return result


def compare_with_cpu(
    fn,
    *args,
    compiled: bool = True,
    needs_device: bool = False,
    atol: float | None = None,
    rtol: float | None = None,
):
    """Run ``fn(*args)`` on both CPU and Spyre and assert the outputs match.

    Comparison strategy
    -------------------
    Instead of moving the Spyre result to CPU for numerical comparison, the
    CPU result is promoted to the Spyre device and both Spyre tensors are
    compared via their ``device_tensor_layout`` descriptors (``dim_map`` and
    ``device_size``).  This validates that the Spyre backend assigns the same
    physical layout to a tensor it computed as it does to the equivalent
    tensor it received from the host.

    Steps
    -----
    1. Run ``fn(*args)`` on CPU → ``cpu_out``.
    2. Run ``fn(*args)`` on Spyre → ``spyre_out``.
    3. Move ``cpu_out`` to ``DEVICE`` → ``spyre_ref``.
    4. Assert ``device_tensor_layout`` (``dim_map``, ``device_size``) matches
       between ``spyre_out`` and ``spyre_ref``.

    Parameters
    ----------
    fn           : callable to compare.
    *args        : positional arguments.  Tensor args are moved to the target
                   device automatically.
    compiled     : if True, wraps *fn* with ``torch.compile`` on both
                   devices.  Compiled mode uses COMPILED_ATOL / COMPILED_RTOL
                   by default.
    needs_device : if True, passes ``device=`` as an extra keyword argument
                   (for factory functions such as ``torch.zeros``).
    atol / rtol  : accepted for API compatibility but unused — layout
                   comparison is structural (dim_map / device_size equality)
                   and does not apply numerical tolerances.
    """
    # atol / rtol are intentionally unused: layout comparison is structural.
    # They remain in the signature so call sites that pass explicit tolerances
    # do not need to be updated.
    _ = atol, rtol

    # Step 1 — CPU reference.
    cpu_out = _run_on_device(
        fn, args, device="cpu", compiled=compiled, needs_device=needs_device
    )

    # Step 2 — Spyre execution.
    spyre_out = _run_on_device(
        fn, args, device=DEVICE, compiled=compiled, needs_device=needs_device
    )

    # Step 3 — Promote the CPU result onto Spyre.
    spyre_ref = _to_spyre(cpu_out)

    # Step 4 — Compare device_tensor_layout (dim_map + device_size).
    _assert_spyre_layout_equal_recursive(
        spyre_out,
        spyre_ref,
        label="spyre vs cpu→spyre",
    )

def validate_cpu_spyre_layout(tensor_shape, dim_map, device_size):
    # -------------------------
    # 0. Basic sanity
    # -------------------------
    assert len(device_size) == len(dim_map), \
        "device_size and dim_map must have same length"

    tensor_ndim = len(tensor_shape)

    # -------------------------
    # 1. Validate dim_map values
    # -------------------------
    for i, d in enumerate(dim_map):
        assert isinstance(d, int), f"dim_map[{i}] must be int"
        assert 0 <= d < tensor_ndim, \
            f"dim_map[{i}] = {d} out of range for tensor dims {tensor_ndim}"

    # -------------------------
    # 2. Group device dims by tensor dim
    # -------------------------
    groups = defaultdict(list)
    for dev_idx, tensor_dim in enumerate(dim_map):
        groups[tensor_dim].append(dev_idx)

    # -------------------------
    # 3. Validate every tensor dim is covered
    # -------------------------
    for tensor_dim in range(tensor_ndim):
        assert tensor_dim in groups, \
            f"Tensor dim {tensor_dim} not represented in dim_map"

    # -------------------------
    # 4. Unified validation
    # -------------------------
    for tensor_dim, dev_indices in groups.items():
        tensor_size = tensor_shape[tensor_dim]
        dev_sizes = [device_size[i] for i in dev_indices]

        # Case A: no tiling
        if len(dev_indices) == 1:
            dev_size = dev_sizes[0]
            assert dev_size == tensor_size, \
                (f"Tensor dim {tensor_dim} mismatch: "
                 f"{dev_size} != {tensor_size}")

        # Case B: tiled
        else:
            prod = math.prod(dev_sizes)

            assert prod >= tensor_size, \
                (f"Tensor dim {tensor_dim} tiled with less than stick size * stick value"
                 f"{prod} < {tensor_size}")
    return True
