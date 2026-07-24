# Merge notes — `merge-official-latest`

This branch = **latest official AI-Hypercomputer/maxtext `main`** (`123ec6008`, 2026-07-23,
`src/maxtext/` layout, includes the tpu-vLLM inference engine and distillation)
**+ ported custom changes** from the `Zephyr271828/maxtext` `test_new` branch.

`test_new` has a *disconnected* git history (fresh "initial commit", no common
ancestor with official), so a normal `git merge` was impossible. Its snapshot base
was official ≈ `2adc3bab1` (2025-07-22). The custom diff against that base was small
and localized, so it was re-applied semantically onto the restructured latest tree.

Scope chosen: **core only** = q/k handling + pruned-model configs + checkpoint
conversion. (Other `test_new` workflow features were intentionally *not* ported — see
"Not ported".)

---

## What was changed

### 1. Q/K attention scaling (the correctness fix)
**Decision: conditional forward scaling.** The `1/sqrt(head_dim)` attention scaling
is applied in the **forward pass** instead of being folded into the query weights, so
stored MaxText query weights stay **identical to the source (HF) weights** — required
for weight-magnitude pruning and exact HF round-tripping.

- `src/maxtext/layers/attentions.py`
  - `init_query_w`: stopped folding `1/sqrt(head_dim)` into the query initializer
    (`query_init` now returns `kernel_init` unchanged).
  - `__call__`: after RoPE/qk_norm, scale the query:
    - if `query_pre_attn_scalar` is set (Qwen3 = `head_dim**-0.5`, Gemma its own) → use it (unchanged);
    - **elif not `use_qk_norm`** → apply `1/sqrt(head_dim)` (the standard-model case).
  - Net effect: standard Llama-style models scale in the forward with **unscaled
    weights**; `qk_norm` / `query_pre_attn_scalar` models (Qwen3, gpt-oss, Gemma —
    the distillation targets) are **completely unchanged** (no double-scaling).
- `src/maxtext/checkpoint_conversion/standalone_scripts/llama_or_mistral_ckpt.py`
  - `scale_query` default flipped **True → False** (both conversion functions), so the
    HF→Orbax converter no longer pre-divides query weights.
- `src/maxtext/checkpoint_conversion/standalone_scripts/llama_mistral_mixtral_orbax_to_hf.py`
  - `reverse_scale()` made a **no-op** (weights are stored unscaled, so nothing to
    reverse on export). Standard-model export is correct with this change.

> ⚠️ **Checkpoint compatibility:** Orbax checkpoints previously converted by *official*
> MaxText (with `scale_query=True`, i.e. pre-scaled query weights) are **NOT compatible**
> with this branch — loading one would double-apply `1/sqrt(head_dim)`. Re-convert from
> the HF checkpoint with this branch's converter (which now stores unscaled weights).
> This matches `test_new`'s behavior and is by design.

### 2. Pruned / down-scaled Llama model support
- Copied **23 pruned-model configs** into `src/maxtext/configs/models/`
  (`llama3.1-4b-{width,depth,shear,flap}`, `llama3(.1)-{1,1.5,2,3,4}b-*`, `llama2-{1.3,2.7}b`,
  `llama3.1-{440m,1b,3b}`, …). The active `pyconfig.py` loads model configs by filename
  (no `valid_model_names` allow-list anymore), so no code change was needed to register them.
- `src/maxtext/configs/base.yml`: added `use_bias_in_projections` and `use_bias_in_mlp`
  (both default `false`) — needed by the FLAP variants.
- `src/maxtext/models/llama2.py`: thread `config.use_bias_in_projections` into `Attention`
  and `config.use_bias_in_mlp` into `MlpBlock`.
- `llama_or_mistral_ckpt.py`:
  - Added **21 pruned entries** to `MODEL_PARAMS_DICT` (with `base_emb_dim`, `base_mlp_dim`/
    `intermediate_size` where relevant).
  - Attention Q/K/V/O reshapes now use **`base_emb_dim`** (with a safe fallback to
    `num_query_heads*head_dim`) so width-pruned models — where `emb_dim ≠ heads*head_dim` —
    convert correctly. No-op for standard models.

### RoPE permute — kept official's conditional (NOT `test_new`'s unconditional disable)
Official already **skips** `permute_to_match_maxtext_rope` for llama3.1 (`model_size[:8]`
∈ `{"llama3.1","llama3.3"}`), which covers your main pruned models — and its import/export
permute logic round-trips consistently. So official's logic was left intact rather than
copying `test_new`'s unconditional disable. ⚠️ If your **llama3-** (non-.1) or **llama2-**
based pruned checkpoints were converted with the permute disabled, tell me and I'll gate it.

---

## Not ported (available in `test_new` if wanted later)
Per the "core only" scope:
- Workflow features: W&B logging, `StopTraining.is_error`/`sys.exit(1)` exit-codes,
  single-replica checkpoint restore, FMS-style LR schedule, sparse/pruning training,
  resumable grain data-sharding, the `orbax` lm-eval-harness adapter, intermediate-activation
  `sow`, HF-style embedding init.
- `base.yml` **global** RoPE default retuning (kept official defaults; put 8k / plain-RoPE
  values in your run configs instead).
- Conversion extras: fp32 (vs bf16) cast, `.bin` checkpoint support.
- **Export of *pruned* models to HF** (`orbax_to_hf`): needs `base_emb_dim` reshapes +
  programmatic `load_hf_model` configs for the custom sizes — deferred. Standard-model
  export works. **Import (HF→Orbax) of pruned models is done.**
- **FLAP checkpoint conversion**: the model can *use* bias (plumbing done), but converting an
  external FLAP HF checkpoint's bias tensors needs the ckpt-side bias-conversion logic — deferred.
  The other pruned variants (width/depth/shear, no bias) convert fine.

---

## Verification status
- All edited Python files pass `py_compile`. **No runtime/TPU test was possible** in this
  environment (no `jax`). Recommend running your parity tests (`test_eq.py`, `hf_out_test.py`)
  after `pip install` on a TPU/GPU host, converting one Llama-3.1 checkpoint with the new
  converter and checking logits against HF.
