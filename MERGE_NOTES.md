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

## Evaluation — two paths available

There are **two** ways to evaluate on this branch. Both reuse the *patched* MaxText
model, so unscaled (`scale_query=False`) checkpoints score correctly.

### Path A (default for likelihood/ppl): direct-forward `orbax_lm` adapter — **ported**
`test_new`'s direct-forward lm-eval adapter (`MaxText/inference/orbax_adapter.py`,
which only implemented `loglikelihood`) was **ported and completed** onto the new
tree at **`src/maxtext/eval/orbax/`**:
- `orbax_lm.py` — `OrbaxLM(LM)`, a complete lm-eval model that runs a **direct NNX
  forward pass** (no vLLM). Implements **all three** request types, so one instance
  covers every lm-eval task (`evaluator.py` dispatches `getattr(lm, request_type)`):
  `loglikelihood` (MC/QA), `loglikelihood_rolling` (ppl), `generate_until`
  (generation). Builds via `model_creation_utils.from_pretrained` → patched
  `attentions.py`, so the q/k scaling is applied here too.
- `orbax_eval.py` — driver ported from `scripts/test_orbax_eval.py`: `PPL_TASKS`
  (c4/wikitext/wikitext2/cnn_dailymail/dclm, custom forward-pass ppl loop) +
  `ACC_TASKS` (winogrande/arc/hellaswag/mmlu/… via `simple_evaluate`).

```bash
pip install lm_eval datasets       # no vllm-tpu needed; no harness re-vendoring
python -m maxtext.eval.orbax.orbax_eval \
  src/maxtext/configs/base.yml \
  model_name=llama3.1-4b-depth \
  load_parameters_path=gs://<bucket>/.../checkpoints/<step>/items \
  max_target_length=8192 \
  --hf_model_path=meta-llama/Llama-3.1-8B --limit=1000
```
⚠️ `generate_until` here is a **dense** decode loop (one forward pass per token, no
KV cache) — correct but slow; fine for small/`--limit`-ed gen sets. For heavy
generation, use Path B. All standard ACC tasks are stock lm-eval tasks (no
re-vendoring); dclm loads from a local JSONL (`--dclm_path`, default
`~/gcs-bucket/datasets/dclm/...`). Not runtime-tested here (no jax/TPU) — validate
on-device (see Verification).

### Path B (for generation / throughput): official vLLM framework
The official vLLM-server framework under `src/maxtext/eval/runner/` boots the
tpu-vLLM engine on your checkpoint → OpenAI `/v1/completions` → stock lm-eval
`local-completions`. Best for generation tasks (efficient paged-attention decode)
and large-scale runs.

```bash
pip install "lm_eval[api]"          # + vllm-tpu / tpu_inference on the host
python -m maxtext.eval.runner.run \
  --runner lm_eval \
  --checkpoint_path gs://<bucket>/.../checkpoints/<step>/items \
  --model_name llama3.1-4b-depth \
  --hf_path meta-llama/Llama-3.1-8B \
  --tasks gsm8k ifeval hellaswag arc_challenge winogrande piqa wikitext \
  --num_fewshot 0 \
  --base_output_directory gs://<bucket>/ --run_name eval_4b_depth \
  --max_model_len 8192 --tensor_parallel_size 4 --hf_token $HF_TOKEN
```
Multiple-choice → `acc`/`acc_norm`; perplexity → `wikitext` (loglikelihood_rolling;
server supports it via `echo`+`prompt_logprobs`); generation → gsm8k/ifeval, etc.

**Why this is correct for our unscaled checkpoints:** the vLLM path reuses the
*patched* model — `MaxTextForCausalLM` →
`model_creation_utils.from_pretrained(maxtext_config)` → the standard decoder →
our patched `layers/attentions.py`. The `1/sqrt(head_dim)` forward-scaling
(`attentions.py` ~L1226-1230) sits **above** the vLLM RPA dispatch (~L1259) and
`forward_serve_vllm` calls the RPA kernel with **sm_scale=1.0**, so scaling is
applied exactly once (no double-scale) on the eval path. Pruned configs load by
`--model_name`.

⚠️ Path B caveats: (1) numbers come through vLLM RPA + KV cache, not a dense
forward pass — very close but not bit-identical to old `test_new` numbers, so
**re-baseline** reference checkpoints before comparing. (2) Requires
`vllm-tpu`/`tpu_inference` on the host. (3) Pruned **width** models
(emb_dim ≠ heads·head_dim) and **FLAP** (bias) through RPA are the untested
corners — smoke-test one first.

> **Which to use.** Path A (direct forward pass) uses the *same* dense-forward
> mechanism as your old `test_new` eval, so it should **reproduce your historical
> ppl/accuracy numbers** — prefer it for likelihood + ppl and for apples-to-apples
> comparison with prior results. Use Path B when you need generation throughput or
> large-scale runs.

---

## Ported after the initial merge (2026-07-25)
- **`sparse_model_training` (prune-and-freeze), from test_new** — keeps externally-pruned
  weights frozen at 0 during continued training by masking gradients where the weight is
  exactly 0 (so optimizer moments stay 0). Sparsity-pattern-agnostic (unstructured
  Wanda/SparseGPT or 2:4). Implementation:
  - `utils/maxtext_utils.py`: `apply_gradient_mask(grads, params, ref_params=None)` and
    `check_sparsity(params)` (logs achieved sparsity).
  - `configs/types.py` `Optimizer.sparse_model_training` field + `configs/base.yml` key.
  - `trainers/pre_train/train.py` `train_step` (**Linen path**): mask grads after clipping,
    gated OFF when the qwix N:M `weight_sparsity_n/m` path is active (it re-wraps grads into
    a `{'params',...}` dict), and log `learning/zeros` + `learning/params`.
  - ⚠️ **Linen-only** (mirrors test_new). The NNX `train_step` branch is NOT covered — if a
    pruned run uses the NNX trainer, add an NNX-specific mask (against `nnx.state(model, nnx.Param)`).
  - Distinct from official's qwix N:M `weight_sparsity` (which *creates* structured masks by
    magnitude during training); this *preserves* an externally-produced mask of any pattern.
- **Config-strictness fixes** (current official validates config against a strict pydantic
  schema, `extra="forbid"` + `Literal`-validated `model_name` — the old free-form assumption
  in this doc is wrong): (1) the FLAP bias plumbing now uses official's registered
  `attention_bias`/`mlp_bias` instead of the custom `use_bias_in_projections`/`use_bias_in_mlp`
  keys (which broke every model load); (2) the 23 pruned model config names are registered in
  the `ModelName` Literal in `configs/types.py` (files alone were not enough).

## Not ported (available in `test_new` if wanted later)
Per the "core only" scope:
- Workflow features: W&B logging, `StopTraining.is_error`/`sys.exit(1)` exit-codes,
  single-replica checkpoint restore, FMS-style LR schedule,
  resumable grain data-sharding, intermediate-activation
  `sow`, HF-style embedding init. (The `orbax` lm-eval adapter **was ported** — see
  **Evaluation** Path A.)
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
- All edited Python files pass `py_compile`.
- **On-device test (2026-07-25, v4-8 TPU, us-central2-b).** Validated: the branch installs
  (uv Python 3.12 + `pip install -e '.[tpu]'`, jax 0.11.0, 4 devices), config loads, an 8B
  Orbax checkpoint **restores**, and the HF tokenizer loads. This test is what surfaced the
  two config-strictness bugs fixed above. Install prerequisite: py≥3.12 + pyproject `.[tpu]`
  (the stock `maxtext_bootstrap.sh`, which builds a py3.10 venv from a now-absent
  `requirements.txt`, does NOT install this branch and needs updating).
- **KNOWN OPEN BUG — Path A numeric eval blocked (orthogonal to this merge's code).**
  `model_creation_utils.from_pretrained` restoring a **scanned Linen checkpoint into the NNX
  model** silently drops all 9 scanned `decoder.layers.*` leaves (mlp wi_0/wi_1/wo, both
  layernorms, self_attention q/k/v/out); non-scanned params restore fine; **dtype-independent**
  (same at float32 and bf16). Because `from_pretrained` pre-frees the init buffers
  (`_free_device_memory`) and `nnx.update(model, checkpoint)` only writes back keys present in
  the aligned tree, the dropped layers keep freed buffers → `RuntimeError: Array has been
  deleted` on the first forward. Root cause is the Linen→NNX key/scan mapping for the `layers`
  subtree, NOT the q/k / pruned-config / eval-adapter changes. Fix: correct the scanned
  Linen→NNX restore mapping, or load via a Linen path (as test_new's original adapter did).
- Still recommended once the above is fixed: run the parity checks (`test_eq.py`,
  `hf_out_test.py`) on a TPU/GPU host and diff logits against HF.
