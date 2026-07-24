# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct (no-vLLM) lm-evaluation-harness adapter for MaxText Orbax checkpoints.

This registers an ``orbax_lm`` lm-eval model that scores / generates by running a
**direct forward pass** on the in-process MaxText NNX model — no inference server.
It implements the full ``LM`` interface, so a single instance covers every task
type lm-eval dispatches (``evaluator.py`` calls ``getattr(lm, request_type)(reqs)``):

  * ``loglikelihood``          — multiple-choice / QA (ARC, HellaSwag, MMLU, ...)
  * ``loglikelihood_rolling``  — perplexity tasks (wikitext, ...)
  * ``generate_until``         — generation tasks (gsm8k, ifeval, ...)

Scaling correctness: the model is built by ``model_creation_utils.from_pretrained``
→ the standard MaxText decoder → the patched ``layers/attentions.py``. The
``1/sqrt(head_dim)`` query scaling therefore applies here exactly as in training,
so unscaled (``scale_query=False``) checkpoints score correctly.

Performance note: ``generate_until`` here is a *dense* decode loop (a full forward
pass per generated token, no KV cache) — correct but slow. It is intended for
small / ``--limit``-ed generation sets. For heavy generative benchmarks, prefer the
vLLM eval framework (``python -m maxtext.eval.runner.run --runner lm_eval``), whose
``generate_until`` runs on the paged-attention decode engine.

Requires ``pip install lm_eval``. Standard lm-eval tasks need no re-vendoring.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import flax.linen as nn
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from maxtext.common.common_types import MODEL_MODE_TRAIN

logger = logging.getLogger(__name__)


@nnx.jit
def _forward_logits_jit(model, input_ids, positions, segment_ids):
  """Full-sequence forward pass → logits [batch, seq_len, vocab].

  ``model`` is a weights-bound NNX module (from ``from_pretrained``); ``nnx.jit``
  splits/threads its state. ``enable_dropout``/``model_mode`` are compile-time
  constants. TRAIN mode gives dense (non-cached) logits at every position.
  """
  return model(
      decoder_input_tokens=input_ids,
      decoder_positions=positions,
      decoder_segment_ids=segment_ids,
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
  )


@register_model("orbax_lm")
class OrbaxLM(LM):
  """lm-eval ``LM`` backed by an in-process MaxText NNX model (direct forward pass).

  Construct via the driver (``orbax_eval.py``) and hand the instance to
  ``lm_eval.evaluator.simple_evaluate(model=<instance>, ...)`` — no entry-point
  registration is needed for instance-based use.

  Args:
    model: weights-bound NNX model from ``model_creation_utils.from_pretrained``.
    tokenizer: an HF tokenizer (``AutoTokenizer``) for the source model.
    config: the MaxText ``pyconfig`` HyperParameters used to build ``model``.
    mesh: the ``jax.sharding.Mesh`` the model is sharded over.
    max_gen_toks: default generation budget for ``generate_until``.
    pad_bucket: forward passes are right-padded up to a multiple of this many
      tokens (capped at ``max_length``) so XLA compiles a bounded set of shapes
      instead of one per unique sequence length.
  """

  def __init__(self, model, tokenizer, config, mesh, max_gen_toks: int = 256, pad_bucket: int = 128) -> None:
    super().__init__()
    self.model = model
    self.tokenizer = tokenizer
    self.config = config
    self.mesh = mesh
    self._max_length = int(config.max_target_length)
    self._max_gen_toks = int(max_gen_toks)
    self._pad_bucket = int(pad_bucket)
    # Token used to give the very first predicted token something to condition on.
    self.prefix_token_id = (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    )

  # ---- properties lm-eval reads -------------------------------------------------

  @property
  def eot_token_id(self) -> int:
    return self.tokenizer.eos_token_id

  @property
  def max_length(self) -> int:
    return self._max_length

  @property
  def max_gen_toks(self) -> int:
    return self._max_gen_toks

  # ---- tokenization -------------------------------------------------------------

  def tok_encode(self, string: str, left_truncate_len: int | None = None, add_special_tokens: bool | None = None) -> list[int]:
    del add_special_tokens  # CausalLM default: no extra specials (matches test_new adapter)
    encoding = self.tokenizer.encode(string)
    if left_truncate_len:
      encoding = encoding[-left_truncate_len:]
    return encoding

  def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
    # Re-attach any whitespace the task split onto the continuation, then encode
    # jointly so the boundary tokenizes exactly as the model would see it.
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
      continuation = context[-n_spaces:] + continuation
      context = context[:-n_spaces]
    whole_enc = self.tok_encode(context + continuation)
    context_enc = self.tok_encode(context)
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc

  # ---- core forward -------------------------------------------------------------

  def _bucket_len(self, length: int) -> int:
    b = self._pad_bucket
    return int(min(self._max_length, ((length + b - 1) // b) * b))

  def forward(self, input_ids) -> jax.Array:
    """Batched dense forward. ``input_ids``: [batch, seq_len] (equal length).

    Returns logits [batch, seq_len, vocab]. Used by the perplexity loop, where all
    chunks share one length (so a single compiled shape).
    """
    ids = np.asarray(input_ids, dtype=np.int32)
    if ids.ndim == 1:
      ids = ids[None, :]
    batch, seq_len = ids.shape
    positions = np.tile(np.arange(seq_len, dtype=np.int32), (batch, 1))
    segment_ids = np.ones((batch, seq_len), dtype=np.int32)
    with self.mesh, nn.logical_axis_rules(self.config.logical_axis_rules):
      logits = _forward_logits_jit(
          self.model, jnp.asarray(ids), jnp.asarray(positions), jnp.asarray(segment_ids)
      )
    return logits

  def _forward_single(self, token_ids: list[int]) -> jax.Array:
    """Forward one sequence (right-padded to a bucket). Returns logits [len, vocab]
    trimmed back to the real ``len(token_ids)``."""
    real_len = len(token_ids)
    padded = self._bucket_len(real_len)
    ids = np.zeros((1, padded), dtype=np.int32)
    ids[0, :real_len] = token_ids
    positions = np.zeros((1, padded), dtype=np.int32)
    positions[0, :real_len] = np.arange(real_len, dtype=np.int32)
    # Real tokens -> segment 1; padding -> segment 0 (isolated, so it cannot affect
    # real-token logits under the causal + segment mask).
    segment_ids = np.zeros((1, padded), dtype=np.int32)
    segment_ids[0, :real_len] = 1
    with self.mesh, nn.logical_axis_rules(self.config.logical_axis_rules):
      logits = _forward_logits_jit(
          self.model, jnp.asarray(ids), jnp.asarray(positions), jnp.asarray(segment_ids)
      )
    return logits[0, :real_len, :]

  # ---- loglikelihood (multiple-choice / QA) ------------------------------------

  def loglikelihood(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
    new_reqs = []
    for context, continuation in [req.args for req in requests]:
      if context == "":
        context_enc, continuation_enc = [self.prefix_token_id], self.tok_encode(continuation)
      else:
        context_enc, continuation_enc = self._encode_pair(context, continuation)
      new_reqs.append(((context, continuation), context_enc, continuation_enc))
    return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

  def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
    results: list[tuple[float, bool]] = []
    for _key, context_enc, continuation_enc in tqdm(requests, disable=disable_tqdm):
      cont_len = len(continuation_enc)
      inp = list(context_enc) + list(continuation_enc)
      # Left-truncate the context if the pair exceeds the model window; the
      # continuation (scored region) is always kept.
      if len(inp) > self._max_length:
        inp = inp[-self._max_length:]
      seq_len = len(inp)
      cont_start = seq_len - cont_len

      logits = self._forward_single(inp)                     # [seq_len, vocab]
      logprobs = jax.nn.log_softmax(logits, axis=-1)
      # Token continuation_enc[j] (absolute position cont_start+j) is predicted by
      # the logits at position cont_start+j-1.
      pred_idx = jnp.arange(cont_start - 1, cont_start - 1 + cont_len)
      pred_logprobs = jnp.take(logprobs, pred_idx, axis=0)   # [cont_len, vocab]
      cont_toks = jnp.asarray(continuation_enc, dtype=jnp.int32)
      tok_logprobs = jnp.take_along_axis(pred_logprobs, cont_toks[:, None], axis=-1)[:, 0]
      greedy_toks = jnp.argmax(pred_logprobs, axis=-1)
      is_greedy = bool(jnp.all(greedy_toks == cont_toks))
      ll = float(jnp.sum(tok_logprobs))
      results.append((ll, is_greedy))
    return results

  # ---- loglikelihood_rolling (perplexity) --------------------------------------

  def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
    results: list[float] = []
    for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
      token_list = self.tok_encode(string)
      windows = [
          make_disjoint_window(w)
          for w in get_rolling_token_windows(
              token_list=token_list,
              prefix_token=self.prefix_token_id,
              max_seq_len=self._max_length - 1,
              context_len=1,
          )
      ]
      # Each disjoint window is (context, continuation); score the continuation and
      # sum across windows to get the whole-string loglikelihood.
      window_reqs = [(None, ctx, cont) for ctx, cont in windows]
      window_lls = self._loglikelihood_tokens(window_reqs, disable_tqdm=True)
      results.append(sum(ll for ll, _ in window_lls))
    return results

  # ---- generate_until (generation) ---------------------------------------------

  def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
    results: list[str] = []
    for context, gen_kwargs in tqdm([req.args for req in requests], disable=disable_tqdm):
      gen_kwargs = dict(gen_kwargs or {})
      until = gen_kwargs.get("until", []) or []
      if isinstance(until, str):
        until = [until]
      max_gen = int(gen_kwargs.get("max_gen_toks", self._max_gen_toks))
      if gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0.0):
        logger.warning("orbax_lm.generate_until only supports greedy decoding; ignoring sampling kwargs.")

      # Leave room for the generation budget within the model window.
      context_enc = self.tok_encode(context)[-(self._max_length - max_gen):]
      cur = list(context_enc)
      generated: list[int] = []
      for _ in range(max_gen):
        logits = self._forward_single(cur)                 # [len(cur), vocab]
        next_id = int(jnp.argmax(logits[len(cur) - 1]))
        if next_id == self.eot_token_id:
          break
        cur.append(next_id)
        generated.append(next_id)
        text_so_far = self.tokenizer.decode(generated)
        if any(stop and stop in text_so_far for stop in until):
          break
      text = self.tokenizer.decode(generated)
      for stop in until:
        if stop:
          text = text.split(stop)[0]
      results.append(text)
    return results

  # ---- registry hook ------------------------------------------------------------

  @classmethod
  def create_from_arg_string(cls, arg_string, additional_config=None):
    raise NotImplementedError(
        "orbax_lm builds an in-process MaxText model from a full pyconfig; construct it "
        "via maxtext.eval.orbax.orbax_eval and pass the instance to simple_evaluate(model=...)."
    )
