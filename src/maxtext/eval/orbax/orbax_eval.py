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

"""Direct-forward-pass eval driver for MaxText Orbax checkpoints (no vLLM).

Ports ``test_new``'s ``scripts/test_orbax_eval.py`` onto the ``src/maxtext/`` NNX
tree. Runs two things against an in-process MaxText model:

  * PPL   — custom perplexity loop over raw datasets (c4, wikitext, wikitext2,
            cnn_dailymail, dclm), scoring with a direct forward pass. These are
            NOT lm-eval tasks; they load datasets directly.
  * ACC   — multiple-choice / QA accuracy via lm-eval ``simple_evaluate`` on the
            ``OrbaxLM`` adapter (which also supports rolling-ppl and generation —
            see ``orbax_lm.py`` — so any standard lm-eval task works here too).

Usage (on a TPU host, after ``pip install lm_eval datasets``)::

  python -m maxtext.eval.orbax.orbax_eval \
      src/maxtext/configs/base.yml \
      model_name=llama3.1-4b-depth \
      load_parameters_path=gs://<bucket>/.../checkpoints/<step>/items \
      max_target_length=8192 \
      --hf_model_path=meta-llama/Llama-3.1-8B \
      --limit=1000

MaxText config args (``key=value``) are passed through to ``pyconfig``; the
``--…`` flags below are consumed by this driver and stripped before pyconfig sees
argv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from lm_eval import evaluator

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.eval.orbax.orbax_lm import OrbaxLM
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


PPL_TASKS = ["c4", "wikitext", "wikitext2", "cnn_dailymail", "dclm"]

ACC_TASKS = [
    {"name": "winogrande", "num_fewshot": 0, "acc_key": "acc,none"},
    {"name": "arc_easy", "num_fewshot": 0, "acc_key": "acc_norm,none"},
    {"name": "arc_challenge", "num_fewshot": 0, "acc_key": "acc_norm,none"},
    {"name": "hellaswag", "num_fewshot": 0, "acc_key": "acc_norm,none"},
    {"name": "truthfulqa_mc1", "num_fewshot": 0, "acc_key": "acc,none"},
    {"name": "truthfulqa_mc2", "num_fewshot": 0, "acc_key": "acc,none"},
    {"name": "piqa", "num_fewshot": 0, "acc_key": "acc_norm,none"},
    {"name": "sciq", "num_fewshot": 0, "acc_key": "acc,none"},
    {"name": "boolq", "num_fewshot": 0, "acc_key": "acc,none"},
    {"name": "anli_r1", "num_fewshot": 0, "acc_key": None},
    {"name": "anli_r2", "num_fewshot": 0, "acc_key": None},
    {"name": "anli_r3", "num_fewshot": 0, "acc_key": None},
    {"name": "openbookqa", "num_fewshot": 0, "acc_key": None},
    {"name": "rte", "num_fewshot": 0, "acc_key": None},
    {"name": "mmlu", "num_fewshot": 0, "acc_key": None},
    {"name": "record", "num_fewshot": 0, "acc_key": None},
]


def _dclm_default_path() -> str:
  return os.path.join(os.path.expanduser("~"), "gcs-bucket/datasets/dclm/dclm_baseline_1.0.val.jsonl")


def get_ppl_enc(task, tokenizer, add_special_tokens: bool = True, dclm_path: str | None = None):
  """Load a raw dataset for ``task`` and return one long tokenized tensor [1, N]."""
  # Use namespaced HF repo ids and drop trust_remote_code: newer `datasets`
  # (>=3) removed dataset loading scripts, so bare ids like "wikitext" +
  # trust_remote_code now raise "Repository id must be 'namespace/name'".
  if task == "wikitext":
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n\n".join(dataset[:32768]["text"])
  elif task == "wikitext2":
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset[:32768]["text"])
  elif task == "cnn_dailymail":
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    text = " ".join(dataset[:16384]["article"])
  elif task == "c4":
    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        verification_mode="no_checks",
    )
    text = " ".join(dataset[:8192]["text"])
  elif task == "dclm":
    data_path = dclm_path or _dclm_default_path()
    dataset = load_dataset("json", data_files={"train": data_path}, split="train", verification_mode="no_checks")
    text = " ".join(dataset[:8192]["text"])
  else:
    raise NotImplementedError(f"Unsupported PPL task: {task}")
  ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
  return np.asarray(ids, dtype=np.int32)[None, :]


def get_ppl(lm, tokenizer, tasks, batch_size=1, calib_size=256, max_length=8192,
            add_special_tokens=True, eval_log_path=None, dclm_path=None):
  """Perplexity via a direct forward pass over fixed-length chunks (exp of mean CE)."""
  done = {}
  if eval_log_path and os.path.exists(eval_log_path):
    with open(eval_log_path, "r", encoding="utf-8") as f:
      for line in f:
        d = json.loads(line)
        if "num_fewshot" not in d:
          done[d["alias"]] = d

  ppl_res = {}
  for task in tasks:
    if task in done:
      print(done[task])
      ppl_res[task] = done[task].get("ppl")
      continue
    testenc = get_ppl_enc(task, tokenizer, add_special_tokens=add_special_tokens, dclm_path=dclm_path)
    seq_len = max_length
    nsamples = min(testenc.size // seq_len, calib_size)
    tot_loss, tot_tokens = 0.0, 0
    for i in range(0, nsamples, batch_size):
      j = min(i + batch_size, nsamples)
      inputs = testenc[:, (i * seq_len):(j * seq_len)].reshape(j - i, seq_len)
      logits = lm.forward(inputs)                                   # [b, seq_len, vocab]
      logprobs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
      labels = jnp.asarray(inputs[:, 1:], dtype=jnp.int32)
      tok_lp = jnp.take_along_axis(logprobs, labels[..., None], axis=-1)[..., 0]
      loss = float(-jnp.mean(tok_lp))                               # mean CE over shifted tokens
      # Weight by seq_len*(j-i) to match test_new's aggregation convention.
      tot_loss += loss * seq_len * (j - i)
      tot_tokens += seq_len * (j - i)
    ppl = math.exp(tot_loss / tot_tokens)
    ppl_res[task] = ppl
    res = {"alias": task, "ppl": ppl}
    print(res)
    if task == "dclm":
      print("dclm val loss", math.log(ppl))
    if eval_log_path:
      with open(eval_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
  return ppl_res


def get_acc(lm, tasks, limit=1000000, eval_log_path=None):
  """Multiple-choice / QA accuracy via lm-eval simple_evaluate on the OrbaxLM adapter."""
  done = set()
  if eval_log_path and os.path.exists(eval_log_path):
    with open(eval_log_path, "r", encoding="utf-8") as f:
      for line in f:
        d = json.loads(line)
        if "num_fewshot" in d:
          done.add(f"{d['alias']}_fs_{d['num_fewshot']}")

  acc_res = {}
  for cfg in tasks:
    task, fewshot = cfg["name"], cfg["num_fewshot"]
    if f"{task}_fs_{fewshot}" in done:
      continue
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=fewshot,
        log_samples=False,
        confirm_run_unsafe_code=True,
        limit=limit,
    )
    res = dict(results["results"][task])
    res["alias"] = res.get("alias", task)
    res["num_fewshot"] = fewshot
    print(res)
    if eval_log_path:
      with open(eval_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    if cfg["acc_key"] is not None:
      acc_res[f"{task}_fs_{fewshot}"] = res.get(cfg["acc_key"])
  return acc_res


def main(config, test_args):
  tokenizer = AutoTokenizer.from_pretrained(test_args.hf_model_path)

  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  rng = jax.random.PRNGKey(config.init_weights_seed)
  # from_pretrained builds the NNX model AND restores/sharding-casts params from
  # config.load_parameters_path — no separate setup_decode_state needed.
  model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rng_key=rng)

  lm = OrbaxLM(model, tokenizer, config, mesh)

  model_name = config.load_parameters_path.strip("/").split("/")[-4] if config.load_parameters_path else config.model_name
  eval_log_dir = os.path.join(os.path.expanduser("~"), "gcs-bucket", "eval_logs", model_name)
  os.makedirs(eval_log_dir, exist_ok=True)
  eval_log_path = os.path.join(eval_log_dir, "results.jsonl")
  print("[INFO] eval log:", eval_log_path)

  tasks = test_args.tasks or ["ppl", "acc"]
  if "ppl" in tasks:
    ppl_res = get_ppl(
        lm, tokenizer,
        tasks=PPL_TASKS,
        batch_size=1,
        calib_size=min(256, test_args.limit),
        max_length=config.max_target_length,
        add_special_tokens=test_args.add_special_tokens,
        eval_log_path=eval_log_path,
        dclm_path=test_args.dclm_path,
    )
    print("PPL:", ppl_res)
  if "acc" in tasks:
    acc_res = get_acc(lm, ACC_TASKS, limit=test_args.limit, eval_log_path=eval_log_path)
    print("ACC:", acc_res)


def _str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  if v.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError("Boolean value expected")


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  parser = argparse.ArgumentParser()
  parser.add_argument("--hf_model_path", type=str, required=True, help="HF model id/path for the tokenizer.")
  parser.add_argument("--add_special_tokens", type=_str2bool, default=True)
  parser.add_argument("--limit", type=int, default=1000000, help="Max samples per task.")
  parser.add_argument("--dclm_path", type=str, default=None, help="Local JSONL for the dclm ppl set.")
  parser.add_argument("--tasks", type=lambda x: [] if not x else x.split(","), default=[],
                      help="Subset of {ppl,acc} to run (default: both).")
  # `remaining` is everything argparse didn't consume — i.e. the config file and
  # pyconfig key=value args. Prepend argv[0] so pyconfig sees a normal argv. This
  # is robust to both `--flag=value` and `--flag value` driver-flag forms.
  test_args, remaining = parser.parse_known_args()
  model_args = [sys.argv[0]] + remaining

  cfg = pyconfig.initialize(model_args)
  main(cfg, test_args)
