from transformers import AutoModelForCausalLM
import torch

hf_model = AutoModelForCausalLM.from_pretrained("/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_hf", torch_dtype=torch.float32)
hf_state_dict = hf_model.state_dict()

import orbax.checkpoint
from flax.training import checkpoints
from jax.experimental import multihost_utils

ckpt_path = "/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_orbax"
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
ckpt = checkpointer.restore(ckpt_path)
jax_params = ckpt['params']  # or just ckpt if unwrapped

from maxtext import checkpointing

state, _ = checkpointing.load_state_if_possible(config)
jax_params = state.params

from flax.traverse_util import flatten_dict
import torch
import numpy as np

def flatten_jax_params(params):
    flat = flatten_dict(params, sep='.')
    return {k: np.array(v) for k, v in flat.items()}

def normalize_name(name):
    # Replace known name differences between HF and Orbax
    return name.replace("model.layers.", "transformer.h.") \
               .replace("attention.wq", "self_attn.q_proj") \
               .replace("attention.wk", "self_attn.k_proj") \
               .replace("attention.wv", "self_attn.v_proj")
               
jax_flat = flatten_jax_params(jax_params)

mismatches = []
for name, torch_tensor in hf_state_dict.items():
    torch_np = torch_tensor.detach().cpu().numpy()
    jax_name = normalize_name(name)

    if jax_name not in jax_flat:
        print(f"Missing in JAX: {jax_name}")
        continue

    jax_tensor = jax_flat[jax_name]

    if not np.allclose(jax_tensor, torch_np, atol=1e-5):
        print(f"❌ Mismatch: {jax_name}")
        mismatches.append(jax_name)
    else:
        print(f"✅ Match: {jax_name}")