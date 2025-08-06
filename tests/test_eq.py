import argparse
import pathlib
import os
import gc
import re
import logging
import json
from dataclasses import dataclass

from safetensors import safe_open

import ml_dtypes

import psutil

from tqdm import tqdm

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

import torch

import jax
from jax import tree

from flax.training import train_state

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText.inference_utils import str2bool
from MaxText.utils import gcs_utils

MODEL_PARAMS_DICT = {
    "llama2-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama2-13b": {
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 40,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama2-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama3-4b-width": {
      "num_layers": 32,
      "num_heads": 32,
      "num_kv_heads": 8,
      "dims_per_head": 128,
      "vocab": 128256,
      "base_emb_dim": 3072,
      "base_mlp_dim": 9216,
    },
    "llama3-4b-depth": {
      "num_layers": 16,
      "num_heads": 32,
      "num_kv_heads": 8,
      "dims_per_head": 128,
      "vocab": 128256,
    },
    "llama3-8b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3.1-8b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3.1-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3.1-405b": {
        "num_layers": 126,
        "num_heads": 128,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3.3-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama4-17b-16e": {
        "num_layers": 48,
        "num_heads": 40,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 202048,
        "base_emb_dim": 5120,
        "num_experts": 16,
        "rope_type": "llama3.1",
        "scale_query": False,
        "interleave_moe_layer_step": 1,
        "inhomogeneous_layer_cycle_interval": 4,
        "num_layers_vit": 34,
        "num_att_head_vit": 16,
        "hidden_size_vit": 1408,
    },
    "llama4-17b-128e": {
        "num_layers": 48,
        "num_heads": 40,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 202048,
        "base_emb_dim": 5120,
        "num_experts": 128,
        "rope_type": "llama3.1",
        "scale_query": False,
        "interleave_moe_layer_step": 2,
        "inhomogeneous_layer_cycle_interval": 4,
    },
    "mistral-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32000,
        "base_emb_dim": 4096,
        "base_mlp_dim": 14336,
    },
    "mixtral-8x7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32000,
        "base_emb_dim": 4096,
        "base_mlp_dim": 14336,
        "num_experts": 8,
    },
    "mixtral-8x22b": {
        "num_layers": 56,
        "num_heads": 48,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32768,
        "base_emb_dim": 6144,
        "base_mlp_dim": 16384,
        "num_experts": 8,
    },
}

llama3_variants = {"llama3.1", "llama3.3"}

SIMULATED_CPU_DEVICES_COUNT = 16

# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def _incoming_ckpt_to_maxtext_mapping(layer_idx: int = -1, expert_idx: int = -1, model_size="") -> dict:
  """
  Maps from an incoming checkpoint (e.g. downloaded Llama checkpoint (.pth or .saftensors))
  to MaxText model weights.

  Args:
  layer_idx: The layer index of the model.
  expert_idx: The expert index of the model.
  model_size: The model size/name.

  Returns:
  A dictionary mapping from the incoming checkpoint to the MaxText model weights.
  """
  # pylint: disable=line-too-long
  base_mapping = {
      "tok_embeddings.weight": "model.embed_tokens.weight",
      "norm.weight": "model.norm.weight",
      "output.weight": "lm_head.weight",
      # MOE model
      f"layers.{layer_idx}.attention_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
      f"layers.{layer_idx}.ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
      f"layers.{layer_idx}.attention.wq.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
      f"layers.{layer_idx}.attention.wk.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
      f"layers.{layer_idx}.attention.wv.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
      f"layers.{layer_idx}.attention.wo.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
      f"layers.{layer_idx}.feed_forward.gate.weight": f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
      # MOE model: routed experts
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight",
      # MOE model: shared experts
      f"layers.{layer_idx}.feed_forward.w_in_shared_FD.weight": f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
      f"layers.{layer_idx}.feed_forward.w_out_shared_DF.weight": f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w_swiglu_FD.weight": f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
      # dense model
      f"layers.{layer_idx}.feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
      f"layers.{layer_idx}.feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
      # LoRA Adapter
      f"layers.{layer_idx}.attention.wq.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wq.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight",
      f"layers.{layer_idx}.attention.wk.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wk.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_B.weight",
      f"layers.{layer_idx}.attention.wv.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wv.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight",
      f"layers.{layer_idx}.attention.wo.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wo.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj.lora_B.weight",
  }

  if model_size.startswith("llama4"):
    base_mapping[f"layers.{layer_idx}.attention.wkq.layer_norm_weight"] = base_mapping.pop(
        f"layers.{layer_idx}.attention_norm.weight"
    )
    base_mapping[f"layers.{layer_idx}feed_forward.norm.weight"] = base_mapping.pop(f"layers.{layer_idx}.ffn_norm.weight")

    base_mapping[f"layers.{layer_idx}.feed_forward.experts.moe_w_in_eD_F"] = base_mapping.pop(
        f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight"
    )

    base_mapping[f"layers.{layer_idx}.feed_forward.experts.moe_w_out_eF_D"] = base_mapping.pop(
        f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight"
    )

    base_mapping[f"layers.{layer_idx}.feed_forward.experts.moe_w_swiglu_eD_F"] = base_mapping.pop(
        f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight"
    )
  return base_mapping


def _hf_to_maxtext_mapping(layer_idx: int = -1, expert_idx: int = -1) -> dict:
  """
  Returns a mapping from HuggingFace model weight names to MaxText model weight names.

  Args:
    layer_idx (int): Layer index.
    expert_idx (int): Expert index.

  Returns:
    dict [str, str]: Mapping from HuggingFace model weight names to MaxText model weight names.
  """
  # pylint: disable=line-too-long
  return {
      "model.embed_tokens.weight": "tok_embeddings.weight",
      "model.norm.weight": "norm.weight",
      "lm_head.weight": "output.weight",
      f"model.layers.{layer_idx}.input_layernorm.weight": f"layers.{layer_idx}.attention_norm.weight",
      f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"layers.{layer_idx}.ffn_norm.weight",
      f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"layers.{layer_idx}.attention.wq.weight",
      f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"layers.{layer_idx}.attention.wk.weight",
      f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"layers.{layer_idx}.attention.wv.weight",
      f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"layers.{layer_idx}.attention.wo.weight",
      f"model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq": f"layers.{layer_idx}.attention.rotary_emb.inv_freq",
      # MOE model
      f"model.layers.{layer_idx}.block_sparse_moe.gate.weight": f"layers.{layer_idx}.feed_forward.gate.weight",
      f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight": f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight",
      f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight": f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight",
      f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight": f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight",
      # FFN
      f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"layers.{layer_idx}.feed_forward.w1.weight",
      f"model.layers.{layer_idx}.mlp.up_proj.weight": f"layers.{layer_idx}.feed_forward.w2.weight",
      f"model.layers.{layer_idx}.mlp.down_proj.weight": f"layers.{layer_idx}.feed_forward.w3.weight",
      # llama4
      "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
      "language_model.model.norm.weight": "norm.weight",
      "language_model.lm_head.weight": "output.weight",
      f"language_model.model.layers.{layer_idx}.input_layernorm.weight": f"layers.{layer_idx}.attention_norm.weight",
      f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight": f"layers.{layer_idx}.ffn_norm.weight",
      f"language_model.model.layers.{layer_idx}.self_attn.q_proj.weight": f"layers.{layer_idx}.attention.wq.weight",
      f"language_model.model.layers.{layer_idx}.self_attn.k_proj.weight": f"layers.{layer_idx}.attention.wk.weight",
      f"language_model.model.layers.{layer_idx}.self_attn.v_proj.weight": f"layers.{layer_idx}.attention.wv.weight",
      f"language_model.model.layers.{layer_idx}.self_attn.o_proj.weight": f"layers.{layer_idx}.attention.wo.weight",
      # llama4 MoE
      f"language_model.model.layers.{layer_idx}.feed_forward.router.weight": f"layers.{layer_idx}.feed_forward.gate.weight",
      f"language_model.model.layers.{layer_idx}.feed_forward.experts.down_proj": f"layers.{layer_idx}.feed_forward.experts.down_proj",
      # NOTE: this contains up_proj and gate_proj concated together (we'll split/chunk them later)
      f"language_model.model.layers.{layer_idx}.feed_forward.experts.gate_up_proj": f"layers.{layer_idx}.feed_forward.experts.gate_up_proj",
      f"language_model.model.layers.{layer_idx}.feed_forward.shared_expert.gate_proj.weight": f"layers.{layer_idx}.feed_forward.shared_experts.gate_proj.weight",
      f"language_model.model.layers.{layer_idx}.feed_forward.shared_expert.down_proj.weight": f"layers.{layer_idx}.feed_forward.shared_experts.down_proj.weight",
      f"language_model.model.layers.{layer_idx}.feed_forward.shared_expert.up_proj.weight": f"layers.{layer_idx}.feed_forward.shared_experts.up_proj.weight",
      # llama4 FFN
      f"language_model.model.layers.{layer_idx}.feed_forward.gate_proj.weight": f"layers.{layer_idx}.feed_forward.w1.weight",
      f"language_model.model.layers.{layer_idx}.feed_forward.up_proj.weight": f"layers.{layer_idx}.feed_forward.w2.weight",
      f"language_model.model.layers.{layer_idx}.feed_forward.down_proj.weight": f"layers.{layer_idx}.feed_forward.w3.weight",
  }


@dataclass
class _NamespaceMapper:
  """A class to dynamically map Mistral/Llama weight names to HF/PT weights."""

  collection: dict
  model_size: str = ""
  delimiter: str = "."

  def __getitem__(self, key):
    if key in self.collection:
      return self.collection[key]  # original key takes precedence
    fields = key.split(self.delimiter)
    num_fields = [int(field) for field in fields if re.match(r"[0-9]+", field) is not None]
    mapping = _incoming_ckpt_to_maxtext_mapping(*num_fields, model_size=self.model_size)
    if key not in mapping:
      raise ValueError(f"Key `{key}` is missing from the original collection and from the mapping.")
    new_key = mapping[key]
    if new_key not in self.collection:
      raise ValueError(f"New key `{new_key}` mapped from `{key}` is missing from the collection.")
    return self.collection[new_key]


def permute_to_match_maxtext_rope(arr):
  """
  Permutes the input array to match the MaxText attention implementation.

  Args:
    arr (np.ndarray): Input array to permute.

  Returns:
    np.ndarray: Permutated array.
  """
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return np.concatenate((evens, odds), axis=arr.ndim - 1)

def _convert_huggingface_to_jax_weights(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
  """Convert a Huggingface Checkpoint to a dictionary of Numpy arrays representing the weights.

  Args:
    base_model_path (str): Path to the base model checkpoint.
    model_size (str): Size of the base model.
    model_params (dict): Dictionary containing model parameters.
    mem_info (psutil.Process): Process object to track memory usage.

  Returns:
    jax_weights (dict): Dictionary containing the converted weights.
  """
  base_emb_dim = model_params["base_emb_dim"]
  base_num_decoder_layers = model_params["num_layers"]
  base_num_query_heads = model_params["num_heads"]
  head_dim = model_params["dims_per_head"]
  base_num_kv_heads = model_params["num_kv_heads"]
  vocab_size = model_params["vocab"]
  num_experts = model_params["num_experts"] if "num_experts" in model_params else None

  is_llama4_model = model_size[:6] == "llama4"
  interleave_moe_layer = model_params.get("interleave_moe_layer_step")
  layer_cycle_interval = model_params.get("inhomogeneous_layer_cycle_interval")
  scale_query = model_params.get("scale_query", True)

  max_logging.log(f"Loading the base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        if is_llama4_model:
          layer = int(parts[3]) if "layers" in key else 0
          # TODO: update when mutli-modality support is added
          if "vision" in key or "multi_modal_projector" in key:
            print("WARNING: skipping vision or multi-modal key: ", key)
            continue
        else:
          layer = int(parts[2]) if "layers" in key else 0
        mapped_key = _hf_to_maxtext_mapping(layer)[key]
        chkpt_vars[mapped_key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  if is_llama4_model:
    jax_weights = {
        "decoder": {"decoder_norm": {"scale": None}, "logits_dense": {"kernel": None}, "layers": {}},
        "token_embedder": {"embedding": None},
    }
    # block 0, 1, 2, 3
    for block_idx in range(layer_cycle_interval):
      layer_name = f"layers_{block_idx}"
      jax_weights["decoder"]["layers"].update(
          {
              layer_name: {
                  "self_attention": {
                      "query": {"kernel": None},
                      "key": {"kernel": None},
                      "value": {"kernel": None},
                      "out": {"kernel": None},
                  },
                  "pre_self_attention_layer_norm": {"scale": None},
                  "post_self_attention_layer_norm": {"scale": None},
              },
          }
      )
      # scout: 0, 1, 2, 3 are moe; maverick: 0, 2 are mlp, 1, 3 are moe
      is_dense_layer = (block_idx + 1) % interleave_moe_layer != 0
      if is_dense_layer:
        # mlp_dict
        jax_weights["decoder"]["layers"][layer_name]["mlp"] = {
            "wi_0": {"kernel": None},
            "wi_1": {"kernel": None},
            "wo": {"kernel": None},
        }
      else:
        # moe_dict
        jax_weights["decoder"]["layers"][layer_name]["Llama4MoEBlock_0"] = {
            "MoeBlock_0": {
                "wi_0": None,
                "wi_1": None,
                "wo": None,
                "gate": {"kernel": None},
            },
            "shared_experts": {
                "wi_0": {"kernel": None},
                "wi_1": {"kernel": None},
                "wo": {"kernel": None},
            },
        }
  else:
    jax_weights = {
        "decoder": {
            "layers": {
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
                "self_attention": {
                    "query": {"kernel": None},
                    "key": {"kernel": None},
                    "value": {"kernel": None},
                    "out": {"kernel": None},
                },
            },
            "decoder_norm": {"scale": None},
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
    if num_experts is None:
      jax_weights["decoder"]["layers"]["mlp"] = {
          "wi_0": {"kernel": None},
          "wi_1": {"kernel": None},
          "wo": {"kernel": None},
      }
    else:
      jax_weights["decoder"]["layers"]["MoeBlock_0"] = {
          "wi_0": None,
          "wi_1": None,
          "wo": None,
          "gate": {"kernel": None},
      }

  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  decoder_norm_scale = chkpt_vars["norm.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
  jax_weights["decoder"]["decoder_norm"]["scale"] = decoder_norm_scale

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")

  jax_weights["decoder"]["logits_dense"]["kernel"] = (
      chkpt_vars["output.weight"].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()[:, :vocab_size]
  )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")

  if model_size[:6] in ["llama3", "llama4"]:
    jax_weights["token_embedder"]["embedding"] = (
        chkpt_vars["tok_embeddings.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
    )
  else:
    jax_weights["token_embedder"]["embedding"] = (
        chkpt_vars["tok_embeddings.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)[:vocab_size, :]
    )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # self attention ###############################################
  max_logging.log("Processing self attention")
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    if is_llama4_model:
      # e.g., interval=4, layer 11 is sublayer 2 in block 3
      block_layer_idx, block_idx = divmod(layer_idx, layer_cycle_interval)
      stack_shape = (base_num_decoder_layers // layer_cycle_interval,)
      self_attention = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]["self_attention"]
    else:
      block_layer_idx = layer_idx
      stack_shape = (base_num_decoder_layers,)
      self_attention = jax_weights["decoder"]["layers"]["self_attention"]

    wq = chkpt_vars[f"layers.{layer_idx}.attention.wq.weight"].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
    wk = chkpt_vars[f"layers.{layer_idx}.attention.wk.weight"].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()
    wv = chkpt_vars[f"layers.{layer_idx}.attention.wv.weight"].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()

    wq = np.reshape(wq, [base_emb_dim, base_num_query_heads, head_dim])
    wk = np.reshape(wk, [base_emb_dim, base_num_kv_heads, head_dim])
    wv = np.reshape(wv, [base_emb_dim, base_num_kv_heads, head_dim])

    if model_size[:8] == "llama3.1":
      wq = max_utils.permute_to_match_maxtext_rope(wq)
      wk = max_utils.permute_to_match_maxtext_rope(wk)

    w_post = chkpt_vars[f"layers.{layer_idx}.attention.wo.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)

    w_post = np.reshape(w_post, [base_emb_dim, base_num_query_heads, head_dim])

    if self_attention["query"]["kernel"] is None:
      self_attention["query"]["kernel"] = np.zeros(stack_shape + wq.shape, dtype=CAST_DTYPE)
      self_attention["key"]["kernel"] = np.zeros(stack_shape + wk.shape, dtype=CAST_DTYPE)
      self_attention["value"]["kernel"] = np.zeros(stack_shape + wv.shape, dtype=CAST_DTYPE)
      self_attention["out"]["kernel"] = np.zeros(stack_shape + w_post.shape, dtype=CAST_DTYPE)

    self_attention["query"]["kernel"][block_layer_idx, ...] = wq  # pylint: disable=E1137
    self_attention["key"]["kernel"][block_layer_idx, ...] = wk  # pylint: disable=E1137
    self_attention["value"]["kernel"][block_layer_idx, ...] = wv  # pylint: disable=E1137
    self_attention["out"]["kernel"][block_layer_idx, ...] = w_post  # pylint: disable=E1137

  self_attention_list = (
      [jax_weights["decoder"]["layers"]["self_attention"]]
      if not is_llama4_model
      else [
          jax_weights["decoder"]["layers"][f"layers_{block_idx}"]["self_attention"]
          for block_idx in range(layer_cycle_interval)
      ]
  )
  for self_attention in self_attention_list:
    self_attention["query"]["kernel"] = np.transpose(
        self_attention["query"]["kernel"], axes=(1, 0, 2, 3)
    )  # [embed, layer, q, head_dim]
    self_attention["key"]["kernel"] = np.transpose(
        self_attention["key"]["kernel"], axes=(1, 0, 2, 3)
    )  # [embed, layer, kv, head_dim]
    self_attention["value"]["kernel"] = np.transpose(
        self_attention["value"]["kernel"], axes=(1, 0, 2, 3)
    )  # [embed, layer, kv, head_dim]
    # layers, base_num_query_heads * head_dim, base_num_query_heads, head_dim =>
    # base_num_query_heads, layers,head_dim, base_num_query_heads * head_dim
    self_attention["out"]["kernel"] = np.transpose(
        self_attention["out"]["kernel"], axes=(2, 0, 3, 1)
    )  # [q, layer, head_dim, embed]

    # scale the query weights
    # NOTE: the np.sqrt here will silently cast to float64, so we add a manual cast to ensure the CAST_DTYPE is respected
    if scale_query:
      self_attention["query"]["kernel"] = self_attention["query"]["kernel"] / (np.sqrt(head_dim).astype(CAST_DTYPE))  # pylint: disable=E1137

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight pre and post self attention norm ################
  max_logging.log("Processing pre and post self attention norms")

  # self attention layer norm and swap the layer index
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    if is_llama4_model:
      # e.g., interval=4, layer 11 is sublayer 2 in block 3
      block_layer_idx, block_idx = divmod(layer_idx, layer_cycle_interval)
      stack_shape = (base_num_decoder_layers // layer_cycle_interval,)
      layer_weight = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]
    else:
      block_layer_idx = layer_idx
      stack_shape = (base_num_decoder_layers,)
      layer_weight = jax_weights["decoder"]["layers"]

    pre_self_attention_layernorm = (
        chkpt_vars[f"layers.{layer_idx}.attention_norm.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
    )
    post_self_attention_layernorm = (
        chkpt_vars[f"layers.{layer_idx}.ffn_norm.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
    )
    if layer_weight["pre_self_attention_layer_norm"]["scale"] is None:
      layer_weight["pre_self_attention_layer_norm"]["scale"] = np.zeros(
          stack_shape + pre_self_attention_layernorm.shape, dtype=CAST_DTYPE
      )
      layer_weight["post_self_attention_layer_norm"]["scale"] = np.zeros(
          stack_shape + post_self_attention_layernorm.shape, dtype=CAST_DTYPE
      )
    layer_weight["pre_self_attention_layer_norm"]["scale"][block_layer_idx, ...] = pre_self_attention_layernorm  # pylint: disable=E1137
    layer_weight["post_self_attention_layer_norm"]["scale"][block_layer_idx, ...] = post_self_attention_layernorm  # pylint: disable=E1137

  layer_weight_list = (
      [jax_weights["decoder"]["layers"]]
      if not is_llama4_model
      else [jax_weights["decoder"]["layers"][f"layers_{block_idx}"] for block_idx in range(layer_cycle_interval)]
  )
  for layer_weight in layer_weight_list:
    layer_weight["pre_self_attention_layer_norm"]["scale"] = np.transpose(
        layer_weight["pre_self_attention_layer_norm"]["scale"], axes=(1, 0)
    )
    layer_weight["post_self_attention_layer_norm"]["scale"] = np.transpose(
        layer_weight["post_self_attention_layer_norm"]["scale"], axes=(1, 0)
    )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weights ################################################
  max_logging.log("Processing layer weights")

  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    if is_llama4_model:
      # e.g., interval=4, layer 11 is sublayer 2 in block 3
      block_layer_idx, block_idx = divmod(layer_idx, layer_cycle_interval)
      stack_shape = (base_num_decoder_layers // layer_cycle_interval,)
      layer_weight = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]
      is_dense_layer = (layer_idx + 1) % interleave_moe_layer != 0
    else:
      block_layer_idx = layer_idx
      stack_shape = (base_num_decoder_layers,)
      is_dense_layer = num_experts is None
      layer_weight = jax_weights["decoder"]["layers"]
      stack_shape = (base_num_decoder_layers,)

    if is_dense_layer:
      wi_0 = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.w1.weight"].type(torch.float32).numpy().astype(CAST_DTYPE).transpose()
      )
      wi_1 = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.w2.weight"].type(torch.float32).numpy().astype(CAST_DTYPE).transpose()
      )
      wo = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.w3.weight"].type(torch.float32).numpy().astype(CAST_DTYPE).transpose()
      )
      if layer_weight["mlp"]["wi_0"]["kernel"] is None:
        layer_weight["mlp"]["wi_0"]["kernel"] = np.zeros(stack_shape + wi_0.shape, dtype=CAST_DTYPE)
        layer_weight["mlp"]["wi_1"]["kernel"] = np.zeros(stack_shape + wi_1.shape, dtype=CAST_DTYPE)
        layer_weight["mlp"]["wo"]["kernel"] = np.zeros(stack_shape + wo.shape, dtype=CAST_DTYPE)
      layer_weight["mlp"]["wi_0"]["kernel"][block_layer_idx, ...] = wi_0  # pytype: disable=unsupported-operands
      layer_weight["mlp"]["wi_1"]["kernel"][block_layer_idx, ...] = wi_1  # pytype: disable=unsupported-operands
      layer_weight["mlp"]["wo"]["kernel"][block_layer_idx, ...] = wo  # pytype: disable=unsupported-operands
    elif is_llama4_model:
      # no need to gather for llama4 safetensors
      # 1 gate: Llama4MoEBlock_0.MoeBlock_0.gate.kernel
      gate = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.gate.weight"]
          .type(torch.float32)
          .numpy()
          .astype(CAST_DTYPE)
          .transpose()
      )
      if layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["gate"]["kernel"] is None:
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["gate"]["kernel"] = np.zeros(
            stack_shape + gate.shape, dtype=CAST_DTYPE
        )
      layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["gate"]["kernel"][block_layer_idx, ...] = gate

      # 2 routed experts: Llama4MoEBlock_0.MoeBlock_0.wi_0, Llama4MoEBlock_0.MoeBlock_0.wi_1, Llama4MoEBlock_0.MoeBlock_0.wo
      wi_0_1 = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj"].type(torch.float32).numpy().astype(CAST_DTYPE)
      )
      # pylint: disable=unbalanced-tuple-unpacking
      wi_0, wi_1 = np.split(wi_0_1, 2, axis=-1)
      del wi_0_1

      wo = chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.down_proj"].type(torch.float32).numpy().astype(CAST_DTYPE)

      if layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_0"] is None:
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=CAST_DTYPE)
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=CAST_DTYPE)
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wo"] = np.zeros(stack_shape + wo.shape, dtype=CAST_DTYPE)

      layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_0"][block_layer_idx, ...] = wi_0
      layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_1"][block_layer_idx, ...] = wi_1
      layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wo"][block_layer_idx, ...] = wo

      # 3 shared experts: Llama4MoEBlock_0.shared_experts.wi_0.kernel,
      # Llama4MoEBlock_0.shared_experts.wi_1.kernel, Llama4MoEBlock_0.shared_experts.wo.kernel
      wi_0 = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.shared_experts.gate_proj.weight"]
          .type(torch.float32)
          .numpy()
          .astype(CAST_DTYPE)
          .transpose()
      )

      wi_1 = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.shared_experts.up_proj.weight"]
          .type(torch.float32)
          .numpy()
          .astype(CAST_DTYPE)
          .transpose()
      )

      wo = (
          chkpt_vars[f"layers.{layer_idx}.feed_forward.shared_experts.down_proj.weight"]
          .type(torch.float32)
          .numpy()
          .astype(CAST_DTYPE)
          .transpose()
      )

      if layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_0"]["kernel"] is None:
        layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_0"]["kernel"] = np.zeros(
            stack_shape + wi_0.shape, dtype=CAST_DTYPE
        )
        layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_1"]["kernel"] = np.zeros(
            stack_shape + wi_1.shape, dtype=CAST_DTYPE
        )
        layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wo"]["kernel"] = np.zeros(
            stack_shape + wo.shape, dtype=CAST_DTYPE
        )
      layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_0"]["kernel"][block_layer_idx, ...] = wi_0
      layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_1"]["kernel"][block_layer_idx, ...] = wi_1
      layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wo"]["kernel"][block_layer_idx, ...] = wo
    else:
      # 1 MoeBlock_0.gate.kernel
      gate = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.gate.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=0,
      ).transpose()
      if layer_weight["MoeBlock_0"]["gate"]["kernel"] is None:
        layer_weight["MoeBlock_0"]["gate"]["kernel"] = np.zeros(stack_shape + gate.shape, dtype=CAST_DTYPE)
      layer_weight["MoeBlock_0"]["gate"]["kernel"][layer_idx, ...] = gate

      # 2 MoeBlock_0.wi_0, MoeBlock_0.wi_1, MoeBlock_0.wo
      for k in tqdm(range(num_experts), desc="experts", leave=False):
        wi_0 = (
            chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.{k}.w1.weight"]
            .type(torch.float32)
            .numpy()
            .astype(CAST_DTYPE)
            .transpose()
        )
        wi_1 = (
            chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.{k}.w3.weight"]
            .type(torch.float32)
            .numpy()
            .astype(CAST_DTYPE)
            .transpose()
        )
        wo = (
            chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.{k}.w2.weight"]
            .type(torch.float32)
            .numpy()
            .astype(CAST_DTYPE)
            .transpose()
        )

        if layer_weight["MoeBlock_0"]["wi_0"] is None:
          stack_shape_expert = (num_experts, base_num_decoder_layers)
          layer_weight["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape_expert + wi_0.shape, dtype=CAST_DTYPE)
          layer_weight["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape_expert + wi_1.shape, dtype=CAST_DTYPE)
          layer_weight["MoeBlock_0"]["wo"] = np.zeros(stack_shape_expert + wo.shape, dtype=CAST_DTYPE)
        ei, li = k, layer_idx
        layer_weight["MoeBlock_0"]["wi_0"][ei, li, ...] = wi_0
        layer_weight["MoeBlock_0"]["wi_1"][ei, li, ...] = wi_1
        layer_weight["MoeBlock_0"]["wo"][ei, li, ...] = wo
      gc.collect()

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  if not is_llama4_model:
    if not num_experts:
      # swap the layer index
      layer_weight["mlp"]["wi_0"]["kernel"] = np.transpose(layer_weight["mlp"]["wi_0"]["kernel"], axes=(1, 0, 2))
      layer_weight["mlp"]["wi_1"]["kernel"] = np.transpose(layer_weight["mlp"]["wi_1"]["kernel"], axes=(1, 0, 2))
      layer_weight["mlp"]["wo"]["kernel"] = np.transpose(layer_weight["mlp"]["wo"]["kernel"], axes=(1, 0, 2))
    else:
      # no need to transpose for wi_0, wi_1, wo in MoeBlock_0
      layer_weight["MoeBlock_0"]["gate"]["kernel"] = np.transpose(
          layer_weight["MoeBlock_0"]["gate"]["kernel"], axes=(1, 0, 2)
      )
  else:
    for block_idx in range(layer_cycle_interval):
      layer_weight = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]
      is_dense_layer = (block_idx + 1) % interleave_moe_layer != 0
      if is_dense_layer:
        layer_weight["mlp"]["wi_0"]["kernel"] = np.transpose(layer_weight["mlp"]["wi_0"]["kernel"], axes=(1, 0, 2))
        layer_weight["mlp"]["wi_1"]["kernel"] = np.transpose(layer_weight["mlp"]["wi_1"]["kernel"], axes=(1, 0, 2))
        layer_weight["mlp"]["wo"]["kernel"] = np.transpose(layer_weight["mlp"]["wo"]["kernel"], axes=(1, 0, 2))
      else:
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["gate"]["kernel"] = np.transpose(
            layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["gate"]["kernel"], axes=(1, 0, 2)
        )
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_0"] = np.transpose(
            layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_0"], axes=(1, 0, 2, 3)
        )
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_1"] = np.transpose(
            layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wi_1"], axes=(1, 0, 2, 3)
        )
        layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wo"] = np.transpose(
            layer_weight["Llama4MoEBlock_0"]["MoeBlock_0"]["wo"], axes=(1, 0, 2, 3)
        )
        layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_0"]["kernel"] = np.transpose(
            layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_0"]["kernel"], axes=(1, 0, 2)
        )
        layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_1"]["kernel"] = np.transpose(
            layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wi_1"]["kernel"], axes=(1, 0, 2)
        )
        layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wo"]["kernel"] = np.transpose(
            layer_weight["Llama4MoEBlock_0"]["shared_experts"]["wo"]["kernel"], axes=(1, 0, 2)
        )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights

def convert_to_jax_weights(base_model_path: str, model_size: str, huggingface_ckpt: bool):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText

  Attributes:
    base_model_path: checkpoint path
    model_size: llama2-7b to 70b, mistral-7b, or mixtral-8x7b, mixtral-8x22b
  """
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  max_logging.log(f"Loading the base model from {base_model_path}")

  if huggingface_ckpt:
    return _convert_huggingface_to_jax_weights(base_model_path, model_size, model_params, mem_info)

  raise NotImplementedError


def save_weights_to_checkpoint(
    maxtext_model_path: str, jax_weights: dict, device_count: int, use_ocdbt: bool, use_zarr3: bool
):
  """
  Function to save jax_weights ready for MaxText to a parameters checkpoint.

  Args:
      maxtext_model_path: Path to save the MaxText checkpoint.
      jax_weights: The JAX model weights to be saved.
      device_count: The number of simulated devices.
      use_ocdbt: Whether to use Optimized Checkpoint Database with Transactions.
      use_zarr3: Whether to use Zarr3 or not.
  """
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  gc.collect()
  mesh = jax.sharding.Mesh(jax.devices(), "checkpoint_sharding_axis")
  s1 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("checkpoint_sharding_axis"))  # shards first axis
  s2 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "checkpoint_sharding_axis"))  # shards second axis
  s3 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))  # no sharding

  def checkpoint_device_put(arr):
    if arr.shape[0] % device_count == 0:
      max_logging.log("sharding first axis")
      return jax.device_put(arr, device=s1)
    elif len(arr.shape) > 1 and arr.shape[1] % device_count == 0:
      max_logging.log("sharding second axis")
      return jax.device_put(arr, device=s2)
    else:
      max_logging.log("no sharding was possible, replicating")
      return jax.device_put(arr, device=s3)

  # convert all weights to jax.numpy with sharding if applicable
  jax_weights_flat, jax_weights_struct = tree.flatten(jax_weights)
  jax_weights_new = []
  while len(jax_weights_flat) > 0:
    jax_weight = jax_weights_flat.pop(0)
    jax_weights_new.append(checkpoint_device_put(jax_weight))
    del jax_weight
    gc.collect()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  jax_weights = tree.unflatten(jax_weights_struct, jax_weights_new)

  # dummy configs for the checkpoint_manager
  step_number_to_save_new_ckpt = 0
  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      maxtext_model_path,
      enable_checkpointing,
      async_checkpointing,
      save_interval_steps,
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
  )

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
#   if checkpoint_manager is not None:
#     if checkpointing.save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
#       max_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
#     # Upon preemption, exit when and only when all ongoing saves are complete.
#     checkpoint_manager.wait_until_finished()

# from jax.typing import PyTree
from jax import tree_util
def compare_hf_and_orbax(hf_params, orbax_params, rtol=1e-5, atol=1e-6):
    hf_flat, hf_struct = tree_util.tree_flatten(hf_params)
    orbax_flat, orbax_struct = tree_util.tree_flatten(orbax_params)
    
    import pdb; pdb.set_trace()

    assert hf_struct == orbax_struct, "❌ Structure mismatch between HF and Orbax parameters."

    all_match = True
    for i, (h, o) in enumerate(zip(hf_flat, orbax_flat)):
        h_arr = np.asarray(h)
        o_arr = np.asarray(o)

        if not np.allclose(h_arr, o_arr, rtol=rtol, atol=atol):
            print(f"[Mismatch #{i}] Shape: {h_arr.shape}, Dtype: {h_arr.dtype}")
            print(f"  HuggingFace sample: {h_arr.flatten()[0:3]}")
            print(f"  Orbax       sample: {o_arr.flatten()[0:3]}")
            all_match = False

    if all_match:
        print("✅ All parameters match between HF and Orbax.")
    else:
        print("❌ Some parameters differ.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--maxtext-model-path", type=str, required=True)
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--lora-input-adapters-path", type=str, required=False)
    parser.add_argument("--huggingface-checkpoint", type=str2bool, required=False, default=False)
    parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
    parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
    args = parser.parse_args()

    if args.model_size not in MODEL_PARAMS_DICT:
        raise NotImplementedError

    if args.model_size.startswith("llama4") and not args.huggingface_checkpoint:
        raise NotImplementedError("Currently, llama4 scanned checkpoint conversion only supports huggingface safetensor.")

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"
    base_weights_path = args.maxtext_model_path

    if args.lora_input_adapters_path:
        base_weights_path += "/base"

    hf_weights =  MODEL_PARAMS_DICT[args.model_size]
    jax_weights = convert_to_jax_weights(args.base_model_path, args.model_size, args.huggingface_checkpoint)

    # print("Comparing weights between Hugging Face and Orbax checkpoint...")
    # compare_hf_and_orbax(hf_weights, jax_weights)

    TEXT = "The quick brown fox jumps over the lazy dog."
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import jax.numpy as jnp
    hf_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    hf_input = hf_tokenizer(TEXT, return_tensors="pt")
    hf_input_ids = hf_input["input_ids"]
    
    jax_input_ids = jnp.array(hf_input_ids.numpy(), dtype=jnp.int32)
    
    with torch.no_grad():
        hf_logits = hf_model(input_ids=hf_input_ids).logits
        hf_logits_np = hf_logits.cpu().numpy()    

    from MaxText.layers.models import Transformer
    from MaxText.train_utils import create_model
    # from MaxText.configs.load_config import load_config
    from jax.sharding import Mesh
    import yaml
    # config = load_config(f"/home/zephyr/gcs-bucket/maxtext/MaxText/configs/models/{args.model_size}.yml")
    with open(f"/home/zephyr/gcs-bucket/maxtext/MaxText/configs/models/llama3.1-4b-width.yml", "r") as f:
        config = yaml.safe_load(f)
    devices_array = maxtext_utils.create_device_mesh(config, devices)
    mesh = Mesh(devices_array, config.mesh_axes)
    model = Transformer(config)

    with partitioning.axis_rules([]):
        fwd_output = model.apply({"params": {"params": jax_weights}}, jax_input_ids, deterministic=True)
    
    import pdb; pdb.set_trace()    
    
    def compare_logits(a, b, tol=1e-4):
        abs_diff = np.abs(a - b)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        print(f"Max diff: {max_diff:.6f}")
        print(f"Mean diff: {mean_diff:.6f}")

        if max_diff < tol:
            print("✅ Logits match within tolerance.")
        else:
            print("❌ Logits mismatch.")

    compare_logits(hf_logits_np, np.array(fwd_output))


#   save_weights_to_checkpoint(
#       args.maxtext_model_path,
#       convert_to_jax_weights(args.base_model_path, args.model_size, args.huggingface_checkpoint),
#       SIMULATED_CPU_DEVICES_COUNT,
#       args.use_ocdbt,
#       args.use_zarr3,
#   )
#   max_logging.log(f"Successfully saved base_weights to {base_weights_path}.")

  