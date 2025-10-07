#!/bin/bash

set +x
set -euo pipefail

source scripts/get_tpu_bucket_name.sh

export BUCKET_NAME="$(get_bucket_name)"
export TPU_PREFIX="$(get_tpu_name)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mode> [--model=MODEL] [--bucket_name=BUCKET] [--lr=LR] [--hf_model_path=PATH]"
  echo "Modes: hf_to_orbax | gen_param_ckpt | orbax_to_hf | logits_test | eval"
  exit 1
fi
MODE=$1
shift
for arg in "$@"; do
  case $arg in
    --model=*) MODEL="${arg#*=}" ;;
    --orbax_ckpt_path=*) CONVERTED_CHECKPOINT="${arg#*=}" ;;
    --hf_model_path=*) HF_MODEL_PATH="${arg#*=}" ;;
    --direct_run_name=*) DIRECT_PARAMETER_CHECKPOINT_RUN="${arg#*=}" ;;
    *) echo "[WARN] Unknown arg $arg" ;;
  esac
done

### ====== CONFIG ======
export BASE_OUTPUT_DIRECTORY="gs://${BUCKET_NAME}/model_ckpts/maxtext"
export PYTHONPATH="/home/zephyr/maxtext":${PYTHONPATH:-''}
export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"

case "$MODE" in
  hf_to_orbax)
    echo "[INFO] 🚀 Converting Hugging Face → Orbax..."
    JAX_PLATFORMS=cpu python3 -m MaxText.llama_or_mistral_ckpt \
      --base-model-path ${HF_MODEL_PATH} \
      --huggingface-checkpoint True \
      --model-size $MODEL \
      --maxtext-model-path ${CONVERTED_CHECKPOINT_PATH}
    ;;

  gen_param_ckpt)
    echo "[INFO] 🧩 Generating parameter-only checkpoint..."
    JAX_PLATFORMS=cpu python3 -m MaxText.generate_param_only_checkpoint \
      MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
      checkpoint_dir=${BASE_OUTPUT_DIRECTORY} \
      base_output_directory=${BASE_OUTPUT_DIRECTORY} \
      load_parameters_path=${CONVERTED_CHECKPOINT} \
      run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
      model_name=$MODEL \
      force_unroll=true
    ;;

  orbax_to_hf)
    echo "[INFO] 🔁 Converting Orbax → Hugging Face..."
    JAX_PLATFORMS=cpu python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
      MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
      base_output_directory=${BASE_OUTPUT_DIRECTORY} \
      load_parameters_path=${CONVERTED_CHECKPOINT} \
      run_name=convert_to_hf \
      model_name=${MODEL} \
      hf_model_path=${HF_MODEL_PATH}
    ;;

  logits_test)
    echo "[INFO] 🧪 Running forward pass equivalence test..."
    python3 -u tests/test_eq.py \
      MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
      load_parameters_path=${UNSCANNED_CKPT_PATH} \
      run_name=forward_pass_test \
      per_device_batch_size=1 \
      model_name=${MODEL} \
      max_prefill_predict_length=4 \
      max_target_length=4 \
      dataset_type=synthetic \
      dtype=bfloat16 \
      scan_layers=false \
      --run_hf_model=True \
      --hf_model_path=${HF_MODEL_PATH}
    ;;

  eval)
  echo "[INFO] 🧪 Running evaluation..."
  cd lm-evaluation-harness
  python3 -u scripts/test_orbax_eval.py \
    ../MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test \
    per_device_batch_size=1 \
    model_name=${MODEL} \
    max_prefill_predict_length=4 \
    max_target_length=8192 \
    dataset_type=synthetic \
    dtype=bfloat16 \
    scan_layers=false \
    --hf_model_path=${HF_MODEL_PATH} 
  cd ..
;;

  *)
    echo "[ERROR] Unknown mode: $MODE"
    exit 1
    ;;
esac