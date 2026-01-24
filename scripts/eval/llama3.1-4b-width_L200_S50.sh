#!/bin/bash
# set -euo pipefail

source scripts/get_tpu_bucket_name.sh

export TPU_PREFIX="$(get_tpu_name)"
export BUCKET_NAME="$(get_bucket_name)"
# export NUM_HOSTS=$(get_num_hosts)
export NUM_HOSTS=32

for arg in "$@"; do
    case $arg in
        --lr=*) LR="${arg#*=}" ;;
        --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
        --global_batch_size=*) GLOBAL_BATCH_SIZE="${arg#*=}" ;;
        --grad_clip=*) GRAD_CLIP="${arg#*=}" ;;
        --min_lr_ratio=*) MIN_LR_RATIO="${arg#*=}" ;;
        --warmup_ratio=*) WARMUP_RATIO="${arg#*=}" ;;
        --max_to_keep=*) MAX_TO_KEEP="${arg#*=}" ;;
        --data_files=*) DATA_FILES="${arg#*=}" ;;
        --shuffle=*) SHUFFLE="${arg#*=}" ;;
        --tag=*) TAG="${arg#*=}" ;;
        *) echo "[WARN] Unknown arg $arg" ;;
    esac
done

export MODEL_NAME="llama3.1-4b-width"
export NUM_STEPS=12500
export SEQ_LEN=8192
export BATCH_SIZE=${BATCH_SIZE:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
export GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / BATCH_SIZE / NUM_HOSTS / 4))
export GRAD_CLIP=${GRAD_CLIP:-1.0}
export LR=${LR:-0.0003}
export MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
export WARMUP_RATIO=${WARMUP_RATIO:-0.05}
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://${BUCKET_NAME}/model_ckpts/maxtext"
export MAX_TO_KEEP=${MAX_TO_KEEP:-1}
export DATA_FILES="${DATA_FILES:-/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record}"
export SHUFFLE="${SHUFFLE:-True}"

# export RUN_NAME="${MODEL_NAME}_L200_S50_seqlen_${SEQ_LEN}_bs_${BATCH_SIZE}_grad_accum_${GRAD_ACCUM}_lr_${LR}_min_lr_ratio_${MIN_LR_RATIO}_warmup_ratio_${WARMUP_RATIO}"
# if [ ! -z "${TAG:-}" ]; then
#     export RUN_NAME="${RUN_NAME}_${TAG}"
# fi

CKPT_DIR=$(ls -d /home/zephyr/gcs-bucket/model_ckpts/maxtext/${MODEL_NAME}_L200_S50_seqlen_${SEQ_LEN}_bs_*_grad_accum_*_lr_${LR/e/*e}_min_lr_ratio_${MIN_LR_RATIO}_warmup_ratio_${WARMUP_RATIO}*/checkpoints/$(( NUM_STEPS - 1 )) )
RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")

export JAX_PLATFORMS=tpu
export SPARSE_MODEL_TRAINING=False

bash scripts/convert.sh gen_param_ckpt \
    --model=${MODEL_NAME} \
    --orbax_ckpt_name=${RUN_NAME} \
    --step=12499 \
    --hf_model_name=Llama-3.1-8B \
    --direct_run_name=${RUN_NAME}

bash scripts/convert.sh eval \
    --model=${MODEL_NAME} \
    --hf_model_name=Llama-3.1-8B \
    --direct_run_name=${RUN_NAME}
