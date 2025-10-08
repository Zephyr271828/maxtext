#!/bin/bash
set -euo pipefail

source scripts/get_tpu_bucket_name.sh

export TPU_PREFIX="$(get_tpu_name)"
export BUCKET_NAME="$(get_bucket_name)"

export MODEL_NAME="llama3.1-1b"
export NUM_STEPS=12500
export SEQ_LEN=8192
export BATCH_SIZE=2
export GRAD_ACCUM=4
export GRAD_CLIP=1.0
export LR=0.0003
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://${BUCKET_NAME}/model_ckpts/maxtext"
export DATA_FILES="/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record"
export RUN_NAME="llama3.1-1b_L200_S50_seqlen_8192_bs_2_grad_accum_4_lr_0.0003_min_lr_ratio_0.1_warmup_ratio_0.05"
export JAX_PLATFORMS=tpu

python -u multihost_runner_orig.py \
    --TPU_PREFIX=${TPU_PREFIX} \
    --COMMAND="
    export TPU_LOG_DIR=/home/zephyr/tpu_logs
    export WANDB_API_KEY='7d11bbca76b3081b6bd1efbbcf1572aab26c5d56'
    source ~/maxtext_env/bin/activate
    python3.10 -u -m MaxText.train MaxText/configs/base.yml \
        run_name=${RUN_NAME} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        dataset_type=grain \
        grain_train_files=${DATA_FILES} \
        start_from_file_index=0 \
        grain_file_type='arrayrecord' \
        grain_worker_count=1 \
        enable_data_shuffling=False \
        tokenize_train_data=False \
        tokenize_eval_data=False \
        max_target_length=${SEQ_LEN} \
        async_checkpointing=${ASYNC_CHECKPOINTING} \
        model_name=${MODEL_NAME} \
        steps=${NUM_STEPS} \
        per_device_batch_size=${BATCH_SIZE} \
        gradient_accumulation_steps=${GRAD_ACCUM} \
        gradient_clipping_threshold=${GRAD_CLIP} \
        learning_rate=${LR} \
        cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
        warmup_steps_fraction=${WARMUP_RATIO} \
        checkpoint_period=10 \
        checkpoint_max_to_keep=1 \
        use_wandb=True \
        wandb_project=llm_pruning \
        wandb_run_name=${TPU_PREFIX}_${RUN_NAME} \
        packing=false
    "

bash scripts/convert.sh gen_param_ckpt \
    --model=${MODEL_NAME} \
    --orbax_ckpt_path=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/12499/items \
    --hf_model_path=/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B \
    --direct_run_name=direct_llama3.1-1b_steps_12500_lr_0.0003_load_none

bash scripts/convert.sh eval \
    --model=${MODEL_NAME} \
    --hf_model_path=/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B \
    --direct_run_name=direct_llama3.1-1b_steps_12500_lr_0.0003_load_none 
