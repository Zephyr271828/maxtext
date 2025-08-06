#!/bin/bash

# export TPU_PREFIX=vision-mix_tpu
export RUN_NAME=llama2-7b
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="~/gcs-bucket/model_ckpts/maxtext/"
export DATASET_PATH='/home/zephyr/gcs-bucket/datasets/'

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/conda_envs/maxtext

LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

python multihost_runner.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ~/conda_envs/maxtext
    export TPU_LOG_DIR=/home/zephyr/tpu_logs

    

    # export XLA_FLAGS='--xla_force_host_platform_device_count=4'
    # export CUDA_VISIBLE_DEVICES=''
    python3.10 -m MaxText.train MaxText/configs/base.yml \
        run_name=$RUN_NAME \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        dataset_type=grain \
        grain_train_files='/home/zephyr/gcs-bucket/datasets/dclm/arrayrecord/dclm_baseline_1.0.chunk.00000.array_record' \
        grain_file_type='arrayrecord' \
        grain_worker_count=1 \
        tokenize_train_data=False \
        tokenize_eval_data=False \
        max_target_length=4096 \
        async_checkpointing=${ASYNC_CHECKPOINTING} \
        model_name=${RUN_NAME} \
        steps=1000 \
        per_device_batch_size=1 \
        learning_rate=3.e-4 \
        cosine_learning_rate_final_fraction=0.1 \
        warmup_steps_fraction=0.05 \
        packing=false
    "

# python3 -m MaxText.train \
#     MaxText/configs/base.yml \
#     run_name=runner_pretraining_${idx}\
#      base_output_directory=${BASE_OUTPUT_DIRECTORY} \
#      dataset_path=${DATASET_PATH} \
#      async_checkpointing=${ASYNC_CHECKPOINTING} \
#      per_device_batch_size=1 \
#      model_name='llama2-7b' \
#      ici_context_parallelism=4 \
#      steps=10 \
#      per_device_batch_size=1 \
#      packing=false
