#!/bin/bash

# export TPU_PREFIX=llm-pruning-v6e
required_vars=(
    "BUCKET_NAME"
    "TPU_PREFIX"
)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done

export MODEL_NAME='llama3.1-8b'
export RUN_NAME="${MODEL_NAME}_dclm_L200"
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/model_ckpts/maxtext"
export DATASET_PATH='/home/zephyr/gcs-bucket/datasets/'
STEPS=50000

# source ~/gcs-bucket/miniconda3/etc/profile.d/conda.sh
# conda activate ~/gcs-bucket/conda_envs/maxtext

LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"


python -u multihost_runner.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/zephyr/tpu_logs
    sudo docker run \
        --privileged \
        --network=host \
        -v /home/zephyr:/home/zephyr \
        -v /home/zephyr/.config/gcloud:/root/.config/gcloud \
        -v /dev:/dev \
        -v /run:/run \
        -w /home/zephyr/maxtext \
        -e PYTHONPATH=/home/zephyr/maxtext \
        yx3038/maxtext_base_image:latest \
        bash -c \"
        pip show jax
        pip show libtpu
        export PYTHONPATH=/home/zephyr/maxtext:\$PYTHONPATH
        python3.10 -m MaxText.train MaxText/configs/base.yml \
            run_name=$RUN_NAME \
            base_output_directory=${BASE_OUTPUT_DIRECTORY} \
            dataset_type=grain \
            grain_train_files='/home/zephyr/gcs-bucket/datasets/dclm/llama3_256_arrayrecord/*.array_record' \
            grain_file_type='arrayrecord' \
            grain_worker_count=1 \
            tokenize_train_data=False \
            tokenize_eval_data=False \
            max_target_length=8192 \
            async_checkpointing=${ASYNC_CHECKPOINTING} \
            model_name=${MODEL_NAME} \
            steps=${STEPS} \
            per_device_batch_size=4 \
            gradient_accumulation_steps=1 \
            learning_rate=3.e-4 \
            cosine_learning_rate_final_fraction=0.1 \
            warmup_steps_fraction=0.05 \
            checkpoint_period=250 \
            packing=false
        \"
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
