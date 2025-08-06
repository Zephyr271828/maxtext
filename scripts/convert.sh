#!/bin/bash

export META_CHECKPOINT_PATH='/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_hf'
export CONVERTED_CHECKPOINT_PATH='gs://llm_pruning_us_east1_d/model_ckpts/minitron/llama3_4b_width_orbax'

export TPU_NAME='vision-mix_tpu'
export ZONE='us-central2-b'

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/conda_envs/maxtext

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --ssh-key-file='~/.ssh/id_rsa' \
  --worker=0 \
  --command "
  # source ~/miniconda3/etc/profile.d/conda.sh
  # conda activate ~/conda_envs/maxtext

  # export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,4
  # export TPU_PROCESS_BOUNDS=1,1,1
  # export TPU_VISIBLE_DEVICES=0

  cd /home/zephyr/gcs-bucket/maxtext
  export TPU_LOG_DIR=~/tpu_logs
  python3 -m MaxText.llama_or_mistral_ckpt \
    --base-model-path ${META_CHECKPOINT_PATH} \
    --model-size llama3-4b-width \
    --maxtext-model-path ${CONVERTED_CHECKPOINT_PATH} \
    --huggingface-checkpoint True
  "