#!/bin/bash

export TPU_NAME=vision-mix_tpu
export ZONE=us-central2-b

mkdir -p logs

LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
WORKER=0 # all

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --ssh-key-file='~/.ssh/id_rsa' \
  --worker=$WORKER \
  --command "
  ps aux | grep maxtext
  "

# cd /home/zephyr/gcs-bucket/pruning
nohup gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --ssh-key-file='~/.ssh/id_rsa' \
  --worker=$WORKER \
  --command "
  export TPU_LOG_DIR=/home/zephyr/tpu_logs

  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate ~/conda_envs/maxtext

  cd ~/maxtext
  python3 -m MaxText.train MaxText/configs/base.yml \
    run_name='quick_start' \
    base_output_directory=/home/zephyr/gcs-bucket/model_ckpts/maxtext \
    dataset_type=synthetic \
    steps=10
  " \
  > "$LOG_FILE" 2>&1 &
