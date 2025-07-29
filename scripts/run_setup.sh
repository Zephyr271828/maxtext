#!/bin/bash

export TPU_NAME=vision-mix_tpu
export ZONE=us-central2-b

# cd /home/zephyr/gcs-bucket/pruning
# gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
#   --zone=$ZONE \
#   --ssh-key-file='~/.ssh/id_rsa' \
#   --worker=all \
#   --command "
#   source ~/miniconda3/etc/profile.d/conda.sh
#   cd /home/zephyr
#   conda create -p ./conda_envs/maxtext python=3.10 -y
#   conda activate ./conda_envs/maxtext
#   git clone https://github.com/AI-Hypercomputer/maxtext.git
#   cd maxtext
#   bash setup.sh
#   "
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --ssh-key-file='~/.ssh/id_rsa' \
  --worker=all \
  --command "
  source /home/zephyr/miniconda3/etc/profile.d/conda.sh
  conda activate /home/zephyr/conda_envs/maxtext
  # pip install etils
  # conda env list
  which pip
  pip install array_record
  "
  