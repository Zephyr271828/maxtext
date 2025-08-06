#!/bin/bash

python3 -m MaxText.train MaxText/configs/base.yml \
  run_name=quick_start \
  base_output_directory=/home/zephyr/gcs-bucket/model_ckpts/maxtext \
  dataset_type=synthetic \
  steps=10