#!/bin/bash

export PYTHONPATH='/home/zephyr/gcs-bucket/maxtext':$PYTHONPATH

python /home/zephyr/gcs-bucket/maxtext/MaxText/tests/hf_checkpoint_conversion_checker.py \
    --original_ckpt '/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_hf' \
    --converted_ckpt '/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_hf_2'