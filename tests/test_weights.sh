#!/bin/bash

# export MODEL='llama3-8b'
# export MODEL='llama2-7b'
export MODEL='llama3-4b-depth'
export bucket_name=llm_pruning_us_central2_b
export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/model_ckpts/maxtext"
# export DIRECT_PARAMETER_CHECKPOINT_RUN="direct_generate_param_only_checkpoint_${MODEL}"


# export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B'
# export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-2-7b-hf'
export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/llama3-4b-depth-fms-to-hf'
# export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"
export UNSCANNED_CKPT_PATH="gs://$bucket_name/model_ckpts/maxtext/direct_generate_param_only_checkpoint_llama3-4b-depth_from_fms/checkpoints/0/items"

export PYTHONPATH='/home/zephyr/gcs-bucket/maxtext':$PYTHONPATH
python3 -u tests/test_weights.py \
    MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test per_device_batch_size=1 \
    model_name=${MODEL} \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    dtype=bfloat16 \
    scan_layers=false \
    --hf_model_path=${HF_MODEL_PATH}
