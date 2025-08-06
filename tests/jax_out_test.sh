#!/bin/bash

export TPU_PREFIX=llm-pruning-v6e
export bucket_name=llm_pruning_us_east1_d
export MODEL_NAME='llama3.1-4b-width'
export RUN_NAME="test"
export ASYNC_CHECKPOINTING=false
export CONVERTED_CHECKPOINT="gs://$bucket_name/model_ckpts/minitron/llama3_4b_width_orbax/0/items"
export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/model_ckpts/maxtext/${RUN_NAME}"
export DATASET_PATH='/home/zephyr/gcs-bucket/datasets/'

export PYTHONPATH='/home/zephyr/gcs-bucket/maxtext':$PYTHONPATH

# python3 tests/jax_out_test.py MaxText/configs/base.yml \
#     load_parameters_path=${CONVERTED_CHECKPOINT} \
#     run_name=runner_direct_${idx} \
#     base_output_directory=${BASE_OUTPUT_DIRECTORY} \
#     per_device_batch_size=1 \
#     model_name='llama3.1-4b-width' \
#     max_prefill_predict_length=4  \
#     max_target_length=5 \
#     prompt="I love to" \
#     attention=dot_product

export DIRECT_PARAMETER_CHECKPOINT_RUN="llama3_4b_width_orbax_direct"
python3 -m MaxText.generate_param_only_checkpoint \
    MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
    model_name='llama3.1-4b-width' \
    force_unroll=true

export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items
python3 -m MaxText.tests.forward_pass_logit_checker \
    MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test per_device_batch_size=1 \
    model_name=llama3.1-4b-width \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    dtype=float32 \
    scan_layers=false \
    # --hf_model_path='/home/zephyr/gcs-bucket/model_ckpts/DeepSeek-R1-Distill-Llama-8B'