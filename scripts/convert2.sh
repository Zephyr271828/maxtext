export META_CHECKPOINT_PATH='/home/zephyr/gcs-bucket/model_ckpts/DeepSeek-R1-Distill-Llama-8B'
export CONVERTED_CHECKPOINT_PATH='gs://llm_pruning_us_east1_d/model_ckpts/maxtext/deepseek-r1-distill-llama3.1-8b'

export bucket_name=llm_pruning_us_east1_d
export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/model_ckpts/maxtext"
export CONVERTED_CHECKPOINT="${CONVERTED_CHECKPOINT_PATH}/0/items"

# python3 -m MaxText.llama_or_mistral_ckpt \
#     --base-model-path ${META_CHECKPOINT_PATH} \
#     --model-size llama3-8b \
#     --maxtext-model-path ${CONVERTED_CHECKPOINT_PATH} \
#     --huggingface-checkpoint True

export DIRECT_PARAMETER_CHECKPOINT_RUN="deepeseek_r1_distill_llama3.1_8b_orbax_direct"
# python3 -m MaxText.generate_param_only_checkpoint \
#     MaxText/configs/base.yml \
#     base_output_directory=${BASE_OUTPUT_DIRECTORY} \
#     load_parameters_path=${CONVERTED_CHECKPOINT} \
#     run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
#     model_name='llama3-8b' \
#     force_unroll=true

export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items
python3 -m MaxText.tests.forward_pass_logit_checker \
    MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test per_device_batch_size=1 \
    model_name=llama3-8b \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    dtype=float32 \
    scan_layers=false \
    --hf_model_path='/home/zephyr/gcs-bucket/model_ckpts/DeepSeek-R1-Distill-Llama-8B'