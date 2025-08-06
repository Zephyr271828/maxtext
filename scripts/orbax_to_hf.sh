#!/bin/bash

CONVERTED_CHECKPOINT='gs://llm_pruning_us_east1_d/model_ckpts/maxtext/llama3_4b_width_orbax/0/items'

JAX_PLATFORMS=cpu python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
    MaxText/configs/base.yml \
    base_output_directory=gs://llm_pruning_us_central2_b/model_ckpts/maxtext \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=convert_to_hf \
    model_name=llama3.1-4b-width \
    hf_model_path=/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_hf_2
