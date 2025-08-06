#!/bin/bash

# CONVERTED_CHECKPOINT='/home/zephyr/gcs-bucket/model_ckpts/maxtext/150/items'
# CONVERTED_CHECKPOINT='/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_orbax/0/items'
CONVERTED_CHECKPOINT='/home/zephyr/gcs-bucket/model_ckpts/maxtext/llama3.1-4b-width_dclm_50B/llama3.1-4b-width_dclm_50B/checkpoints/0/items'

cd /home/zephyr/gcs-bucket/maxtext

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
    bash -c "
    pip install torch==2.5.0
    JAX_PLATFORMS=cpu python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
        MaxText/configs/base.yml \
        base_output_directory=/home/zephyr/gcs-bucket/model_ckpts/maxtext \
        load_parameters_path=${CONVERTED_CHECKPOINT} \
        run_name=convert_to_hf \
        model_name=llama3.1-4b-width \
        hf_model_path='/home/zephyr/gcs-bucket/model_ckpts/maxtext/llama3.1-4b-width_dclm_0_hf'
    "