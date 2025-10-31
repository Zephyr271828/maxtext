model=llama3.1-1b
hf_model_name=llama3.1_1b_scratch


bash scripts/convert.sh orbax_to_hf \
    --model=$model \
    --hf_model_name=$hf_model_name \
    --orbax_ckpt_name=$hf_model_name 

bash scripts/convert.sh gen_param_ckpt \
    --model=$model \
    --hf_model_name=$hf_model_name \
    --orbax_ckpt_name=$hf_model_name \
    --direct_run_name=$hf_model_name

bash scripts/convert.sh weights_test \
    --model=$model \
    --hf_model_name=$hf_model_name \
    --orbax_ckpt_name=$hf_model_name \
    --direct_run_name=$hf_model_name

bash scripts/convert.sh logits_test \
    --model=$model \
    --hf_model_name=$hf_model_name \
    --orbax_ckpt_name=$hf_model_name \
    --direct_run_name=$hf_model_name

bash scripts/convert.sh orbax_to_hf \
    --model=$model \
    --hf_model_name=${hf_model_name}_back \
    --orbax_ckpt_name=$hf_model_name \
    --direct_run_name=$hf_model_name