# model=llama3.1-1b
# hf_model_name=llama3.1_1b_scratch

model=llama3.1-4b-depth
hf_model_name=Llama-3.1-8B_minitron_depth_nlayers_16_fp16

# model=llama3.1-4b-width
# hf_model_name=Llama-3.1-8B_minitron_width_hiddensize_3072_mlpsize_9216_fp16


# bash scripts/convert.sh hf_to_orbax \
#     --model=$model \
#     --hf_model_name=$hf_model_name \
#     --orbax_ckpt_name=$hf_model_name 

# bash scripts/convert.sh gen_param_ckpt \
#     --model=$model \
#     --hf_model_name=$hf_model_name \
#     --orbax_ckpt_name=$hf_model_name \
#     --direct_run_name=$hf_model_name


# bash scripts/convert.sh orbax_to_hf \
#     --model=$model \
#     --hf_model_name=${hf_model_name}_back \
#     --orbax_ckpt_name=$hf_model_name \
#     --direct_run_name=$hf_model_name


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

# bash scripts/convert.sh weights_test \
#     --model=$model \
#     --hf_model_name=$hf_model_name \
#     --orbax_ckpt_name=$hf_model_name \
#     --direct_run_name=$hf_model_name
