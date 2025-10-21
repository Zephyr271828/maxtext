#!/bin/bash


host_id=$(python -c "import socket; print(socket.gethostname())" | awk -F'-' '{print $NF}')

step=$((host_id * 250))
orbax_ckpt_name="llama3.1-4b-width_S50_seqlen_8192_bs_2_grad_accum_4_lr_0.0003_min_lr_ratio_0.1_warmup_ratio_0.05_test"
direct_run_name="${orbax_ckpt_name}_step_${step}"
bash scripts/convert.sh gen_param_ckpt \
    --model=llama3.1-4b-width \
    --orbax_ckpt_name=$orbax_ckpt_name \
    --step=$step \
    --direct_run_name=$direct_run_name

bash scripts/convert.sh eval \
    --model=llama3.1-4b-width \
    --direct_run_name=$direct_run_name \
    --hf_model_name=Llama-3.1-8B