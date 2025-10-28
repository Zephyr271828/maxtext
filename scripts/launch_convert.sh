#!/bin/bash

set -euo pipefail

host_id=$(python -c "import socket; print(socket.gethostname())" | awk -F'-' '{print $NF}')
host_id=0

model=llama3.1-1b
step=$((host_id * 500 + 500))
orbax_ckpt_name="llama3.1-1b_S50_seqlen_8192_bs_2_grad_accum_4_lr_3e-4_min_lr_ratio_0.1_warmup_ratio_0.05"
direct_run_name="${orbax_ckpt_name}_step_${step}"
bash scripts/convert.sh gen_param_ckpt \
    --model=${model} \
    --orbax_ckpt_name=$orbax_ckpt_name \
    --step=$step \
    --direct_run_name=$direct_run_name

bash scripts/convert.sh eval \
    --model=${model} \
    --direct_run_name=$direct_run_name \
    --hf_model_name=Llama-3.1-8B