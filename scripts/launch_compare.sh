#!/bin/bash

set -euo pipefail

host_id=$(python -c "import socket; print(socket.gethostname())" | awk -F'-' '{print $NF}')
host_id=0

model=llama3.1-1b
step=${host_id}
orbax_ckpt_name="llama3.1-1b_S50_seqlen_8192_bs_1_grad_accum_1_lr_0.0003_min_lr_ratio_0.1_warmup_ratio_0.05"
direct_run_name="${orbax_ckpt_name}_step_${step}"
bash scripts/convert.sh gen_param_ckpt \
    --model=${model} \
    --orbax_ckpt_name=$orbax_ckpt_name \
    --step=$step \
    --direct_run_name=${direct_run_name}

bash scripts/convert.sh weights_test \
    --model=${model} \
    --direct_run_name=${direct_run_name} \
    --hf_model_name="llama3.1_1b_seq_8192_bs_2_grad_accum_8_steps_12500_lr_0.0003_minlr_ratio_0.1_warmup_0.05/step_${step}_ckp_hf"