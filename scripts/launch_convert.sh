#!/bin/bash

set -euo pipefail

host_id=$(python -c "import socket; print(socket.gethostname())" | awk -F'-' '{print $NF}')

for arg in "$@"; do
  case $arg in
    --model=*) model="${arg#*=}" ;;
    --orbax_ckpt_name=*) orbax_ckpt_name="${arg#*=}" ;;
    *) echo "[WARN] Unknown arg $arg" ;;
  esac
done

model=llama3.1-1b
step=$((host_id * 500 + 500))
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