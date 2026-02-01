import os
from textwrap import dedent

def generate_training_script(
    model_name: str,
    num_steps: int = 12500,
    seq_len: int = 8192,
    async_checkpointing: bool = False,
    data_files: str = "/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record",
    load_parameters_path: str = "",
    output_path: str = None,
    sparse_model_training: bool = False,
    pretrain_tokens = None,
    # start_from_file_index: int = 0,
):
    """Generate a TPU MaxText training shell script from template."""

    # Dynamically include load_parameters_path only if provided
    load_path_line = f"load_parameters_path=gs://${{BUCKET_NAME}}/{load_parameters_path} \\\n            " if load_parameters_path else ""

    load_part = load_parameters_path.replace("/", "_") if load_parameters_path else "none"
    
    # handle experiment type
    exp_type = "unknown"
    start_from_file_index = 0
    if not load_parameters_path:
        # if we are using a small model
        if any(x in model_name.lower() for x in ["4b", "3b", "2b", "1.5b", "1b", "440m", "1.3b", "2.7b"]):
            exp_type = f"S{num_steps // 250}"
        elif any(x in model_name.lower() for x in ["8b", "7b"]):
            exp_type = f"L{num_steps // 250}"
        start_from_file_index = 0
    else:
        exp_type = f"S{num_steps // 250}"
        if pretrain_tokens is not None:
            exp_type = f"{pretrain_tokens}_{exp_type}"
            start_from_file_index = 50
        sparsities = ["unstructured", "4:8", "2:4"]
        if any(x in load_parameters_path for x in sparsities):
            sparsity = [s for s in sparsities if s in load_parameters_path][0]
            exp_type = f"{sparsity}_{exp_type}"
            if "sparsegpt" in load_parameters_path:
                exp_type = f"sparsegpt_{exp_type}"
            else:
                exp_type = f"wanda_{exp_type}"

    job_name = f"{model_name}/{exp_type}"

    script = dedent(f"""\
    #!/bin/bash
    set -euo pipefail
    
    source scripts/get_tpu_bucket_name.sh

    export TPU_PREFIX="$(get_tpu_name)"
    export BUCKET_NAME="$(get_bucket_name)"
    export NUM_HOSTS=$(get_num_hosts)
    
    for arg in "$@"; do
        case $arg in
            --lr=*) LR="${{arg#*=}}" ;;
            --batch_size=*) BATCH_SIZE="${{arg#*=}}" ;;
            --global_batch_size=*) GLOBAL_BATCH_SIZE="${{arg#*=}}" ;;
            --grad_clip=*) GRAD_CLIP="${{arg#*=}}" ;;
            --min_lr_ratio=*) MIN_LR_RATIO="${{arg#*=}}" ;;
            --warmup_ratio=*) WARMUP_RATIO="${{arg#*=}}" ;;
            --max_to_keep=*) MAX_TO_KEEP="${{arg#*=}}" ;;
            --data_files=*) DATA_FILES="${{arg#*=}}" ;;
            --shuffle=*) SHUFFLE="${{arg#*=}}" ;;
            --tag=*) TAG="${{arg#*=}}" ;;
            *) echo "[WARN] Unknown arg $arg" ;;
        esac
    done

    export MODEL_NAME="{model_name}"
    export NUM_STEPS={num_steps}
    export SEQ_LEN={seq_len}
    export BATCH_SIZE=${{BATCH_SIZE:-2}}
    export GLOBAL_BATCH_SIZE=${{GLOBAL_BATCH_SIZE:-512}}
    export GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / BATCH_SIZE / NUM_HOSTS / 4))
    export GRAD_CLIP=${{GRAD_CLIP:-1.0}}
    export LR=${{LR:-0.0003}}
    export MIN_LR_RATIO=${{MIN_LR_RATIO:-0.1}}
    export WARMUP_RATIO=${{WARMUP_RATIO:-0.05}}
    export ASYNC_CHECKPOINTING={str(async_checkpointing).lower()}
    export BASE_OUTPUT_DIRECTORY="gs://${{BUCKET_NAME}}/model_ckpts/maxtext"
    export MAX_TO_KEEP=${{MAX_TO_KEEP:-1}}
    export DATA_FILES="${{DATA_FILES:-{data_files}}}"
    export SHUFFLE="${{SHUFFLE:-True}}"
    export RUN_NAME="${{MODEL_NAME}}_{exp_type}_seqlen_${{SEQ_LEN}}_bs_${{BATCH_SIZE}}_grad_accum_${{GRAD_ACCUM}}_lr_${{LR}}_min_lr_ratio_${{MIN_LR_RATIO}}_warmup_ratio_${{WARMUP_RATIO}}"
    if [ ! -z "${{TAG:-}}" ]; then
        export RUN_NAME="${{RUN_NAME}}_${{TAG}}"
    fi
    export JAX_PLATFORMS=tpu
    export SPARSE_MODEL_TRAINING={sparse_model_training}

    python -u multihost_runner_orig.py \\
        --TPU_PREFIX=${{TPU_PREFIX}} \\
        --COMMAND="
        export TPU_LOG_DIR=/home/zephyr/tpu_logs
        export WANDB_API_KEY='7d11bbca76b3081b6bd1efbbcf1572aab26c5d56'
        source ~/maxtext_env/bin/activate
        ~/maxtext_env/bin/python -u -m MaxText.train MaxText/configs/base.yml \\
            run_name=${{RUN_NAME}} \\
            {load_path_line}base_output_directory=${{BASE_OUTPUT_DIRECTORY}} \\
            dataset_type=grain \\
            grain_train_files=${{DATA_FILES}} \\
            start_from_file_index={start_from_file_index} \\
            grain_file_type='arrayrecord' \\
            grain_worker_count=1 \\
            enable_data_shuffling=${{SHUFFLE}} \\
            tokenize_train_data=False \\
            tokenize_eval_data=False \\
            max_target_length=${{SEQ_LEN}} \\
            async_checkpointing=${{ASYNC_CHECKPOINTING}} \\
            model_name=${{MODEL_NAME}} \\
            steps=${{NUM_STEPS}} \\
            per_device_batch_size=${{BATCH_SIZE}} \\
            gradient_accumulation_steps=${{GRAD_ACCUM}} \\
            gradient_clipping_threshold=${{GRAD_CLIP}} \\
            learning_rate=${{LR}} \\
            cosine_learning_rate_final_fraction=${{MIN_LR_RATIO}} \\
            warmup_steps_fraction=${{WARMUP_RATIO}} \\
            checkpoint_period=500 \\
            checkpoint_max_to_keep=${{MAX_TO_KEEP}} \\
            use_wandb=True \\
            wandb_project=llm_pruning \\
            wandb_run_name=${{TPU_PREFIX}}_${{RUN_NAME}} \\
            packing=false \\
            sparse_model_training=${{SPARSE_MODEL_TRAINING}} \\
        "
    
    bash scripts/convert.sh gen_param_ckpt \\
        --model=${{MODEL_NAME}} \\
        --orbax_ckpt_name=${{RUN_NAME}} \\
        --step={num_steps-1} \\
        --hf_model_name=Llama-3.1-8B \\
        --direct_run_name=${{RUN_NAME}}
        
    # bash scripts/convert.sh eval \\
    #     --model=${{MODEL_NAME}} \\
    #     --hf_model_name=Llama-3.1-8B \\
    #     --direct_run_name=${{RUN_NAME}}
    """)

    # Default script name if not provided
    if output_path is None:
        output_path = f"scripts/{job_name}.sh"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)

    print(f"✅ Generated script at {output_path}")


def generate_eval_script(
    model_name: str,
    num_steps: int = 12500,
    seq_len: int = 8192,
    async_checkpointing: bool = False,
    data_files: str = "/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record",
    load_parameters_path: str = "",
    output_path: str = None,
    sparse_model_training: bool = False,
    pretrain_tokens = None,
    # start_from_file_index: int = 0,
):
    """Generate a TPU MaxText training shell script from template."""

    # Dynamically include load_parameters_path only if provided
    load_path_line = f"load_parameters_path=gs://${{BUCKET_NAME}}/{load_parameters_path} \\\n            " if load_parameters_path else ""

    load_part = load_parameters_path.replace("/", "_") if load_parameters_path else "none"
    
    # handle experiment type
    exp_type = "unknown"
    start_from_file_index = 0
    if not load_parameters_path:
        # if we are using a small model
        if any(x in model_name.lower() for x in ["4b", "3b", "2b", "1.5b", "1b", "440m", "1.3b", "2.7b"]):
            exp_type = f"S{num_steps // 250}"
        elif any(x in model_name.lower() for x in ["8b", "7b"]):
            exp_type = f"L{num_steps // 250}"
        start_from_file_index = 0
    else:
        exp_type = f"S{num_steps // 250}"
        if pretrain_tokens is not None:
            exp_type = f"{pretrain_tokens}_{exp_type}"
            start_from_file_index = 50
        sparsities = ["unstructured", "4:8", "2:4"]
        if any(x in load_parameters_path for x in sparsities):
            sparsity = [s for s in sparsities if s in load_parameters_path][0]
            exp_type = f"{sparsity}_{exp_type}"
            if "sparsegpt" in load_parameters_path:
                exp_type = f"sparsegpt_{exp_type}"
            else:
                exp_type = f"wanda_{exp_type}"

    job_name = f"{model_name}/{exp_type}"

    script = dedent(f"""\
    #!/bin/bash
    # set -euo pipefail
    
    source scripts/get_tpu_bucket_name.sh

    export TPU_PREFIX="$(get_tpu_name)"
    export BUCKET_NAME="$(get_bucket_name)"
    export NUM_HOSTS=$(get_num_hosts)
    # export NUM_HOSTS=32
    
    for arg in "$@"; do
        case $arg in
            --lr=*) LR="${{arg#*=}}" ;;
            --batch_size=*) BATCH_SIZE="${{arg#*=}}" ;;
            --global_batch_size=*) GLOBAL_BATCH_SIZE="${{arg#*=}}" ;;
            --grad_clip=*) GRAD_CLIP="${{arg#*=}}" ;;
            --min_lr_ratio=*) MIN_LR_RATIO="${{arg#*=}}" ;;
            --warmup_ratio=*) WARMUP_RATIO="${{arg#*=}}" ;;
            --max_to_keep=*) MAX_TO_KEEP="${{arg#*=}}" ;;
            --data_files=*) DATA_FILES="${{arg#*=}}" ;;
            --shuffle=*) SHUFFLE="${{arg#*=}}" ;;
            --tag=*) TAG="${{arg#*=}}" ;;
            *) echo "[WARN] Unknown arg $arg" ;;
        esac
    done

    export MODEL_NAME="{model_name}"
    export NUM_STEPS={num_steps}
    export SEQ_LEN={seq_len}
    export BATCH_SIZE=${{BATCH_SIZE:-2}}
    export GLOBAL_BATCH_SIZE=${{GLOBAL_BATCH_SIZE:-512}}
    export GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / BATCH_SIZE / NUM_HOSTS / 4))
    export GRAD_CLIP=${{GRAD_CLIP:-1.0}}
    export LR=${{LR:-0.0003}}
    export MIN_LR_RATIO=${{MIN_LR_RATIO:-0.1}}
    export WARMUP_RATIO=${{WARMUP_RATIO:-0.05}}
    export ASYNC_CHECKPOINTING={str(async_checkpointing).lower()}
    export BASE_OUTPUT_DIRECTORY="gs://${{BUCKET_NAME}}/model_ckpts/maxtext"
    export MAX_TO_KEEP=${{MAX_TO_KEEP:-1}}
    export DATA_FILES="${{DATA_FILES:-{data_files}}}"
    export SHUFFLE="${{SHUFFLE:-True}}"
    
    # export RUN_NAME="${{MODEL_NAME}}_{exp_type}_seqlen_${{SEQ_LEN}}_bs_${{BATCH_SIZE}}_grad_accum_${{GRAD_ACCUM}}_lr_${{LR}}_min_lr_ratio_${{MIN_LR_RATIO}}_warmup_ratio_${{WARMUP_RATIO}}"
    # if [ ! -z "${{TAG:-}}" ]; then
    #     export RUN_NAME="${{RUN_NAME}}_${{TAG}}"
    # fi
    
    CKPT_DIR=$(gsutil ls -d gs://${{BUCKET_NAME}}/model_ckpts/maxtext/${{MODEL_NAME}}_{exp_type}_seqlen_${{SEQ_LEN}}_bs_*_grad_accum_*_lr_${{LR/e/*e}}_min_lr_ratio_${{MIN_LR_RATIO}}_warmup_ratio_${{WARMUP_RATIO}}*/checkpoints/$(( NUM_STEPS - 1 )) )
    RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")
    
    export JAX_PLATFORMS=tpu
    export SPARSE_MODEL_TRAINING={sparse_model_training}
    
    bash scripts/convert.sh gen_param_ckpt \\
        --model=${{MODEL_NAME}} \\
        --orbax_ckpt_name=${{RUN_NAME}} \\
        --step={num_steps-1} \\
        --hf_model_name=Llama-3.1-8B \\
        --direct_run_name=${{RUN_NAME}}
        
    CKPT_DIR=$(gsutil ls -d gs://${{BUCKET_NAME}}/model_ckpts/direct/${{MODEL_NAME}}_{exp_type}_seqlen_${{SEQ_LEN}}_bs_*_grad_accum_*_lr_${{LR/e/*e}}_min_lr_ratio_${{MIN_LR_RATIO}}_warmup_ratio_${{WARMUP_RATIO}}*/checkpoints/0 )
    RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")
        
    bash scripts/convert.sh eval \\
        --model=${{MODEL_NAME}} \\
        --hf_model_name=Llama-3.1-8B \\
        --direct_run_name=${{RUN_NAME}}
    """)

    # Default script name if not provided
    if output_path is None:
        output_path = f"scripts/eval/{job_name}.sh"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)

    print(f"✅ Generated script at {output_path}")

def generate_script(*args, **kwargs):
    generate_training_script(*args, **kwargs)
    generate_eval_script(*args, **kwargs)

# Example usage
if __name__ == "__main__":

    # train from scratch
    for model_name in ["llama3.1-4b-depth"]:
        for num_steps in [12500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    # L200_finetune
    for model_name in ["llama3.1-4b-depth"]:
        for num_steps in [2500, 7500, 12500, 25000, 37500, 50000, 62500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                load_parameters_path="model_ckpts/maxtext/llama3-8b-l200_depth_task_wikitext_nlayers_16_calib_size_128_seqlen_8192/checkpoints/0/items",
                pretrain_tokens="L200",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )

    # for model_name in ["llama3.1-4b-depth", "llama3.1-4b-width"]:
    #     for num_steps in [2500, 7500, 12500, 25000, 37500, 50000, 62500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
            
    # for model_name in ["llama3.1-8b"]:
    #     for num_steps in [50000, 37500, 25000, 12500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
            
    # for model_name in [
    #     "llama3.1-440m", "llama3.1-1b", 
    #     "llama3.1-1.5b-depth", "llama3.1-2b-depth", "llama3.1-3b-depth",
    #     "llama3.1-2b-width", "llama3.1-3b-width",
    # ]:
    #     for num_steps in [12500, 62500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
            
    # for load_path, model_name in zip(
    #     [
    #         "model_ckpts/maxtext/llama3-8b-l200_depth_task_wikitext_nlayers_16_calib_size_128_seqlen_8192/checkpoints/0/items",
    #         "model_ckpts/maxtext/llama3-8b-l200_width_task_wikitext_hidden_size_3072_ffn_hidden_size_9216_calib_size_128_seqlen_8192/checkpoints/0/items", 
    #     ],
    #     [
    #         "llama3.1-4b-depth", 
    #         "llama3.1-4b-width"
    #     ]
    # ):
    #     for num_steps in [2500, 7500, 12500, 25000, 37500, 50000, 62500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             load_parameters_path=load_path,
    #             pretrain_tokens="L200",
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
            
    # for load_path, model_name in zip(
    #     ["model_ckpts/maxtext/llama3.1-4b-depth-orbax/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-4b-width-orbax/checkpoints/0/items"],
    #     ["llama3.1-4b-depth", "llama3.1-4b-width"]
    # ):
    #     for num_steps in [12500, 62500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             load_parameters_path=load_path,
    #             pretrain_tokens="HF",
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
                    
                    
    # for load_path, model_name in zip(
    #     ["model_ckpts/maxtext/llama3.1-1.5b-depth-minitron/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-2b-depth-minitron/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-3b-depth-minitron/checkpoints/0/items"],
    #     ["llama3.1-1.5b-depth", "llama3.1-2b-depth", "llama3.1-3b-depth"]
    # ):
    #     for num_steps in [12500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             load_parameters_path=load_path,
    #             pretrain_tokens="L200",
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
            
    # for load_path, model_name in zip(
    #     [
    #         "model_ckpts/maxtext/llama3-8b-l200_width_task_wikitext_hidden_size_1792_ffn_hidden_size_5632_calib_size_128_seqlen_8192/checkpoints/0/items", 
    #         "model_ckpts/maxtext/llama3-8b-l200_width_task_wikitext_hidden_size_2432_ffn_hidden_size_6144_calib_size_128_seqlen_8192/checkpoints/0/items"
    #     ],
    #     [   
    #         "llama3.1-2b-width", 
    #         "llama3.1-3b-width"
    #     ]
    # ):
    #     for num_steps in [12500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             load_parameters_path=load_path,
    #             pretrain_tokens="L200",
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
                    
    # for load_path in [
    #     "model_ckpts/maxtext/llama3.1_8b_L200_unstructured_0.5/checkpoints/0/items",
    #     "model_ckpts/maxtext/llama3.1_8b_L200_4:8_0.5/checkpoints/0/items", 
    #     "model_ckpts/maxtext/llama3.1_8b_L200_2:4_0.5/checkpoints/0/items",
        
    #     "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_unstructured_0.5/checkpoints/0/items",
    #     "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_2:4_0.5/checkpoints/0/items",
       
    # ]:
    #     generate_script(
    #         model_name="llama3.1-8b",
    #         num_steps=12500,
    #         load_parameters_path=load_path,
    #         sparse_model_training=True,
    #         pretrain_tokens="L200",
    #         # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #         # load_parameters_path=args.load_parameters_path,
    #         # output_path=args.output_path,
    #     )
    
    # for load_path in [
    #     "model_ckpts/maxtext/llama3.1_8b_L200_unstructured_0.5_reinit/checkpoints/0/items",
    #     # "model_ckpts/maxtext/llama3.1_8b_L200_4:8_0.5_reinit/checkpoints/0/items",
    #     "model_ckpts/maxtext/llama3.1_8b_L200_2:4_0.5_reinit/checkpoints/0/items",
    #     "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_2:4_0.5_reinit/checkpoints/0/items",
    #     "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_2:4_0.5_reinit/checkpoints/0/items",
    # ]:
    #     for num_steps in [12500, 62500]:
    #         generate_script(
    #             model_name="llama3.1-8b",
    #             num_steps=num_steps,
    #             load_parameters_path=load_path,
    #             sparse_model_training=True,
    #             # pretrain_tokens="L200",
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
        
    # for model_name in ["llama2-1.3b", "llama2-2.7b"]:
    #     for num_steps in [12500, 62500]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             seq_len=4096,
    #             data_files="/home/zephyr/gcs-bucket/datasets/dclm/llama2_array_record_with_special_tokens_64/*.array_record",
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )
            
    # for model_name in ["llama2-7b"]:
    #     for num_steps in [50000]:
    #         generate_script(
    #             model_name=model_name,
    #             num_steps=num_steps,
    #             seq_len=4096,
    #             data_files="/home/zephyr/gcs-bucket/datasets/dclm/llama2_array_record_with_special_tokens_64/*.array_record",
    #             # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
    #             # load_parameters_path=args.load_parameters_path,
    #             # output_path=args.output_path,
    #         )