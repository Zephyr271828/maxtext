import os
from textwrap import dedent

def generate_script(
    model_name: str,
    num_steps: int = 12500,
    seq_len: int = 8192,
    batch_size: int = 4,
    grad_accum: int = 2,
    grad_clip: float = 1.0,
    lr: float = 3e-4,
    min_lr_ratio: float = 0.1,
    warmup_ratio: float = 0.05,
    async_checkpointing: bool = False,
    data_files: str = "/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record",
    load_parameters_path: str = "",
    output_path: str = None,
    # start_from_file_index: int = 0,
):
    """Generate a TPU MaxText training shell script from template."""

    # Dynamically include load_parameters_path only if provided
    load_path_line = f"load_parameters_path=gs://${{BUCKET_NAME}}/{load_parameters_path} \\\n            " if load_parameters_path else ""

    load_part = load_parameters_path.replace("/", "_") if load_parameters_path else "none"
    
    # handle experiment type
    exp_type = "unknown"
    if not load_parameters_path:
        # if we are using a small model
        if any(x in model_name.lower() for x in ["4b", "3b", "2b", "1.5b", "1b"]):
            exp_type = f"S{num_steps // 250}"
        elif any(x in model_name.lower() for x in ["8b", "7b"]):
            exp_type = f"L{num_steps // 250}"
        start_from_file_index = 0
    else:
        if "minitron" in load_parameters_path:
            exp_type = f"L200_S{num_steps // 250}"
        else:
            exp_type = f"HF_S{num_steps // 250}"
        start_from_file_index = 50

    job_name = f"{model_name}_{exp_type}_seqlen_{seq_len}_bs_{batch_size}_grad_accum_{grad_accum}_lr_{lr}_minlr_{min_lr_ratio}_warmup_{warmup_ratio}"

    script = dedent(f"""\
    #!/bin/bash
    set -euo pipefail
    
    source scripts/get_tpu_bucket_name.sh

    export TPU_PREFIX="$(get_tpu_name)"
    export BUCKET_NAME="$(get_bucket_name)"

    export MODEL_NAME="{model_name}"
    export NUM_STEPS={num_steps}
    export SEQ_LEN={seq_len}
    export BATCH_SIZE={batch_size}
    export GRAD_ACCUM={grad_accum}
    export GRAD_CLIP={grad_clip}
    export LR={lr}
    export MIN_LR_RATIO={min_lr_ratio}
    export WARMUP_RATIO={warmup_ratio}
    export ASYNC_CHECKPOINTING={str(async_checkpointing).lower()}
    export BASE_OUTPUT_DIRECTORY="gs://${{BUCKET_NAME}}/model_ckpts/maxtext"
    export DATA_FILES="{data_files}"
    export RUN_NAME="${{MODEL_NAME}}_{exp_type}_seqlen_${{SEQ_LEN}}_bs_${{BATCH_SIZE}}_grad_accum_${{GRAD_ACCUM}}_lr_${{LR}}_min_lr_ratio_${{MIN_LR_RATIO}}_warmup_ratio_${{WARMUP_RATIO}}"
    export JAX_PLATFORMS=tpu

    python -u multihost_runner_orig.py \\
        --TPU_PREFIX=${{TPU_PREFIX}} \\
        --COMMAND="
        export TPU_LOG_DIR=/home/zephyr/tpu_logs
        export WANDB_API_KEY='7d11bbca76b3081b6bd1efbbcf1572aab26c5d56'
        source ~/maxtext_env/bin/activate
        python3.10 -u -m MaxText.train MaxText/configs/base.yml \\
            run_name=${{RUN_NAME}} \\
            {load_path_line}base_output_directory=${{BASE_OUTPUT_DIRECTORY}} \\
            dataset_type=grain \\
            grain_train_files=${{DATA_FILES}} \\
            start_from_file_index={start_from_file_index} \\
            grain_file_type='arrayrecord' \\
            grain_worker_count=1 \\
            enable_data_shuffling=False \\
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
            checkpoint_period=250 \\
            checkpoint_max_to_keep=1 \\
            use_wandb=True \\
            wandb_project=llm_pruning \\
            wandb_run_name=${{TPU_PREFIX}}_${{RUN_NAME}} \\
            packing=false
        "
    
    bash scripts/convert.sh gen_param_ckpt \\
        --model=${{MODEL_NAME}} \\
        --orbax_ckpt_name=${{RUN_NAME}} \\
        --step={num_steps-1} \\
        --hf_model_name=Llama-3.1-8B \\
        --direct_run_name=${{RUN_NAME}}
        
    bash scripts/convert.sh eval \\
        --model=${{MODEL_NAME}} \\
        --hf_model_name=Llama-3.1-8B \\
        --direct_run_name=${{RUN_NAME}}
    """)

    # Default script name if not provided
    if output_path is None:
        output_path = f"scripts/{job_name}.sh"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)

    print(f"✅ Generated script at {output_path}")


# Example usage
if __name__ == "__main__":

    for model_name in ["llama3.1-4b-depth", "llama3.1-4b-width"]:
        for num_steps in [12500, 62500]:
            generate_script(
                model_name=model_name,
                lr=3e-4,
                num_steps=num_steps,
                batch_size=2,
                grad_accum=4,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for model_name in ["llama3.1-8b"]:
        for num_steps in [50000]:
            generate_script(
                model_name=model_name,
                lr=3e-4,
                num_steps=num_steps,
                batch_size=2,
                grad_accum=4,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for model_name in ["llama3.1-1b", "llama3.1-1.5b-depth", "llama3.1-2b-depth", "llama3.1-3b-depth"]:
        for num_steps in [12500]:
            generate_script(
                model_name=model_name,
                lr=3e-4,
                num_steps=num_steps,
                batch_size=2,
                grad_accum=4,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for load_path, model_name in zip(
        ["model_ckpts/maxtext/llama3.1_minitron_depth_hf/0/items", "model_ckpts/maxtext/llama3.1_minitron_width_hf/0/items"],
        ["llama3.1-4b-depth", "llama3.1-4b-width"]
    ):
        for num_steps in [12500]:
            generate_script(
                model_name=model_name,
                lr=3e-4,
                num_steps=num_steps,
                batch_size=2,
                grad_accum=4,
                load_parameters_path=load_path,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            for load_path, model_name in zip(
                ["model_ckpts/maxtext/llama3.1-4b-depth-orbax/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-4b-width-orbax/checkpoints/0/items"],
                ["llama3.1-4b-depth", "llama3.1-4b-width"]
            ):
                for num_steps in [12500]:
                    generate_script(
                        model_name=model_name,
                lr=1e-4,
                num_steps=num_steps,
                batch_size=2,
                grad_accum=4,
                load_parameters_path=load_path,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            for load_path, model_name in zip(
                ["model_ckpts/maxtext/llama3.1-1.5b-depth-minitron/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-2b-depth-minitron/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-3b-depth-minitron/checkpoints/0/items"],
                ["llama3.1-1.5b-depth", "llama3.1-2b-depth", "llama3.1-3b-depth"]
            ):
                for num_steps in [12500]:
                    generate_script(
                        model_name=model_name,
                lr=3e-4,
                num_steps=num_steps,
                batch_size=2,
                grad_accum=4,
                load_parameters_path=load_path,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )