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
    start_from_file_index: int = 0,
):
    """Generate a TPU MaxText training shell script from template."""

    run_name = (
        f"{model_name}_L200_S50_seqlen_{seq_len}_bs_{batch_size}_"
        f"grad_accum_{grad_accum}_lr_{lr}_min_lr_ratio_{min_lr_ratio}_"
        f"warmup_ratio_{warmup_ratio}"
    )

    # Dynamically include load_parameters_path only if provided
    load_path_line = f"load_parameters_path=gs://${{BUCKET_NAME}}/{load_parameters_path} \\\n            " if load_parameters_path else ""

    load_part = load_parameters_path.replace("/", "_") if load_parameters_path else "none"
    
    job_name = f"{model_name}_steps_{num_steps}_lr_{lr}_load_{load_part}"

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
    export RUN_NAME="{run_name}"
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
            checkpoint_period=10 \\
            checkpoint_max_to_keep=1 \\
            use_wandb=True \\
            wandb_project=llm_pruning \\
            wandb_run_name=${{TPU_PREFIX}}_${{RUN_NAME}} \\
            packing=false
        "
    
    bash scripts/convert.sh gen_param_ckpt \\
        --model=${{MODEL_NAME}} \\
        --orbax_ckpt_path=${{BASE_OUTPUT_DIRECTORY}}/${{RUN_NAME}}/checkpoints/{num_steps-1}/items \\
        --hf_model_path=/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B \\
        --direct_run_name=direct_{job_name}
        
    bash scripts/convert.sh eval \\
        --model=${{MODEL_NAME}} \\
        --hf_model_path=/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B \\
        --direct_run_name=direct_{job_name} 
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
            
        