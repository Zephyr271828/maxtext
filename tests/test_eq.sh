source ~/gcs-bucket/miniconda3/etc/profile.d/conda.sh
conda activate ~/conda_envs/maxtext

export PYTHONPATH='/home/zephyr/gcs-bucket/maxtext':$PYTHONPATH

export META_CHECKPOINT_PATH='/home/zephyr/gcs-bucket/model_ckpts/minitron/llama3_4b_width_hf'
export CONVERTED_CHECKPOINT_PATH='gs://llm_pruning_us_central2_b/model_ckpts/minitron/llama3_4b_width_orbax'

export TPU_LOG_DIR=~/tpu_logs
JAX_PLATFORMS=cpu python /home/zephyr/gcs-bucket/maxtext/tests/test_eq.py \
    --base-model-path ${META_CHECKPOINT_PATH} \
    --model-size llama3-4b-width \
    --maxtext-model-path ${CONVERTED_CHECKPOINT_PATH} \
    --huggingface-checkpoint True
  