# MaxText Modified

This is a modified version of MaxText for my personal training use. The features I plan to add and have added will be listed below.  
For the native README of MaxText see [README_ORIGINAL.md](README_ORIGINAL.md).

## Implementation Roadmap
- [x] add mak-number-of-checkpoints support
- [ ] add wandb logging to MaxText
- [x] add orbax lm-eval-harness adapter
- [ ] add a checker and doc to verify the consistency between Huggingface and ORBAX format

## Max Number of Checkpoints Support
MaxText does not natively support limiting the number of checkpoints to save (i.e., deleting the older ones and only keep the last `k` checkpoints). This branch supports this features by adding [this line](https://github.com/Zephyr271828/maxtext/blob/0ac88df254f6d4ae1da377a1549e29309223f878/MaxText/checkpointing.py#L86).  
In order to use this feature, you can simply add `checkpoint_max_to_keep=` to your config. You may refer to the [base config](https://github.com/Zephyr271828/maxtext/blob/0ac88df254f6d4ae1da377a1549e29309223f878/MaxText/configs/base.yml#L49).

## ORBAX Adapter
Coming Soon!


## Pretraining Llama-3.1-8B
Currently Taiming Lu and I plan to pretrain Llama-3-8B together. Below are some records for verification:

### Training Recipe
Following the original recipe of training Llama-3-8B, we adopt:
| Learning Rate | 
|:--:|
| - |

### Training Data
We plan to use [DCLM-Baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0). The [leaderboard](https://github.com/mlfoundations/dclm?tab=readme-ov-file#leaderboard) showcased the effectiveness of using DCLM to pretrain model.  
In order to train on MaxText with DCLM, I 
1. download the raw `jsonl.zst` files from huggingface.
2. decompress to `.jsonl` files.
3. convert each `.jsonl` file to an `.array_record` file.
See my scripts at `gs://llm_pruning_us_central2_b/datasets/dclm/scripts/{download.py, jsonl2arrayrecord.py}` for verification.

### Training Script
Check my training script at [`pretrain/llama3_8b_L200.sh`](pretrain/llama3_8b_L200.sh).

### Other Modifications
I haev modified `multihost_runner.py` to make it work in my docker environment. To see the original version, see [`multihost_runner_orig.py`](multihost_runner_orig.py).

