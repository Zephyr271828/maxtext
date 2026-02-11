import os
import glob
import shutil
import subprocess

direct_ckpt_dir = "/home/zephyr/gcs-bucket/model_ckpts/direct"
maxtext_ckpt_dir = "/home/zephyr/gcs-bucket/model_ckpts/maxtext"

direct_ckpts = glob.glob(os.path.join(direct_ckpt_dir, "*/checkpoints/0/items"))

maxtext_ckpts = glob.glob(os.path.join(maxtext_ckpt_dir, "*/checkpoints/*/items"))

direct_name2ckpt = {d.split('/')[-4]:d for d in direct_ckpts}

maxtext_name2ckpt = {d.split('/')[-4]:d for d in maxtext_ckpts}

for name, path in direct_name2ckpt.items():
    if name in maxtext_name2ckpt:
        print(f"Direct ckpt exists: {path}")
        parts = maxtext_name2ckpt[name].split('/')
        ckpt_to_delete = '/'.join(parts[:-3])
        res = input(f"Delete the parent directory of {maxtext_name2ckpt[name]}? (y/n)\n> ")
        if res.lower() == 'y':
            shutil.rmtree(ckpt_to_delete)
            print(f"Deleted {ckpt_to_delete}\n")
        # shutil.rmtree(path)
        # shutil.rmtree(maxtext_name2ckpt[name])
