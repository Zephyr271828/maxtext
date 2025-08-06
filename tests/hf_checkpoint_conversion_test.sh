#!/bin/bash

export PYTHONPATH='/home/zephyr/gcs-bucket/maxtext':$PYTHONPATH

python /home/zephyr/gcs-bucket/maxtext/MaxText/tests/hf_checkpoint_conversion_test.py