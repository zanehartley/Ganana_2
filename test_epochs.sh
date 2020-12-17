#!/usr/bin/bash
source /usr2/share/gpu.sbatch
python3 test_all_epochs.py --load_iter 36  --num_test 500
