#!/usr/bin/bash
source /usr2/share/gpu.sbatch
python3 test_all_epochs.py --load_iter 30  --num_test 11
