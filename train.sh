#!/usr/bin/bash
source /usr2/share/gpu.sbatch
python3 train.py --continue_train true --epoch_count 5 --load_iter 4
