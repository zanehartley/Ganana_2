#!/usr/bin/bash
source /usr2/share/gpu.sbatch
python3 time_forwards.py --load_iter 20  --num_test 2
