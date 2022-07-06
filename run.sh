#!/bin/bash

python pys/AMRNet_smartcity.py --seed 42 --log_para 1000 --crop_size 448 --downsample 1 --num_workers 32 --batch_size 8 --n_epochs 120 --mode test