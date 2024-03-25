#!/bin/bash

FIX_RANDOM_SEED=$1
NOHUP_LOG_PATH="/root/LoGoNet-py37/detection/output/det_model_cfgs/kitti/LoGoNet-kitti/nohuplog"
echo ${NOHUP_LOG_PATH}

# CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --${FIX_RANDOM_SEED} >> nohup.log 2>&1 &