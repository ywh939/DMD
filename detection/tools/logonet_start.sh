#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/det_model_cfgs/kitti/LoGoNet-kitti.yaml
