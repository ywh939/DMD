import yaml
from pathlib import Path
from easydict import EasyDict


dataset_cfg = EasyDict(yaml.safe_load(open(f'tools/cfgs/det_dataset_cfgs/kitti_dataset_mm.yaml')))

ROOT_DIR = Path(__file__).resolve().parent

from al3d_det.datasets import test_create_kitti_infos, test_create_dust_infos
# test_create_kitti_infos(
#     dataset_cfg=dataset_cfg,
#     root_dir=ROOT_DIR
# )

dataset_cfg = EasyDict(yaml.safe_load(open(f'tools/cfgs/det_dataset_cfgs/dust_dataset.yaml')))
test_create_dust_infos(
    dataset_cfg=dataset_cfg,
    root_dir=ROOT_DIR
)