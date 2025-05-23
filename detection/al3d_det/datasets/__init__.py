import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DistributedSampler as _DistributedSampler

from al3d_utils import common_utils

# from .dataset import DatasetTemplate
# from .dataset_kitti import DatasetTemplate_KITTI
# from .waymo.waymo_dataset import WaymoTrainingDataset, WaymoInferenceDataset
from .kitti.kitti_dataset import KittiDataset
# from .nuscenes.nuscenes_dataset import NuScenesDataset
from .dust.dust_dataset import DustDataset
from .mine.mine_dataset import MineDataset

__all__ = {
    # 'DatasetTemplate': DatasetTemplate,
    # 'WaymoTrainingDataset': WaymoTrainingDataset,
    # 'WaymoInferenceDataset': WaymoInferenceDataset,
    # 'DatasetTemplate_KITTI': DatasetTemplate_KITTI,
    'KittiDataset': KittiDataset,
    # 'NuScenesDataset': NuScenesDataset,
    'DustDataset': DustDataset,
    'MineDataset': MineDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, length=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    new_dataset = dataset

    if length > 0:
        indices = torch.arange(len(dataset))[:length]
        new_dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        new_dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        # shuffle=False, collate_fn=dataset.collate_batch,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler

def test_create_kitti_infos(dataset_cfg, root_dir):
    from .kitti.kitti_dataset import create_kitti_infos
    create_kitti_infos(dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=root_dir / 'data' / 'kitti_temo',
        save_path=root_dir / 'data' / 'kitti_temo')

def test_create_dust_infos(dataset_cfg, root_dir):
    from .dust.dust_dataset import create_dust_infos
    create_dust_infos(dataset_cfg=dataset_cfg,
        class_names=['Car'],
        data_path=root_dir / 'data' / 'dust',
        save_path=root_dir / 'data' / 'dust')

def test_create_mine_infos(dataset_cfg, root_dir):
    from .mine.mine_dataset import create_mine_infos
    create_mine_infos(dataset_cfg=dataset_cfg,
        class_names=['Mining-Truck'],
        data_path=root_dir / 'data' / 'mine',
        save_path=root_dir / 'data' / 'mine')