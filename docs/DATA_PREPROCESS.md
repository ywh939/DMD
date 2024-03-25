Currently we provide the dataloader of KITTI, Dust.

### KITTI Dataset
* Please follow the instructions from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). We adopt the same data generation process.

```
detection
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── al3d_det
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m al3d_det.datasets.kitti_dataset.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Dust Dataset
* the process is same to KITTI
```
detection
├── data
│   ├── dust
│   │   │── calib/
│   │   │── training
│   │   │   ├──image & label & pcd 
│   │   │── gt_database/
│   │   │── split/
├── al3d_det
```