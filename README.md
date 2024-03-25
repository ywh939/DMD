# DMD
This repo is the official implementation of 3D object detection in dust

## Algorithm Modules
  ```
  detection
  ├── al3d_det
  │   ├── datasets
  │   │   │── DatasetTemplate: the basic class for constructing dataset
  │   │   │── augmentor: different augmentation during training or inference
  │   │   │── processor: processing points into voxel space
  │   │   │── the specific dataset module
  │   ├── models: detection model related modules
  |   |   │── fusion: point cloud and image fusion modules
  │   │   │── image_modules: processing images
  │   │   │── modules: point cloud detector
  │   │   │── ops
  │   ├── utils: the exclusive utils used in detection module
  ├── tools
  │   ├── cfgs
  │   │   │── det_dataset_cfgs
  │   │   │── det_model_cfgs
  │   ├── train/test/visualize scripts  
  ├── data: the path of raw data of different dataset
  ├── output: the path of trained model
  al3d_utils: the shared utils used in every algorithm modules
  docs: the readme docs for DMD
  ```

## Running
- Please cd the specific module and read the corresponding README for details
  - [Installation](docs/INSTALL.md)
  - [Data Preprocess](docs/DATA_PREPROCESS.md)
  - [Getting Started](docs/STARTED.md)