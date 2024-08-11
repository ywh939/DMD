import pickle
mine_DATASET_OPEN3D = False
try:
    import open3d
    mine_DATASET_OPEN3D = True
except:
    pass
import numpy as np
# import cv2
from skimage import io as skio
import torch.utils.data as torch_data
import copy
from collections import defaultdict
from pathlib import Path

from ...utils.kitti_utils import box_utils
from al3d_det.datasets.processor.point_feature_encoder import PointFeatureEncoder
# from al3d_det.datasets.augmentor.data_augmentor import DataAugmentor
from al3d_det.datasets.processor.data_processor import DataProcessor
from al3d_utils.ops.roiaware_pool3d import roiaware_pool3d_utils
from al3d_utils.common_utils import keep_arrays_by_name as common_utils_keep_arrays_by_name
from al3d_utils.common_utils import get_pad_params as common_utils_get_pad_params


def get_objects_from_label(label_file):
    if isinstance(label_file, list):
        lines = label_file
    else:
        with open(label_file, 'r') as f:
            lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

def cls_type_to_id(cls_type):
    type_to_id = {'Mining-Truck': 1}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

MINE_RECT_CLASS_NAME = set(['Excavator/bottom', 'Excavator/top'])

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')

        self.cls_type = label[0]
        if self.cls_type in MINE_RECT_CLASS_NAME:
            self.cls_type = self.cls_type.replace("/", "_")
        
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.box2d = np.array((float(label[3]), float(label[4]), float(label[5]), float(label[6])), dtype=np.float32)
        self.l = float(label[7])
        self.w = float(label[8])
        self.h = float(label[9])
        self.lwh = np.hstack((self.l, self.w, self.h))

        self.loc = np.array((float(label[12]), float(label[10]), float(label[11])), dtype=np.float32)
        self.loc[2] -= self.h / 2

        self.ry = float(label[13])
        self.rz = float(label[14])
        self.rx = float(label[15])
        
        self.level = 0

                    

def format_calib_str(calib_str):
    return np.array(calib_str.strip().split(' '), dtype=np.float32).reshape(1, 4)

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    ca1 = format_calib_str(lines[0])
    ca2 = format_calib_str(lines[1])
    ca3 = format_calib_str(lines[2])
    ca4 = format_calib_str(lines[3])

    li2cam = np.concatenate((ca1, ca2, ca3, ca4), axis=0)

    return li2cam
    
def boxes3d_mine_lidar_to_imageboxes(boxes3d_lidar, calib, image_shape):
    corners_lidar = box_utils.boxes_to_corners_3d(boxes3d_lidar)
    pts_img, _ = calib.lidar_to_img(corners_lidar.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image
    
class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.V2C = calib  # 4 x 4
        self.CAM = np.float32([
            1484.74555368, 0,          1120.46023222,
            0,           1217.0010484, 808.94286133,
            0,           0,            1,
        ]).reshape(3,3)
        
    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom
    
    def lidar_to_camera(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_cam: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_lidar)  # (N, 4)
        pts_lidar = np.dot(pts_rect_hom, self.V2C.T)
        return pts_lidar[:, 0:3]

    def camera_to_img(self, pts_cam):
        """
        :param pts_cam: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = pts_cam @ self.CAM.T
        pts_img = pts_rect[:, 0:2] / pts_rect[:, 2:]
        return pts_img, pts_rect[:, 2]

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pcd_camera = self.lidar_to_camera(pts_lidar)
        pcd_image = self.camera_to_img(pcd_camera)
        return pcd_image


class MineDataset(torch_data.Dataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__()
        
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.use_image = getattr(self.dataset_cfg, "USE_IMAGE", False)
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        
        split_dir = self.root_path / 'split' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
    
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.calib = None

        self.mine_infos = []

        self.spatial_temporal_min_series = getattr(self.dataset_cfg, "SPATIAL_TEMPORAL_MIN_SERIES", 2)
        self.spatial_temporal_series = getattr(self.dataset_cfg, "SPATIAL_TEMPORAL_SERIES", 0)
        self.sorted_samples = []

        self.include_mine_data(self.mode)
    
    @property
    def mode(self):
        return 'train' if self.training else 'test'
        
    def include_mine_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Mine dataset')
        mine_infos = []

        # filter_path = self.root_path / 'split'/'val_30.txt'
        # filter_id = [x.strip() for x in open(filter_path).readlines()]

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
            mine_infos.extend(infos)

        self.mine_infos.extend(mine_infos)

        # tmp_info = []
        # for info in self.mine_infos:
        #     if info['point_cloud']['lidar_idx'] in filter_id:
        #         tmp_info.append(info)

        # self.mine_infos = tmp_info
        
        # self.filter_id = [1661512340_899267, 1661512340997127, 1661512341297394]
        # self.filter_index = []
        if self._open_spatial_temporal():
            for index, info in enumerate(self.mine_infos):
                id = self._get_sample_int_id(info['point_cloud']['lidar_idx'])
                self.sorted_samples.append(id)
                # if id in self.filter_id:
                #     self.filter_index.append(index)

            self.sorted_samples.sort()
            filter_samples = [self._get_sample_str_id(item) for item in self.sorted_samples[0:self.spatial_temporal_series-1]]
            tmp_mine_infos = [item for item in self.mine_infos if item['point_cloud']['lidar_idx'] not in filter_samples]
            self.mine_infos = tmp_mine_infos

        # batch_list = [self.__getitem__(9), self.__getitem__(20)]
        # ret = self.collate_batch(batch_list)
        # import torch
        # ret['image_series'] = torch.from_numpy(ret['image_series']).permute(0, 4, 1, 2, 3).float().contiguous()
        
        # from ...models.image_modules.p3d import P3D
        # self.p3d = P3D([3, 8, 36, 3])
        # self.p3d(ret['image_series'])

        if self.logger is not None:
            self.logger.info('Total samples for Mine dataset: %d' % (len(self.mine_infos)))
    
    def __len__(self):
        return len(self.mine_infos)
    
    def _open_spatial_temporal(self):
        return self.spatial_temporal_series >= self.spatial_temporal_min_series
    
    def _get_sample_str_id(self, int_id):
        rect_id = list(str(int_id))
        rect_id.insert(10, '_')
        return ''.join(rect_id)
    
    def _get_sample_int_id(self, str_id):
        return int(str_id.replace('_', ''))
    
    def _get_image_series(self, sample_idx):
        sorted_sample_index = self.sorted_samples.index(self._get_sample_int_id(sample_idx))
        image_series = []
        for i in range(self.spatial_temporal_series, 0, -1):
            former_sample_id = self.sorted_samples[sorted_sample_index - i + 1]
            image_series.append(self.get_image(self._get_sample_str_id(former_sample_id)))
        return np.stack(image_series, axis=0)

    def __getitem__(self, index):
        # if self._open_spatial_temporal():
        #     rect_sample_idx = int(self.mine_infos[index]['point_cloud']['lidar_idx'].replace('_', ''))
        #     sorted_sample_index = self.sorted_samples.index(rect_sample_idx)
        #     if self.spatial_temporal_series - 1 >= sorted_sample_index + 1:
        #         index += self.spatial_temporal_series - 1 - sorted_sample_index

        # sample_idx = self.mine_infos[index]['point_cloud']['lidar_idx']
        # a = self._get_image_series(sample_idx)
        # return

        info = copy.deepcopy(self.mine_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['gt_boxes_lidar']
            })
            
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            # road_plane = self.get_road_plane(sample_idx)
            # if road_plane is not None:
            #     input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            # if self.dataset_cfg.FOV_POINTS_ONLY:
            #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
            #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            #     points = points[fov_flag]
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)
            input_dict['image_shape'] = info['image']['image_shape']
            if self._open_spatial_temporal():
                input_dict['image_series'] = self._get_image_series(sample_idx)

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            # gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            # calib = data_dict['calib']
            # data_dict = self.data_augmentor.forward(
            #     data_dict={
            #         **data_dict,
            #         'gt_boxes_mask': gt_boxes_mask
            #     }
            # )
            # data_dict['calib'] = calib

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils_keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        data_dict.pop('gt_names', None)

        return data_dict
    
    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        batch_size = len(batch_list)

        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils_get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils_get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = 0 #np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
    
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            
            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_dict['bbox'] = boxes3d_mine_lidar_to_imageboxes(pred_boxes, calib, image_shape)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.mine_infos[0].keys():
            return None, {}

        from al3d_det.datasets.mine import mine_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.mine_infos]

        # annos_path_root = "/root/LoGoNet-py37/annos/origin/"
        # import torch
        # for obj in eval_det_annos:
        #     boxes_lidar_path = annos_path_root + obj['frame_id'] + '.pth'
        #     torch.save(obj['boxes_lidar'], boxes_lidar_path)

        # annos_path_root = "/root/LoGoNet-py37/annos/raw/"
        # for obj in self.mine_infos:
        #     boxes_lidar_path = annos_path_root + obj['point_cloud']['lidar_idx'] + '.pth'
        #     torch.save(obj['annos']['gt_boxes_lidar'], boxes_lidar_path)

        ap_result_str, ap_dict = mine_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)


        return ap_result_str, ap_dict

    def set_split(self, split):
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'split' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_pcd = None 
        if mine_DATASET_OPEN3D:
            lidar_file = self.root_split_path / 'lidar' / ('%s.pcd' % idx)
            assert lidar_file.exists()
            lidar_pcd = np.asarray(open3d.io.read_point_cloud(str(lidar_file)).points, dtype=np.float32)
        else:
            lidar_file = self.root_split_path / 'lidar' / ('%s.bin' % idx)
            assert lidar_file.exists()
            lidar_pcd = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 3)
            
        return self.get_calib(idx).cart_to_hom(lidar_pcd)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image' / ('%s.png' % idx)
        assert img_file.exists()
        image = skio.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image' / ('%s.png' % idx)
        assert img_file.exists()
        image = skio.imread(img_file)
        return np.array(image.shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.txt' % idx)
        assert label_file.exists()
        objs = get_objects_from_label(label_file)
        return objs
    
    def get_calib(self, idx):
        calib_file = self.root_split_path / 'licam_calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return Calibration(calib_file)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            
            calib = self.get_calib(sample_idx)
            calib_info = {'Cam': calib.CAM, 'V2C': calib.V2C}
            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0) # image coordinate
                annotations['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list]) # lidar coordinate
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0) # lidar coordinate
                annotations['rotation_y'] = np.array([obj.rz for obj in obj_list]) # lidar coordinate
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc_lidar = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # with futures.ThreadPoolExecutor(num_workers) as executor:
        #     infos = executor.map(process_single_scene, sample_id_list)
        # return list(infos)
        ret_list = []
        print(f'create {len(sample_id_list)} samples')
        for sample_id in sample_id_list:
            infos = process_single_scene(sample_id)
            ret_list.append(infos)
        return ret_list
    
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = self.root_path / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = self.root_path / ('mine_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                                'difficulty': difficulty[i], 'bbox': bbox[i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))
            
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_mine_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = MineDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('mine_infos_%s.pkl' % train_split)
    val_filename = save_path / ('mine_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    mine_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(mine_infos_train, f)
    print('Mine info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    mine_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(mine_infos_val, f)
    print('Mine info val file is saved to %s' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')