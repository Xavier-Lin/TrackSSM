import numpy as np
import math
import cv2
from collections import defaultdict
import os.path as osp
import pycocotools.coco as coco
import torch
import torch.utils.data as data
import copy
import logging 
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode, ImageList, Instances, Boxes
from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.data import transforms as T
from detectron2.utils import comm
lg = logging.getLogger("detectron2")

__all__ = ['MOTtrainset']

class MOTtrainset(data.Dataset):
    # map from 'category_id' in the annotation files to 0..num_categories !
    def __init__(self, opt):
        super(MOTtrainset, self).__init__()
        self.opt = opt
        self.img_dir = opt.DATASETS.TRAIN_DATA.IMG_PATH
        ann_json_path = opt.DATASETS.TRAIN_DATA.ANN_PATH
    
        lg.info('==> Initializing {} data from {}, \n images from {} ...'.format("train", ann_json_path, self.img_dir))
        self.coco = coco.COCO(ann_json_path)
        self.images = self.coco.getImgIds()
        lg.info('Loaded initial dataset {} samples'.format(len(self.images)))
        cat_ids = sorted(self.coco.getCatIds())
        if min(cat_ids) > 0:
            lg.warning("""
                Category_ids in current annotations file are not start to 0, we will map 'category_id' in the annotation files to 0..num_categories !      
                      """)  
        self.id_map = {v: i for i, v in enumerate(cat_ids)}
        
        # init sampler_len
        self.sampler_len_list = opt.DATASETS.TRAIN_DATA.SAMPLER_LEN # 2 3 4 5
        self.sample_steps = opt.MODEL.MDR.SAMPLER_STEPS# 100 200 300
        assert len(self.sampler_len_list) == len(self.sample_steps) + 1
        
        # create sample videos
        max_sample_len = max(self.sampler_len_list)
        max_dist = self.opt.MODEL.MDR.MAX_FRAME_DIST
        last_drop_per_video = (max_sample_len - 1) * max_dist 

        lg.info('Creating video index and building maps from frame_id to image_id!')
        video_to_images = defaultdict(list)
        self.fid_2_imgid = defaultdict(dict)
        for image in self.coco.dataset['images']:
            video_to_images[image['video_id']].append(image)
            (self.fid_2_imgid[image['video_id']])[image['frame_id']] = image['id']
    
        # get video samples and 
        tol_samples = 0
        unsample_imgid = []
        for vid, v_info in video_to_images.items():
            tol_samples += len(v_info) - last_drop_per_video
            for img_info in v_info[-last_drop_per_video:]:
                unsample_imgid.append(img_info['id'])
        
        self.images = self.sub_imgset(self.images, unsample_imgid)
        self.num_samples = len(self.images)
        assert tol_samples == self.num_samples
        lg.info('Loaded custom dataset {} samples'.format(self.num_samples))
        self.trans_gen =self.build_transform_gen()
    
    def __len__(self):
        return self.num_samples
    
    def sub_imgset(self, tlista, tlistb):
        imgset = {}
        for t in tlista:
            imgset[t] = t
        for tid in tlistb:
            if imgset.get(tid, 0):
                del imgset[tid]
        return list(imgset.values())
    
    def update_sampler_len(self, curr_iter_steps):
        period_idx = 0
        for i in range(len(self.sample_steps)):
            if curr_iter_steps >= self.sample_steps[i]:
                period_idx = i + 1
        self.sampler_len = self.sampler_len_list[period_idx]   
        if comm.is_main_process():
            lg.info("set steps: iter={} period_idx={} sample_len={}".format(curr_iter_steps - 1, period_idx, self.sampler_len))
      

    def _load_image_anns(self, img_id):
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        img_path = osp.join(self.img_dir, img_info['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(self.coco.loadAnns(ids=ann_ids))
        img = utils.read_image(img_path, format=self.opt.INPUT.FORMAT)
        return img, anns, img_info, img_path

    def _load_data(self, index):# idx 
        img_id = self.images[index]# 获得img-id
        img, anns, img_info, img_path = self._load_image_anns(img_id)
        return img, anns, img_info, img_path
    
    def build_transform_gen(self):
        """
        Create a list of :class:`TransformGen` from config.
        Returns:
            list[TransformGen]
        """
        min_size = self.opt.INPUT.MIN_SIZE_TRAIN
        max_size = self.opt.INPUT.MAX_SIZE_TRAIN
        sample_style = self.opt.INPUT.MIN_SIZE_TRAIN_SAMPLING# 默认是choice
        if sample_style == "range":
            assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))
        tfm_gens = []
        tfm_gens.append(T.RandomFlip())
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        # tfm_gens.append(T.RandomApply(T.RandomRotation([0., 10.], False), 0.5))
        lg.info("TransformGens used in training: " + str(tfm_gens))
        return tfm_gens
    
    def load_pre_clip(self, video_id, frame_id):
        fid_2_imgid = self.fid_2_imgid[video_id]
        start_fid, end_fid, sample_interval = self._get_sample_range(frame_id)
    
        pre_imgs_list = []
        pre_anns_list = []
        for idx, fid in enumerate(range(start_fid, end_fid, sample_interval)):
            if idx == 0:
                continue
            img, anns, _, _ = self._load_image_anns(fid_2_imgid[fid])
            pre_imgs_list.append(img)
            pre_anns_list.append(anns)
        return pre_imgs_list, pre_anns_list
    
    def _get_sample_range(self, start_idx):
        sample_interval = np.random.randint(1, self.opt.MODEL.MDR.MAX_FRAME_DIST + 1)
        default_range = start_idx, start_idx + (self.sampler_len - 1) * sample_interval + 1, sample_interval
        return default_range
  
    def randshift(self, img, targets):
        self.xshift = 20 * np.random.rand(1) # 返回bs个 0-100之间的随机数 对应bs个图像的x方向的随机变化offset 
        self.xshift *= (np.random.randn(1) > 0.0) * 2 - 1 # 如果是T 则不变+ 如果是F 则 xshift为 -
        self.yshift = 20 * np.random.rand(1) # 返回bs个 0-100之间的随机数 对应bs个图像的y方向的随机变化offset
        self.yshift *= (np.random.randn(1) > 0.0) * 2 - 1 # 如果是T 则不变+ 如果是F 则 xshift为 -
        # xshift yshift 有正负号
      
        new_targets = copy.deepcopy(targets)
        
        for i, (image, target) in enumerate(zip([img], [targets])):# process for one img and target in a batch
            h, w ,_= image.shape #  h w c
            img_h, img_w = h, w  # original img size（未填充后的图像尺寸）
            nopad_image = image[ :img_h, :img_w, :]# original img （未填充的原图）
            image_patch = \
            nopad_image[
                    max(0, -int(self.yshift[i])) : min(h, h - int(self.yshift[i])), 
                    max(0, -int(self.xshift[i])) : min(w, w - int(self.xshift[i])),
                    :] # 获取偏移 并裁切后 的图像 此时的图像大小尺寸发生了变化！！
            
            patch_h, patch_w, _ = image_patch.shape # 偏移裁切后的图像 
            ratio_h, ratio_w = img_h / patch_h,  img_w / patch_w # 获取 原图/裁切 的 对应边长的比例
            resize_func=T.Resize((img_h, img_w))
            shifted_image, _ = T.apply_transform_gens([resize_func], image_patch)
          
            # shifted_image = T.interpolate(image_patch[None], size=(img_h, img_w))[0] # 进行最近邻上采样得到 c  img_h  img_w
            pad_shifted_image = copy.deepcopy(image)
            pad_shifted_image[:img_h, :img_w, :] = shifted_image 
            # 使用resize之后的偏移图像 填充image中原图所在的区域 --- 得到填充后的 偏移图像
            
            # 跟随图像的变换需要变换box坐标
            for j, gt in enumerate(target):
                bboxes = gt['bbox']# 原图的bbox坐标 x1y1wh
                bboxes -= np.array([max(0, -int(self.xshift[i])), max(0, -int(self.yshift[i])), 0, 0], dtype=np.float32)# 确定偏移之后的左上顶点坐标x1y1
                bboxes *= np.array([ratio_w, ratio_h, ratio_w, ratio_h], dtype=np.float32)# 最终确定扩放之后的 bbox的x1y1 坐标以及 wh
                new_targets[j]['bbox'] = bboxes.tolist() # 跟随变换后的归一化bbox坐标       
        del targets, image
        return pad_shifted_image, new_targets # 对应 偏移之后的图像 以及bbox坐标
    
    def filter_anns_out_img(self, anns, mini_area = 6.25, img_shape = None):
        new_anns = []
        for a in anns:
            w, h = a['bbox'][2] - a['bbox'][0], a['bbox'][3] - a['bbox'][1]
            x1, y1, x2, y2 = a['bbox'][0], a['bbox'][1], a['bbox'][2], a['bbox'][3]
            if not (w * h > mini_area and w > 0 and h > 0):
                continue
            elif not ((x1 < img_shape[1] and x2 > 0) and (y1 < img_shape[0] and y2 > 0)):
                continue
            new_anns.append(a)
        return new_anns
    
    def annotations_to_instances(self, annos, image_size):
        """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.

        Args:
            annos (list[dict]): a list of instance annotations in one image, each
                element for one instance.
            image_size (tuple): height, width

        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes",
                "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
                This is the format that builtin models expect.
        """
        boxes = (
            np.stack(
                [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
            )
            if len(annos)
            else np.zeros((0, 4))
        )
        target = Instances(image_size)
        target.gt_boxes = Boxes(boxes)

        classes = [int(self.id_map[obj["category_id"]]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes
        
        track_ids = []
        for obj in annos:
            if "track_id" in obj and obj["track_id"] != -1:
                track_ids.append(int(obj["track_id"]))
            else:
                track_ids.append(int(obj["id"]))
        track_ids = torch.tensor(track_ids, dtype=torch.int64)
        target.gt_track_ids = track_ids
        return target
    
    def aug_imgs_and_targets(self, data_transforms, image, anns):
        aug_img, transforms_pre = T.apply_transform_gens(data_transforms, image)
        image_shape = aug_img.shape[:2]  # h, w
        anns_copy = copy.deepcopy(anns)
        for ann in anns_copy:
            ann["bbox_mode"] = BoxMode.XYWH_ABS
          
        aug_anns = [
            self.transform_instance_annotations(obj, transforms_pre, image_shape,
                                                  is_clip=False
                                                  if "MOT17" in self.opt.DATASETS.TEST_DATA.DATA_NAME
                                                  else True)  
            for obj in anns_copy
            if obj.get("iscrowd", 0) == 0
        ]
        del anns
        return aug_img, aug_anns, image_shape, transforms_pre
    
    def transform_instance_annotations(
        self, annotation, transforms, image_size, *, keypoint_hflip_indices=None, is_clip = False
    ):
        """
        Apply transforms to box, segmentation and keypoints annotations of a single instance.

        It will use `transforms.apply_box` for the box, and
        `transforms.apply_coords` for segmentation polygons & keypoints.
        If you need anything more specially designed for each data structure,
        you'll need to implement your own version of this function or the transforms.

        Args:
            annotation (dict): dict of instance annotations for a single instance.
                It will be modified in-place.
            transforms (TransformList or list[Transform]):
            image_size (tuple): the height, width of the transformed image
            keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

        Returns:
            dict:
                the same input dict with fields "bbox", "segmentation", "keypoints"
                transformed according to `transforms`.
                The "bbox_mode" field will be set to XYXY_ABS.
        """
        if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)
        # bbox is 1d (per-instance bounding box)
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # clip transformed bbox to image size
        if is_clip:
            bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
            annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
        else:
            annotation["bbox"] = transforms.apply_box(np.array([bbox]))[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        if "segmentation" in annotation:
            # each instance contains 1 or more polygons
            segm = annotation["segmentation"]
            if isinstance(segm, list):
                # polygons
                polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                annotation["segmentation"] = [
                    p.reshape(-1) for p in transforms.apply_polygons(polygons)
                ]
            elif isinstance(segm, dict):
                # RLE
                mask = mask_util.decode(segm)
                mask = transforms.apply_segmentation(mask)
                assert tuple(mask.shape[:2]) == image_size
                annotation["segmentation"] = mask
            else:
                raise ValueError(
                    "Cannot transform segmentation of type '{}'!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict.".format(type(segm))
                )

        if "keypoints" in annotation:
            keypoints = transform_keypoint_annotations(
                annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
            )
            annotation["keypoints"] = keypoints

        return annotation
    
    def __getitem__(self, index):
        img, anns, img_info, img_path = self._load_data(index)
        # im=img.copy()
        # for a in anns:
        #   bbox=a['bbox']
        #   x1, y1, x2, y2 =  bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        #   im=cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=1)
        # cv2.imwrite('./curr_no_aug.jpg', im)
        # import pdb;pdb.set_trace()
        # selected img from xxx dataset
        # 以下分为2种 第一种 是针对crowdhuman （单张图像） 第二种是针对MOT以及DANCETRACK系列的视频数据
        # 第一步 获取当前帧的 标注 以及 图像 并对其进行 增广 保留确定的增广路径 
        aug_img, transforms = T.apply_transform_gens(self.trans_gen, img)
        
        image_shape = aug_img.shape[:2]  # h, w
        curr_anns = copy.deepcopy(anns)
        for cur_ann in curr_anns:
            cur_ann["bbox_mode"] = BoxMode.XYWH_ABS
        aug_anns = [
            self.transform_instance_annotations(obj, transforms, image_shape, 
                                                is_clip=False 
                                                if "MOT17" in self.opt.DATASETS.TEST_DATA.DATA_NAME
                                                else True) 
            for obj in curr_anns
            if obj.get("iscrowd", 0) == 0
        ]
      
        aug_anns = self.filter_anns_out_img(aug_anns, img_shape = image_shape)# 以后记住 任何 数据增强 之后 的必须操作都是 要必须有一个边界框过滤操作 --- 用以滤除 增强后 产生的无效框 
        curr_instances = self.annotations_to_instances(aug_anns, image_shape)
        
        # vis --- 对于增广图像和增广标注的验证无误
        # im=aug_img.copy()
        # for a in aug_anns:
        #   bbox=a['bbox']
        #   im=cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
        # cv2.imwrite('./curr.jpg', im)
        # import pdb;pdb.set_trace()
        
        #第二步 确定第二帧的 标注 以及 图像信息  使用第一步的相同的增广路径进行增广 
        if 'crowdhuman' not in img_path :# 提取 视频数据的 前N-1帧 以及 前N-1帧的标注 -- 这之中用了连续邻域帧提取 以 充分学习 不同时序组合的短视频clip 
            pre_imgs_list, pre_anns_list = self.load_pre_clip(
                img_info['video_id'], img_info['frame_id']
            ) 
            assert self.sampler_len - 1 == len(pre_imgs_list)
            pre_aug_img_set = []
            pre_aug_anns_set = []
            pre_image_shape_set = []
            for i in range(self.sampler_len - 1):
                pre_aug_img, pre_aug_anns, pre_image_shape, transforms_pre = self.aug_imgs_and_targets(transforms, pre_imgs_list[i], pre_anns_list[i])
                pre_aug_img_set.append(pre_aug_img)
                pre_aug_anns_set.append(pre_aug_anns)
                pre_image_shape_set.append(pre_image_shape)
              
            # vis --- 对于前一帧增广图像和增广标注的验证无误
            # for i in range(self.sampler_len - 1):
            #   im=pre_aug_img_set[i].copy()
            #   for a in pre_aug_anns_set[i]:
            #     bbox=a['bbox']
            #     im=cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
            #   cv2.imwrite(f'./pre_{i}.jpg', im)
            # import pdb;pdb.set_trace()
        else:
            # 创建伪视频帧 以及 伪视频帧的标注 
            pre_aug_img_set = []
            pre_aug_anns_set = []
            pre_image_shape_set = []
            for _ in range(self.sampler_len - 1):
                fake_img, fake_anns = self.randshift(img, anns)
                pre_aug_img, pre_aug_anns, pre_image_shape, transforms_pre = self.aug_imgs_and_targets(transforms, fake_img, fake_anns)
                pre_aug_img_set.append(pre_aug_img)
                pre_aug_anns_set.append(pre_aug_anns)
                pre_image_shape_set.append(pre_image_shape)
            # 对所有 伪视频帧 使用相同的数据增强，形成 伪视频clip！！
            # vis --- 对于前一帧增广图像和增广标注的验证无误
            # im = pre_aug_img_set[2].copy()
            # for a in pre_aug_anns_set[2]:
            #   bbox=a['bbox']
            #   im=cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
            # cv2.imwrite('./pre.jpg', im)
            # import pdb;pdb.set_trace()
        del anns
        
        pre_instances_set = []
        for pre_aug_anns, pre_image_shape in zip(pre_aug_anns_set, pre_image_shape_set):
            pre_aug_anns = self.filter_anns_out_img(pre_aug_anns, img_shape = pre_image_shape)# 任何 数据增强后 都 必须加上一个边界框过滤操作 
            pre_instances = self.annotations_to_instances(pre_aug_anns, pre_image_shape)
            pre_instances_set.append(pre_instances)
        
        all_clip_gts = []
        all_clip_imgs = [aug_img] + pre_aug_img_set
        all_clip_instance = [curr_instances] + pre_instances_set
        for i in range(self.sampler_len):
            gt_per_img = {}
            inp = torch.as_tensor(np.ascontiguousarray(all_clip_imgs[i].transpose(2, 0, 1)))
            gt_per_img['image'] = inp
            gt_per_img['width'] = img_info['width'] # 相同视频帧的 大小相同
            gt_per_img['height'] = img_info['height']
            gt_per_img['instances'] = utils.filter_empty_instances(all_clip_instance[i])
            all_clip_gts.append(gt_per_img)

        # vis 
        # import pdb;pdb.set_trace()
        # for i in range(self.sampler_len):
        #   im = all_clip_imgs[i].copy()
        #   # im = ret['image'].permute(1,2,0).cpu().numpy()
        #   # im = np.ascontiguousarray(im.copy()).astype(np.uint8)
        #   cv2.imwrite('./ddd.jpg', im)
        #   for bbox in all_clip_instance[i].gt_boxes.tensor:
        #     # bbox=a['bbox']
        #     im=cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=1)
        #   cv2.imwrite('./ddd.jpg', im)
        #   import pdb;pdb.set_trace()
        return all_clip_gts
