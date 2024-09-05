

# TrackSSM
####  TrackSSM is a general motion predictor with the state space model.

> [**TrackSSM: A General Motion Predictor by State-Space Model**](https://arxiv.org/abs/2409.00487)
> 
> Bin Hu, Run Luo, Zelin Liu, Cheng Wang, Wenyu Liu
> 
> *[arXiv 2409.00487](https://arxiv.org/abs/2409.00487)*


## News
- Submitting the paper on Arxiv at Sep 4 2024.
 
## Tracking performance
### Results on MOT challenge test set
| Dataset    | HOTA | MOTA | IDF1 | AssA | DetA | 
|------------|-------|-------|------|------|-------|
|MOT17       | 61.4 | 81.0 | 80.1 | 54.6% | 14.3% |
|DanceTrack  | 57.7 | 92.2 | 57.5 | 41.0 | 81.5  |
|SportsMOT   | 63.4 | 78.2 | 77.3 | 69.9% | 9.2%  |

 ### Comparison on DanceTrack test set
|  Method  | HOTA | DetA | AssA | MOTA | IDF1 |
|------------|-------|-------|------|------|-------|
| SparseTrack | 55.5 (**+7.8**) | 78.9 (**+7.9**) | 39.1 (**+7.0**) | 91.3 (**+1.7**) | 58.3 (**+4.4**) |
| ByteTrack  |  47.7 | 71.0 | 32.1 | 89.6 | 53.9 | 
    
**Notes**: 

 
## Installation
#### Dependence
This project is an implementation version of [Detectron2](https://github.com/facebookresearch/detectron2) and requires the compilation of [OpenCV](https://opencv.org/), [Boost](https://www.boost.org).

#### Compile GMC(Globle Motion Compensation) module

 
#### Install


## Data preparation
Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them under ROOT/ in the following structure:
```
ROOT
   |
   |——————SparseTrack(repo)
   |           └—————mix
   |                  └——————mix_17/annotations
   |                  └——————mix_20/annotations
   |                  └——————ablation_17/annotations
   |                  └——————ablation_20/annotations
   |——————MOT17
   |        └——————train
   |        └——————test
   └——————crowdhuman
   |         └——————Crowdhuman_train
   |         └——————Crowdhuman_val
   |         └——————annotation_train.odgt
   |         └——————annotation_val.odgt
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————Citypersons
   |        └——————images
   |        └——————labels_with_ids
   └——————ETHZ
   |        └——————eth01
   |        └——————...
   |        └——————eth07
   └——————dancetrack
               └——————train
               └——————train_seqmap.txt
               └——————test
               └——————test_seqmap.txt
               └——————val
               └——————val_seqmap.txt

   
```

Creating different training mix_data:
```
cd <ROOT>/SparseTrack

# training on CrowdHuman and MOT17 half train, evaluate on MOT17 half val.
python3 tools/mix_data_ablation.py

# training on CrowdHuman and MOT20 half train, evaluate on MOT20 half val.
python3 tools/mix_data_ablation_20.py

# training on MOT17, CrowdHuman, ETHZ, Citypersons, evaluate on MOT17 train.
python3 tools/mix_data_test_mot17.py

# training on MOT20 and CrowdHuman, evaluate on MOT20 train.
python3 tools/mix_data_test_mot20.py
```

## Model zoo
See [ByteTrack.model_zoo](https://github.com/ifzhang/ByteTrack#model-zoo). We used the publicly available ByteTrack model zoo trained on MOT17, MOT20 and ablation study for YOLOX object detection.

Additionally, we conducted joint training on MOT20 train half and Crowdhuman, and evaluated on MOT20 val half. The model as follows: [yolox_x_mot20_ablation](https://drive.google.com/file/d/1F2XwyYKj1kefLPUFRHxgnpaAmEwyoocw/view?usp=drive_link)

The model trained on DanceTrack can be available at google:[yolox_x_dancetrack](https://drive.google.com/drive/folders/1-uxcNTi7dhuDNGC5MmzXyllLzmVbzXay?usp=sharing) or baidu: [yolox_x_dancetrack](https://pan.baidu.com/s/1FIIy9mKnNQQrI7ACCAKRjQ), the extracted key as: sptk


## Training
All training is conducted on a unified script. You need to change the **VAL_JSON** and **VAL_PATH** in [register_data.py](https://github.com/hustvl/SparseTrack/blob/main/register_data.py), and then run as follows：
```
# training on MOT17, CrowdHuman, ETHZ, Citypersons, evaluate on MOT17 train set.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --num-gpus 4  --config-file mot17_train_config.py 


# training on MOT20, CrowdHuman, evaluate on MOT20 train set.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --num-gpus 4  --config-file mot20_train_config.py 
```
**Notes**: 
For MOT20, you need to clip the bounding boxes inside the image.

Add clip operation in line 138-139 in [data_augment.py](https://github.com/hustvl/SparseTrack/blob/main/datasets/data/data_augment.py), line 118-121 in [mosaicdetection.py](https://github.com/hustvl/SparseTrack/blob/main/datasets/data/datasets/mosaicdetection.py), line 213-221 in mosaicdetection.py, line 115-118 in [boxes.py](https://github.com/hustvl/SparseTrack/blob/main/utils/boxes.py).

## Tracking
All tracking experimental scripts are run in the following manner. You first place the model weights in the **<ROOT/SparseTrack/pretrain/>**, and change the **VAL_JSON** and **VAL_PATH** in [register_data.py](https://github.com/hustvl/SparseTrack/blob/main/register_data.py).
```
# tracking on mot17 train set or test set
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot17_track_cfg.py 


# tracking on mot20 train set or test set
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot20_track_cfg.py 


# tracking on mot17 val_half set
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot17_ab_track_cfg.py 


# tracking on mot20 val_half set
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot20_ab_track_cfg.py
```

#### Tracking on dancetrack test set
>step 1: Please comment out line 368-373 in the [sparse_tracker.py](https://github.com/hustvl/SparseTrack/blob/main/tracker/sparse_tracker.py) and modify the threshold for low-score matching stage from 0.3 to 0.35 (at line 402 in the sparse_tracker.py).
>
>step 2: Running:
```
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file dancetrack_sparse_cfg.py
```


 
## Citation -->
If you find SparseTrack is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{SparseTrack,
  title={SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth},
  author={Liu, Zelin and Wang, Xinggang and Wang, Cheng and Liu, Wenyu and Bai, Xiang},
  journal={arXiv preprint arXiv:2306.05238},
  year={2023}
}
```

## Acknowledgements
A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FairMOT](https://github.com/ifzhang/FairMOT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BOT-SORT), [Detectron2](https://github.com/facebookresearch/detectron2). 
 Many thanks for their wonderful works.

