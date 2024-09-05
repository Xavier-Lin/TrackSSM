

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
|MOT17       | 61.4 |   |   |   |   |
|DanceTrack  | 57.7 | 92.2 | 57.5 | 41.0 | 81.5  |
|SportsMOT   | 74.4 |  |  |  |   |

 ### Comparison on DanceTrack test set

    
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


## Model zoo



## Training

**Notes**: 


## Tracking


#### Tracking on dancetrack test set


 
## Citation -->


## Acknowledgements


