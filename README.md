

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
|MOT17       | 61.4 | 78.5 | 74.1 | 59.6 | 63.6 |
|DanceTrack  | 57.7 | 92.2 | 57.5 | 41.0 | 81.5 |
|SportsMOT   | 74.4 | 96.8 | 74.5 | 62.4 | 88.8 |

 
## Installation
> Creating a new environment.
> 
> Running: pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
> 
> Compiling [mamba](https://github.com/state-spaces/mamba)
> 
> Running: pip install -r requirement.txt


## Data preparation
Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [DanceTrack](https://github.com/DanceTrack/DanceTrack), [SportsMOT](https://github.com/MCG-NJU/SportsMOT) and put them under ROOT/ in the following structure. The structure of the MIX dataset follows the method used in [DiffMOT](https://github.com/Kroery/DiffMOT):
```
ROOT
   |
   |——————TrackSSM(repo)
   |                         
   |——————mot(MIX)
   |        └——————train(MOT17 train set and MOT20 train set)
   |        └——————test(MOT17 test set and MOT20 test set)
   |——————DanceTrack
   |           └——————train
   |           └——————train_seqmap.txt
   |           └——————test
   |           └——————test_seqmap.txt
   |           └——————val
   |           └——————val_seqmap.txt
   └——————SportsMOT
              └——————train
              └——————test
              └——————val
              └——————splits_txt
                         └——————train.txt
                         └——————val.txt
                         └——————test.txt
```


## Model zoo
### Detection Model
Refer to [Detection Model](https://github.com/Kroery/DiffMOT).

### Motion Model
Refer to [Motion Model].


## Training
### Training Detection Model
Refer to [ByteTrack](https://github.com/ifzhang/ByteTrack).

### Training Motion Model
- Changing the data_dir in config
- Training on the MIX, DanceTrack and SportsMOT:
```
python main.py --config ./configs/dancetrack.yaml
python main.py --config ./configs/sportsmot.yaml
python main.py --config ./configs/mot.yaml
```
**Notes**:
  - For MIX, we should unenable line 60 in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).
  - For MIX and DanceTrack, we should unenable GIoU loss in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).
  - For SportsMOT,  we should use GIoU loss in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).


## Tracking


#### Tracking on dancetrack test set


 
## Citation -->


## Acknowledgements


