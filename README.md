

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
>
> cd external/YOLOX,run: python setup.py develop


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
and then, run
```
python dancetrack_data_process.py
python sports_data_process.py
python mot_data_process.py
```

## Model zoo
### Detection Model
Refer to [Detection Model](https://github.com/Kroery/DiffMOT/releases/tag/v1.0).

### Motion Model
Refer to :
[MOT17-61.4 HOTA](https://drive.google.com/file/d/1KuTmi4t9qwcm2dXCW6xPY2dVhWSs6jK8/view?usp=drive_link),
[DanceTrack-57.7 HOTA](https://drive.google.com/file/d/1VvOjZNG3QPI4TPWl13ibUzuVvFyxCTa9/view?usp=drive_link),
[SportsMOT-74.4 HOTA](https://drive.google.com/file/d/1Uu6S-kYZoTZAq1RbwlZtyBH5Y42W7vB2/view?usp=drive_link).






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
  - For SportsMOT,  we should use both GIoU loss and smooth L1 loss in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).


## Tracking
#### Tracking on MOT17 test set
```
python main.py --config ./configs/mot17_test.yaml
```

#### Tracking on DanceTrack test set
```
python main.py --config ./configs/dancetrack_test.yaml
```

#### Tracking on SportsMOT test set
```
python main.py --config ./configs/sportsmot_test.yaml
```
**Notes**:
  - For tracking on MOT17, we should unenable line 60 in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).
  - Before perform tracking process, change det_dir, info_dir and save_dir in config files.
  - The ***use_detection_model*** is an optional item. When making the ***use_detection_model*** project effective, the detector will participate in the process of tracking inference, not just the motion model.
  - The ***interval*** the length of the historical trajectory involved in training and inference.

 
## Citation
```bibtex
@misc{trackssm,
      title={TrackSSM: A General Motion Predictor by State-Space Model}, 
      author={Bin Hu and Run Luo and Zelin Liu and Cheng Wang and Wenyu Liu},
      year={2024},
      eprint={2409.00487},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.00487}, 
}
```

## Acknowledgements
A large part of the code is borrowed from [DiffMOT](https://github.com/Kroery/DiffMOT), [FairMOT](https://github.com/ifzhang/FairMOT), [ByteTrack](https://github.com/ifzhang/ByteTrack). 
 Many thanks for their wonderful works.


