# For DanceTrack
# gt_folder_path=/data/zelinliu/DanceTrack/dancetrack/val
# val_map_path=/data/zelinliu/DanceTrack/dancetrack/val_seqmap.txt

# For SportsMOT:
# gt_folder_path=/data/zelinliu/sportsmot/val
# val_map_path=/data/zelinliu/sportsmot/splits_txt/val.txt  # 72.761, 73.009 74.049(140) 73.63(180) --- goal -- 74.1.    81.068 HOTA

# For MOT17 train:
gt_folder_path=/data/zelinliu/MOT17/train
val_map_path=/data/zelinliu/MOT17/val_seqmap.txt

# For MOT20 train:
# gt_folder_path=/data/zelinliu/MOT20/train
# val_map_path=/data/zelinliu/mot/20train_seqmap.txt

track_results_path=/data/zelinliu/DiffMOT/results/val/yolox_m_lt_ddm_1000eps_deeper_800_1rev
# need to change 'gt_val_half.txt' or 'gt.txt'
val_type='{gt_folder}/{seq}/gt/gt.txt'

# command
python ./TrackEval/scripts/run_mot_challenge.py  \
        --SPLIT_TO_EVAL train  \
        --METRICS HOTA \
        --GT_FOLDER ${gt_folder_path}   \
        --SEQMAP_FILE ${val_map_path}  \
        --SKIP_SPLIT_FOL True   \
        --TRACKERS_TO_EVAL '' \
        --TRACKER_SUB_FOLDER ''  \
        --USE_PARALLEL True  \
        --NUM_PARALLEL_CORES 8  \
        --PLOT_CURVES False   \
        --TRACKERS_FOLDER  ${track_results_path}  \
        --GT_LOC_FORMA ${val_type}