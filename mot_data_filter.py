import os.path as osp
import os, glob
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/data/zelinliu/mot/train'
label_root = '/data/zelinliu/mot/trackers_gt/'
min_len = 7

seqs = sorted([s for s in os.listdir(seq_root)])

for seq in seqs:
    print(seq)
    gt_txt = osp.join(label_root, seq, "img1/*.txt")
    tracks_per_video = sorted(glob.glob(gt_txt))
    for track in tracks_per_video:
        gt = np.loadtxt(track, dtype=np.float64)
        tracks_len = len(gt.reshape(-1,7))
        if tracks_len < min_len:

            print(track)
            # pad_tracks = np.repeat((gt[-1] if gt.ndim != 1 else gt).reshape(-1,7), min_len - tracks_len, 0)
            # pad_gt = np.concatenate((gt.reshape(-1,7), pad_tracks), axis=0)
 
            seq_label_root = osp.join(label_root, seq, 'img1')
            mkdirs(seq_label_root)
            os.system(f'rm -rf {track}')
            # for _, fid, ncx, ncy, nw, nh, vis in pad_gt:
                
            #     label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            #         int(fid), ncx, ncy, nw, nh, vis)
            #     with open(track, 'a') as f:
            #         f.write(label_str)
