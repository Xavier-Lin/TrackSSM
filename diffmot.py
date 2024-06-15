import os,cv2,glob
import torch
from collections import OrderedDict
import numpy as np
import os.path as osp
import logging
from torch import nn, optim, utils
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import motmetrics as mm
from dataset import DiffMOTDataset
from models.autoencoder import D2MP
from models.condition_embedding import Time_info_aggregation, History_motion_embedding
from pathlib import Path
import time
from tracker.DiffMOTtracker import diffmottracker
from tracker.BYTETracker import BYTETracker
from tracker.BYTETracker_inital import BYTETracker_KF

from tracking_utils.log import logger
from tracking_utils.timer import Timer

def write_results(filename, results, data_type='mot'):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

class DiffMOT():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            self.train_dataset.augment = self.config.augment
            pbar = tqdm(self.train_data_loader, ncols=80)
            for batch in pbar:
                for k in batch:
                    batch[k] = batch[k].to(device='cuda', non_blocking=True)
                    
                train_loss = self.model(batch)
                train_loss = train_loss.mean()

                self.optimizer.zero_grad()
                pbar.set_description(f"Epoch {epoch},  Loss: {train_loss.item():.6f}")
                train_loss.backward()
                self.optimizer.step()

            if epoch % self.config.eval_every == 0:
                checkpoint = {
                    'ddpm': self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))

    def eval(self):
        det_root = self.config.det_dir
        img_root = det_root.replace('/detections/', '/')

        seqs = [s for s in os.listdir(det_root)]
        seqs.sort()

        for seq in seqs:
            print(seq)
            det_path = osp.join(det_root, seq)
            img_path = osp.join(img_root, seq, 'img1')

            info_path = osp.join(self.config.info_dir, seq, 'seqinfo.ini')
            seq_info = open(info_path).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            tracker = BYTETracker(self.config)
            # tracker = diffmottracker(self.config)
            # tracker = BYTETracker_KF(self.config)
            timer = Timer()
            results = []
            frame_id = 0

            frames = [s for s in os.listdir(det_path)]
            frames.sort()
            imgs = [s for s in os.listdir(img_path)]
            imgs.sort()

            for i, f in enumerate(frames):
                if frame_id % 10 == 0:
                    logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

                timer.tic()
                f_path = osp.join(det_path, f)
                dets = np.loadtxt(f_path, dtype=np.float32, delimiter=',').reshape(-1, 6)[:, 1:6]
                
                
                # vis
                # import pdb;pdb.set_trace()
                # img = cv2.imread(osp.join(img_path, imgs[i])).copy()
                
                # bboxes = dets[:, 0:4]
                # bboxes[:, 2:4] += bboxes[:, :2]

                # # preprocessing: resize
                # scale = min(
                #     800 / float(seq_height), 1440 / float(seq_width)
                # )
                # bboxes /= scale
                
                # for det in bboxes:
                #     # det[2:4] += det[:2]
                #     cv2.rectangle(img, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0,0,255), 2)
                # cv2.imwrite('./sss.jpg', img)
                # import pdb;pdb.set_trace()
                
                # track
                online_targets = tracker.update(dets, self.model, seq_width, seq_height)
                # online_targets = tracker.update(dets, self.model, frame_id, seq_width, seq_height, tag)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                timer.toc()
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                frame_id += 1

            # tracker.dump_cache()
            result_root = self.config.save_dir
            mkdirs(result_root)
            result_filename = osp.join(result_root, '{}.txt'.format(seq))
            write_results(result_filename, results)
            
        # eval hota metrics by TrackEval tools
        gt_folder_path='../MOT17/train'
        val_map_path='../MOT17/val_seqmap.txt'
        val_type='{gt_folder}/{seq}/gt/gt.txt'
        os.system(f"python ./TrackEval/scripts/run_mot_challenge.py  \
                                            --SPLIT_TO_EVAL train  \
                                            --METRICS HOTA CLEAR Identity\
                                            --GT_FOLDER {gt_folder_path}   \
                                            --SEQMAP_FILE {val_map_path}  \
                                            --SKIP_SPLIT_FOL True   \
                                            --TRACKERS_TO_EVAL '' \
                                            --TRACKER_SUB_FOLDER ''  \
                                            --USE_PARALLEL True  \
                                            --NUM_PARALLEL_CORES 8  \
                                            --PLOT_CURVES False   \
                                            --TRACKERS_FOLDER  {self.config.save_dir}  \
                                            --GT_LOC_FORMA {val_type}")


    def _build(self):
        self._build_dir()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_optimizer()

        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.eval_expname)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")


        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            # checkpoint_dir = osp.join( f"/data/zelinliu/DiffMOT/mot/sota/_epoch{epoch}.pt")
            self.checkpoint = torch.load(checkpoint_dir, map_location = "cpu")

        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder(self):
        self.encoder = Time_info_aggregation()
        # self.encoder = History_motion_embedding()


    def _build_model(self):
        """ Define Model """
        config = self.config
        
        model = D2MP(config, encoder=self.encoder)

        self.model = model
        if not self.config.eval_mode:
            self.model = torch.nn.DataParallel(self.model, self.config.gpus).to('cuda')
        else:
            self.model = self.model.cuda()
            self.model = self.model.eval()

        if self.config.eval_mode:
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in self.checkpoint['ddpm'].items()})
        # import pdb;pdb.set_trace()
        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        data_path = config.data_dir
        self.train_dataset = DiffMOTDataset(data_path, config)

        self.train_data_loader = utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.preprocess_workers,
            pin_memory=True
        )

    print("> Train Dataset built!")