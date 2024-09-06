import os,cv2,glob
import torch,torchvision
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
from external.YOLOX.yolox.models.build import yolox_custom
import time
# from tracker.DiffMOTtracker import diffmottracker
from tracker.BYTETracker import BYTETracker
from tracker.BYTETracker_inital import BYTETracker_KF
from tracking_utils.utils import xyxy2xywh
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.visualization import plot_tracking

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
        if config.use_detection_model:
            self.num_classes = 1
            self.confthre = 0.01
            self.nmsthre = 0.7
            self.img_size = (800, 1440)
            self.half = True
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
        
        if self.config.use_detection_model:
            detection_model = yolox_custom(
                ckpt_path=self.config.det_ckpt, exp_path=self.config.exp_path, device="cuda").eval()
            logger.info("\tFusing model...")
            detection_model = self.fuse_model(detection_model)
            d_np = self.get_num_params(detection_model)
            m_np = self.get_num_params(self.model)
            logger.info(f'Motion model:  {m_np} parameters. Detection model: {d_np} parameters.')
 
            tensor_type = torch.cuda.HalfTensor if self.half else torch.cuda.FloatTensor
            if self.half:
                detection_model = detection_model.half()
                
        result_root = self.config.save_dir
        mkdirs(result_root)
        
        n_frame = 0
        timer_avgs, timer_calls = [], []
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
            if self.config.show_image:
                mkdirs(f'{result_root}/{seq}')
            for i, f in enumerate(frames):
                if frame_id % 10 == 0:
                    logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
                
                if self.config.use_detection_model:
                    img = cv2.imread(osp.join(img_path, imgs[i]))
                    ori_h, ori_w = img.shape[:2]
                    input_image = torch.as_tensor(self._process_image(img, self.img_size)[0]).type(tensor_type).unsqueeze(0)
                    
                    timer.tic()
                    outputs = detection_model(input_image)
                    dets = self._postprocess_results(outputs, ori_h, ori_w)[:, :5].detach().cpu().numpy()
                else:
                    f_path = osp.join(det_path, f)
                    timer.tic()
                    dets = np.loadtxt(f_path, dtype=np.float32, delimiter=',').reshape(-1, 6)[:, 1:6]
                
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
                if self.config.show_image:
                    img = cv2.imread(osp.join(img_path, imgs[i]))
                    online_im = plot_tracking(img.copy(), online_tlwhs, online_ids, frame_id=frame_id,
                                                fps=1. / timer.average_time)
                    cv2.imwrite(os.path.join(f'{result_root}/{seq}', '{:05d}.jpg'.format(frame_id)), online_im)
                frame_id += 1

            result_filename = osp.join(result_root, '{}.txt'.format(seq))
            write_results(result_filename, results)
            nf, ta, tc = len(frames), timer.average_time, timer.calls
            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)
        
        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

        # eval hota metrics by TrackEval tools
        if 'dance' in self.config.exp_name:
            gt_folder_path='../DanceTrack/dancetrack/val'
            val_map_path='../DanceTrack/dancetrack/val_seqmap.txt'
        elif 'mot17' in self.config.exp_name:
            gt_folder_path='../MOT17/train'
            val_map_path='../MOT17/val_seqmap.txt'
        elif 'sports' in self.config.exp_name:
            gt_folder_path='../sportsmot/val'
            val_map_path='../sportsmot/splits_txt/val.txt'
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
            if not self.config.eval_mode:
                checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            else:
                checkpoint_dir = osp.join( f"weights/sota/{self.config.exp_name}/FFN_epoch{epoch}.pt")
                # checkpoint_dir = osp.join( f"ablation/tracklen/dance/_epoch120.pt")
                # checkpoint_dir = osp.join( f"./SportsMOT_epoch1200.pt")
            self.checkpoint = torch.load(checkpoint_dir, map_location = "cpu")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder(self):
        if self.config.use_diffmot:
            self.encoder = History_motion_embedding()
        else:
            self.encoder = Time_info_aggregation()
        
    def _build_model(self):
        """ Define Model """
        config = self.config
        model = D2MP(config, encoder=self.encoder)

        self.model = model
        if not self.config.eval_mode:
            self.model = torch.nn.DataParallel(self.model, self.config.gpus).to('cuda')
            np = self.get_num_params(self.model)
            logger.info(f'Motion model:  {np} parameters')
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
    
    def _process_image(self, image, input_size, 
                       mean=(0.485, 0.456, 0.406), 
                       std=(0.229, 0.224, 0.225), 
                       swap=(2, 0, 1)
    ):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def _postprocess_results(self, outputs, ori_h, ori_w):
        outputs = self.postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)[0]
        bboxes = outputs[:, 0:4]
        
        # preprocessing: resize
        scale = min(
            self.img_size[0] / float(ori_h), self.img_size[1] / float(ori_w)
        )
        bboxes /= scale
        bboxes = xyxy2xywh(bboxes)

        clses = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]
        
        return torch.concat((bboxes, scores[:,None], clses[:, None]),dim=1)
    

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
            )

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output
    
    def fuse_conv_and_bn(self, conv, bn):
        # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
        fusedconv = (
            nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # prepare spatial bias
        b_conv = (
            torch.zeros(conv.weight.size(0), device=conv.weight.device)
            if conv.bias is None
            else conv.bias
        )
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


    def fuse_model(self, model):
        from yolox.models.network_blocks import BaseConv

        for m in model.modules():
            if type(m) is BaseConv and hasattr(m, "bn"):
                m.conv = self.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        return model
    
    def get_num_params(self, model):
        """Return the total number of parameters in a training model."""
        return sum(x.numel() for x in model.parameters())
