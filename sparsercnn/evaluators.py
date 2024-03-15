import os
import torch
import logging
import numpy as np
from collections import defaultdict
import time
__all__ = ["MOTEvaluator"]

logger = logging.getLogger("detectron2")
    
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time_synchronized() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, cfg, dataloader):
        """
        cfg:
            dataloader (Dataloader): evaluate dataloader.
        """
        self.dataloader = dataloader
        self.cfg = cfg

    def evaluate(
        self,
        model,
        half=False,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        cfg:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        results = []
        timer_avgs, timer_calls = [], []
        video_names = defaultdict()
      
        timer = Timer()
        video_id = 0

        for cur_iter, batch_data in enumerate(self.dataloader):
            with torch.no_grad():
                # init tracker  
                frame_id = int(batch_data[0]["frame_id"])
                if frame_id == 1:
                    video_id += 1 
                img_file_name = batch_data[0]["file_name"]
                video_name = img_file_name.split('/')[-3]
                    
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    model.reset()
                    if len(results) != 0:
                        # import pdb;pdb.set_trace()
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []
                    timer_avgs.append(timer.average_time)
                    timer_calls.append(timer.calls)
                    timer.clear()
                
                # run tracking
                batch_data[0]["image"] = batch_data[0]["image"].type(tensor_type)  
                timer.tic()
                online_targets = model(batch_data) 
                timer.toc() 
                    
            # run tracking
            online_tlwhs = []
            online_ids = []
            online_scores = []
            # for t in online_targets:
            #     tlwh =  t.tlwh# t[:4]
            #     tid =  t.track_id# int(t[4])
            #     # vertical = tlwh[2] / tlwh[3] > 1.6
            #     if tlwh[2] * tlwh[3] > self.cfg.MODEL.MDR.MIX_AREA :#and not vertical
            #         online_tlwhs.append(tlwh)
            #         online_ids.append(tid)
            #         online_scores.append(t.score) #t[5]
        
            for t in online_targets:
                if 'dance' in img_file_name:
                    NOT_KEEP_LOST = True
                else:
                    NOT_KEEP_LOST = t['age'] > 10
                if t['active'] == 0 and NOT_KEEP_LOST: 
                    continue
                tlwh = self._xyxy2tlwh(t['bbox'])
                tid = t['track_id']
                vertical = tlwh[2] / tlwh[3] > 1.6
                if 'dance' in img_file_name:
                    vertical = False
                if tlwh[2] * tlwh[3] > self.cfg.MODEL.MDR.MIX_AREA and not vertical: 
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t['score'])
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))
            
            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results) 
                timer_avgs.append(timer.average_time)
                timer_calls.append(timer.calls)
                
        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))
        
    def _xyxy2tlwh(self, box):
        new_box = box.copy()
        new_box[2] = box[2] - box[0]
        new_box[3] = box[3] - box[1]
        return new_box