import itertools
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *

from tracking_utils.log import logger
from tracking_utils.utils import *
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, buffer_size=30):

        # wait activate
        self.xywh_omemory = deque([], maxlen=buffer_size)
        self.xywh_pmemory = deque([], maxlen=buffer_size)
        self.xywh_amemory = deque([], maxlen=buffer_size)

        self.conds = deque([], maxlen = 5)


        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict_diff(stracks, model, img_w, img_h, use_diffmot):
        if len(stracks) > 0:
            dets = np.asarray([st.xywh.copy() for st in stracks]).reshape(-1, 4)

            dets[:, 0::2] = dets[:, 0::2] / img_w
            dets[:, 1::2] = dets[:, 1::2] / img_h

            conds = [st.conds for st in stracks]
            if use_diffmot:
                multi_track_pred = model.generate(conds, sample=1, bestof=True, img_w=img_w, img_h=img_h)
                track_pred = multi_track_pred.mean(0)
                track_pred = track_pred + dets
            else:
                track_pred = model.generate(conds, sample=1, bestof=True, img_w=img_w, img_h=img_h)
            
            track_pred[:, 0::2] = track_pred[:, 0::2] * img_w
            track_pred[:, 1::2] = track_pred[:, 1::2] * img_h
            track_pred[:, 0] = track_pred[:, 0] - track_pred[:, 2] / 2
            track_pred[:, 1] = track_pred[:, 1] - track_pred[:, 3] / 2


            for i, st in enumerate(stracks):
                st._tlwh = track_pred[i]
                st.xywh_pmemory.append(st.xywh.copy())
                st.xywh_amemory.append(st.xywh.copy())

                tmp_delta_bbox = st.xywh.copy() - st.xywh_amemory[-2].copy()
                tmp_conds = np.concatenate((st.xywh.copy(), tmp_delta_bbox))
                st.conds.append(tmp_conds)


    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_pmemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())


        delta_bbox = self.xywh.copy() - self.xywh.copy()
        tmp_conds = np.concatenate((self.xywh.copy(), delta_bbox))
        self.conds.append(tmp_conds)

    def re_activate(self, new_track, frame_id, new_id=False):
        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory[-1] = self.xywh.copy()

        tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
        tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
        self.conds[-1] = tmp_conds

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory[-1] = self.xywh.copy()

        if self.is_activated == True:
            tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
            tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
            self.conds[-1] = tmp_conds
        else:
            tmp_delta_bbox = self.xywh.copy() - self.xywh_omemory[-2].copy()
            tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
            self.conds[-1] = tmp_conds

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        # ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class BYTETracker(object):
    def __init__(self, config, frame_rate=30):
        self.config = config
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = self.config.high_thres + 0.1
        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size

        self.config = config
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

    def update(self, dets_norm, model, img_w, img_h,img=None):
        self.model = model
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        dets = dets_norm.copy()
        dets[:, 2] = dets[:, 0] + dets[:, 2]
        dets[:, 3] = dets[:, 1] + dets[:, 3]
        remain_inds = dets[:, 4] > self.config.high_thres
        inds_low = dets[:, 4] > self.config.low_thres
        inds_high = dets[:, 4] < self.config.high_thres
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4]) for
                          tlbrs in dets[:, :5]]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict_diff(strack_pool, self.model, img_w, img_h, self.config.use_diffmot)
        
        #vis
        # import pdb;pdb.set_trace()
        # curr_im = img.copy()
        # for track in strack_pool:
        #     x1,y1,x2,y2 = track.tlbr
        #     if track.state == TrackState.Tracked:
        #         pass
        #     else:
        #         import pdb;pdb.set_trace()
        #         cv2.rectangle(curr_im, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
        # import pdb;pdb.set_trace()
        # cv2.imwrite('./sss.jpg', curr_im)
        
        
        dists = matching.iou_distance(strack_pool, detections)
        # if "mot17" in self.config.exp_name:
        #     dists = matching.fuse_score(dists, detections)
        #     thresh = 0.9
        # elif 'mot20' in self.config.exp_name:
        #     thresh = 0.7
        # else:
        #     thresh = 0.8
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh = 0.8)
 
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
       
        ''' Step 3: Second association, with low score detection boxes'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], buffer_size=30) for
                                 tlbrs in dets_second[:, :5]]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            # track = strack_pool[it]
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if "mot17" in self.config.exp_name:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # track.activate(self.kalman_filter, self.frame_id)
            track.activate(self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
          
 
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    # pairs = np.where(pdist < 0.)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
