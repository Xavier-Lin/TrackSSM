
from torch.utils.data  import Dataset
import numpy as np

import os
import glob


class DiffMOTDataset(Dataset):
    def __init__(self, path, config=None):
        self.config = config
        self.interval = self.config.interval + 1 # 6

        self.trackers = {}# 每一个视频序列中 所有轨迹对应的txt文件 其中 一条轨迹对应一个 txt文件
        self.nsamples = {}# 每一个视频序列中，每一条轨迹，的时序采样数量 -- 例如对于某一条 时长度 为1000的轨迹，我们需要对该轨迹在1000中采样798个！
        self.nS = 0# 轨迹 数据集 总的 时序采样数量 --- 数据集中 所有轨迹的 时序采样数量 之和

        self.nds = {}
        self.cds = {}# 整个 数据集中，所有轨迹的 时序采样数量 的累加
        if os.path.isdir(path):
            self.seqs = [s for s in os.listdir(path)]
            self.seqs.sort()
            lastindex = 0
            for seq in self.seqs:
                trackerPath = os.path.join(path + "/" + seq, "img1/*.txt")
                self.trackers[seq] = sorted(glob.glob(trackerPath))
                self.nsamples[seq] = {}
                for i, pa in enumerate(self.trackers[seq]):
                    track_of_len = len(np.loadtxt(pa, dtype=np.float32).reshape(-1,7))
                    if track_of_len <= self.interval:
                        import pdb;pdb.set_trace()
                        lzl = 0
                    self.nsamples[seq][i] = track_of_len - self.interval #每一个 视频序列中的 每一条轨迹 的 采样数量
                    self.nS += self.nsamples[seq][i]


                self.nds[seq] = [x for x in self.nsamples[seq].values()]
                self.cds[seq] = [sum(self.nds[seq][:i]) + lastindex for i in range(len(self.nds[seq]))]
                lastindex = self.cds[seq][-1] + self.nds[seq][-1]

        print('=' * 80)
        print('dataset summary')
        print(self.nS)
        print('=' * 80)
  
    def __getitem__(self, files_index):# 某一个视频中 的某一个轨迹 的某一时刻的索引

        for i, seq in enumerate(self.cds):# 遍历视频序列
            if files_index >= self.cds[seq][0]:# 时刻索引 是否大于 该视频序列的 第一条轨迹 -- 对应的开始时刻索引
                ds = seq # 如果大于 则说明 提取的 时刻索引 --- 属于 这一 视频序列 范围内 ds：确定 提取时刻索引 对应的 视频序列
                for j, c in enumerate(self.cds[seq]):# 遍历该视频序列中，每一条轨迹对应的开始时刻索引
                    if files_index >= c:# 是否 提取的 时刻索引 >= 该轨迹的开始时刻索引
                        trk = j
                        start_index = c
                    else:
                        break
            else:
                break
        # 最终获得 提取时刻索引files_index  对应所属的视频序列名称ds  以及 所属轨迹id - trk    以及    所属轨迹开始的时刻索引 start_index
        track_path = self.trackers[ds][trk]# 提取 对应视频中 某一条轨迹的 对应标注的txt文件 
        track_gt = np.loadtxt(track_path, dtype=np.float32)# 读取该轨迹的标注

        init_index = files_index - start_index # 获得 提取时刻索引 在 该轨迹长度中的 时序位置 --- 例如 该时刻索引 对应 某一条轨迹长度上的 第34个位置

        cur_index = init_index + self.interval # 对应轨迹长度上的位置 + 采样长度 + 1 -- 为何要+1 这是因为要提取六个帧的信息 -- 前五个是历史帧 第六作为当前帧
        cur_gt = track_gt[cur_index] # 得到 对应轨迹 定义时刻的当前帧【40】的索引
        cur_bbox = cur_gt[2:6]# 得到 对应轨迹 定义时刻的当前帧【40】bbox

        boxes = [track_gt[init_index + tmp_ind][2:6] for tmp_ind in range(self.interval)]# 34 35 36 37 38 39 -- 【40】这里的【40】 用作训练阶段的gt标注
        delt_boxes = [boxes[i+1] - boxes[i] for i in range(self.interval - 1)] # [1] - [0] [2] - [1] [3] - [2] [4] - [3] [5] - [4] -- 得到差分 detla_bbox
        conds = np.concatenate((np.array(boxes)[1:], np.array(delt_boxes)), axis=1)#。前5个时刻的 bbox ｜ 前5个时刻的 差分运动 
        # 通过 每条轨迹的 前5个时刻的 运动 以及 位置信息  去预测  当前帧时刻【40】的 轨迹位置信息 --- 这就是基本的训练思路了！！！

        delt = cur_bbox - boxes[-1]
        ret = {"cur_gt": cur_gt, "cur_bbox": cur_bbox, "condition": conds, "delta_bbox": delt}
        # 返回 对应轨迹的 
        # cur_gt = 取定当前帧时刻的 fid bbox[归一化的cx cy w h] vis
        # cur_bbox = 取定当前帧时刻的 归一化bbox
        # conds = 取定前五个时刻的 归一化bbox ｜ 前5个时刻的 差分运动 横向维度为8 -- 与KF的维度相一致
        # delt = 取定 当前帧时刻 与 前一帧时刻的 bbox_delta
        return ret

    def __len__(self):
        return self.nS

# if __name__ == "__main__":
#     data_path = '/mnt/8T/home/estar/data/DanceTrack/trackers_gt_GSI'
#     a = DiffMOTDataset_longterm(data_path)
#     b = a[700]
#     pass












