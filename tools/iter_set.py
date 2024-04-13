import numpy as np
sample_steps = 4 # 采样的阶段数 -- 例如 2 3 4 5 采样的阶段数为4
data_frames_num = 41796 # 训练数据集的 总图片数
num_videos = 40 # 训练数据集的 总视频数
sample_len_list = [2,4,6,8] # 每一个采样阶段的 采样clip长度 
epochs_per_steps = [5, 5, 5, 5] # 每一个采样阶段下 要训练多少  epoch -- 采样clip长度为2时，对应训练模型5个epoch
max_sample_dist = 10 # 采样clip中 帧与帧之间的 最大采样间隔
batch = 4 # 总的训练批量数

for i in range(sample_steps):
    print("step: {}".format(i))
    curr_clip_len = sample_len_list[i]
    print("curr clip len: {}".format(curr_clip_len))
    curr_drop_frames_per_video = (curr_clip_len - 1) * max_sample_dist
    curr_tol_drop_frames  = curr_drop_frames_per_video * num_videos
    curr_data_frames = data_frames_num - curr_tol_drop_frames
    print("curr dataset frames: {}".format(curr_data_frames))
    print("curr step iterations per epoch: {}".format(np.ceil(curr_data_frames / batch)))
    print("curr step total of iterations: {}".format( np.ceil(curr_data_frames / batch) * epochs_per_steps[i]))
    print("================================================")
    