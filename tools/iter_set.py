import numpy as np
sample_steps = 4
data_frames_num = 230000
num_videos = 5
sample_len_list = [2,3,4,5]
epochs_per_steps = [10,10,10,10]
max_sample_dist = 10
batch = 4

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
    