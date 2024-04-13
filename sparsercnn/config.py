# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_MDR_config(cfg):
    """
    Add config for MDR.
    """    
    ## Detector
    cfg.MODEL.MDR = CN()
    cfg.MODEL.MDR.IS_FIX = False
    cfg.MODEL.MDR.NUM_CLASSES = 1
    cfg.MODEL.MDR.NUM_PROPOSALS = 500
    
    # RCNN Head.
    cfg.MODEL.MDR.NHEADS = 8
    cfg.MODEL.MDR.DROPOUT = 0.0
    cfg.MODEL.MDR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MDR.ACTIVATION = 'relu'
    cfg.MODEL.MDR.HIDDEN_DIM = 256
    cfg.MODEL.MDR.NUM_CLS = 1
    cfg.MODEL.MDR.NUM_REG = 3
    cfg.MODEL.MDR.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.MDR.NUM_DYNAMIC = 2
    cfg.MODEL.MDR.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.MDR.CLASS_WEIGHT = 2.0
    cfg.MODEL.MDR.GIOU_WEIGHT = 2.0
    cfg.MODEL.MDR.L1_WEIGHT = 5.0
    cfg.MODEL.MDR.DEEP_SUPERVISION = True
    cfg.MODEL.MDR.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.MDR.USE_FOCAL = True
    cfg.MODEL.MDR.ALPHA = 0.25
    cfg.MODEL.MDR.GAMMA = 2.0
    cfg.MODEL.MDR.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    
    # Track decoder
    cfg.MODEL.MDR.IS_TRAIN = False
    cfg.MODEL.MDR.USE_FOCAL_R = True
    cfg.MODEL.MDR.BOX_RATIO = (0.5, 1.5)
    cfg.MODEL.MDR.TRACK_NUM_CLS = 1
    cfg.MODEL.MDR.MAX_FRAME_DIST = 3
    cfg.MODEL.MDR.SAMPLER_STEPS = (1, )# 
    cfg.MODEL.MDR.D_STATE = 256 # 16
    cfg.MODEL.MDR.EXPAND = 1 # 2
    
    
    # Train dataset
    cfg.DATASETS.TRAIN_DATA_IMG_PATH = ''
    cfg.DATASETS.TRAIN_DATA_ANN_PATH = ''
    cfg.DATASETS.TRAIN_DATA_SAMPLER_LEN = (1,) # 采样的视频帧clip长度
    
    
    # Test dataset name
    cfg.DATASETS.TEST_DATA = CN()
    cfg.DATASETS.TEST_DATA.DATA_NAME = 'MOT20'
    
    # tracker 
    cfg.MODEL.MDR.TRACKING = False
    cfg.MODEL.MDR.TRACKING_SCORE = 0.
    cfg.MODEL.MDR.MIX_AREA = 1
    cfg.MODEL.MDR.NEW_TRACKING_THRES = 0.
    
    
    
