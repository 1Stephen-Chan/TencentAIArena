#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


# Configuration
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:
    # -------------------------
    # Environment constants
    # -------------------------
    MAP_SIZE = 128.0
    MAP_DIAG = 181.0
    MAX_MONSTER_SPEED = 3.0
    MAX_FLASH_CD = 2000.0
    MAX_BUFF_DURATION = 50.0
    MAX_STEP = 2000.0
    EXPLORE_CELL_SIZE = 6.0
    STUCK_MOVE_THRESHOLD = 0.75
    MONSTER_SPEEDUP_THRESHOLD = 1.5

    # -------------------------
    # Feature definition
    # self(10) + monster(20) + target(24) + map(25) + legal(16) + action_eval(32) + risk(1)
    # -------------------------
    SELF_DIM = 10
    MONSTER_DIM = 20
    TARGET_DIM = 24
    MAP_DIM = 25
    LEGAL_DIM = 16
    ACTION_EVAL_DIM = 32
    RISK_DIM = 1

    FEATURES = [
        SELF_DIM,
        MONSTER_DIM,
        TARGET_DIM,
        MAP_DIM,
        LEGAL_DIM,
        ACTION_EVAL_DIM,
        RISK_DIM,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)

    DIM_OF_OBSERVATION = FEATURE_LEN
    FEATURE_VECTOR_SHAPE = (FEATURE_LEN,)

    # Indices for feature slicing in model
    ACTION_EVAL_START = SELF_DIM + MONSTER_DIM + TARGET_DIM + MAP_DIM + LEGAL_DIM
    ACTION_EVAL_END = ACTION_EVAL_START + ACTION_EVAL_DIM

    # Kept for compatibility with older templates
    USE_CNN = False
    VIEW_SIZE = 0
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    # -------------------------
    # Action / value shape
    # -------------------------
    ACTION_NUM = 16
    ACTION_SHAPE = (ACTION_NUM,)

    VALUE_NUM = 1
    VALUE_SHAPE = (VALUE_NUM,)

    # -------------------------
    # PPO hyperparameters
    # -------------------------
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 3e-4
    START_LR = INIT_LEARNING_RATE_START
    BETA_START = 0.001
    BETA_END = 0.0002
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
    PPO_EPOCHS = 2

    # Kept for compatibility with older templates
    VALUE_LOSS_COEFF = VF_COEF
    ENTROPY_LOSS_COEFF = BETA_START
