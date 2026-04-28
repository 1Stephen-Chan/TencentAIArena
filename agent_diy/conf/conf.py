#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase DIY PPO agent.
"""


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
    FLASH_EVAL_RANGE = 8
    VISIT_CELL_SIZE = 8.0
    MAX_VISIT_COUNT = 6.0
    MAX_REPEAT_STEPS = 8.0
    MAX_STUCK_STEPS = 8.0
    MAX_LOST_MONSTER_STEPS = 10.0
    STUCK_MOVE_THRESHOLD = 0.35
    TREASURE_SLOT_NUM = 4
    BUFF_SLOT_NUM = 2

    # -------------------------
    # Feature definition
    # 状态特征(self + monster + target + map) + 动作先验
    # -------------------------
    # self = 基础状态(含视野状态) + 反绕圈历史 + 8个方向访问新颖度
    SELF_DIM = 27
    MONSTER_DIM = 20
    # target = 4个宝箱槽位 * 7 + 2个固定buff槽位 * 8
    TARGET_DIM = 44
    LOCAL_MAP_VIEW_SIZE = 21
    LOCAL_MAP_CHANNELS = 1
    MAP_DIM = LOCAL_MAP_VIEW_SIZE * LOCAL_MAP_VIEW_SIZE * LOCAL_MAP_CHANNELS
    LEGAL_DIM = 16  # 合法动作 mask 单独传入，不拼接进 obs
    ACTION_PRIOR_DIM = 48  # safety + treasure + explore 三组动作先验，各16维

    STATE_FEATURES = [
        SELF_DIM,
        MONSTER_DIM,
        TARGET_DIM,
        MAP_DIM,
    ]
    STATE_DIM = sum(STATE_FEATURES)

    FEATURES = STATE_FEATURES + [ACTION_PRIOR_DIM]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)

    DIM_OF_OBSERVATION = FEATURE_LEN
    FEATURE_VECTOR_SHAPE = (FEATURE_LEN,)

    NON_MAP_STATE_DIM = SELF_DIM + MONSTER_DIM + TARGET_DIM
    MAP_START = NON_MAP_STATE_DIM
    MAP_END = MAP_START + MAP_DIM

    # 模型中动作先验特征的切片区间
    ACTION_PRIOR_START = MAP_END
    ACTION_PRIOR_END = ACTION_PRIOR_START + ACTION_PRIOR_DIM
    NON_SPATIAL_DIM = NON_MAP_STATE_DIM + ACTION_PRIOR_DIM

    # 向后兼容旧命名
    ACTION_EVAL_DIM = ACTION_PRIOR_DIM
    ACTION_EVAL_START = ACTION_PRIOR_START
    ACTION_EVAL_END = ACTION_PRIOR_END

    # 为兼容旧模板保留
    USE_CNN = True
    VIEW_SIZE = LOCAL_MAP_VIEW_SIZE // 2
    FEATURE_IMAGE_SHAPE = (LOCAL_MAP_CHANNELS, LOCAL_MAP_VIEW_SIZE, LOCAL_MAP_VIEW_SIZE)

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

    # 为兼容旧模板保留
    VALUE_LOSS_COEFF = VF_COEF
    ENTROPY_LOSS_COEFF = BETA_START
