#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

DIY Agent Configuration.
峡谷追猎 DIY 智能体配置。

全面考虑数据协议参数。
"""


import numpy as np


class Config:

    USE_CNN = False
    VIEW_SIZE = 50 if USE_CNN else 0

    FEATURE_VECTOR_SHAPE = (68,)
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    ACTION_SHAPE = (16,)
    VALUE_SHAPE = (1,)

    GAMMA = 0.995
    LAMDA = 0.95
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    ENTROPY_COEFF = 0.001
    MAX_GRAD_NORM = 0.5

    START_LR = 3e-4
    END_LR = 1e-5
    DECAY_TYPE = "linear"

    VALUE_LOSS_COEFF = 0.5
    ENTROPY_LOSS_COEFF = 0.001

    EPSILON = 1e-8
    MAX_BUFFER_SIZE = 10000
    BATCH_SIZE = 256
    EPOCH = 10
