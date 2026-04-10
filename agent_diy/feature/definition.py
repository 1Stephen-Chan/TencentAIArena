#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config


ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
)


ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
)


SampleData = create_cls(
    "SampleData",
    obs=Config.FEATURE_VECTOR_SHAPE[0],
    legal_action=Config.ACTION_SHAPE[0],
    act=1,
    prob=Config.ACTION_SHAPE[0],
    reward=1,
    advantage=1,
    value=1,
    reward_sum=1,
    done=1,
    next_value=1,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    reward = 0.0
    if terminated:
        reward -= 5.0
    elif truncated:
        reward += 3.0
    return np.array([reward], dtype=np.float32)
