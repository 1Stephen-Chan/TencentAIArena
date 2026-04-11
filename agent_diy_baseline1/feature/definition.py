#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and reward/sample processing for Gorge Chase DIY agent.
"""

import numpy as np

from common_python.utils.common_func import create_cls
from agent_diy_baseline1.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def _scalar(v):
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return 0.0
        return float(v.reshape(-1)[0])
    try:
        return float(v)
    except Exception:
        return 0.0


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    """Reward shaping aligned with official score: step score + treasure score.

    与官方计分一致：步数得分 + 宝箱得分，并增加安全塑形项提升训练稳定性。
    """
    reward = 0.0

    if remain_info is None or _remain_info is None:
        if terminated:
            reward -= 8.0
        elif truncated:
            reward += 4.0
        return np.array([reward], dtype=np.float32)

    # ----------------------------
    # 1) Score-aligned dense reward
    # ----------------------------
    prev_step_score = float(remain_info.get("step_score", 0.0))
    cur_step_score = float(_remain_info.get("step_score", prev_step_score))
    prev_treasure_score = float(remain_info.get("treasure_score", 0.0))
    cur_treasure_score = float(_remain_info.get("treasure_score", prev_treasure_score))
    prev_total_score = float(remain_info.get("total_score", prev_step_score + prev_treasure_score))
    cur_total_score = float(_remain_info.get("total_score", cur_step_score + cur_treasure_score))

    # Scale official score delta for PPO stability:
    # step_delta usually 1.5 -> +0.03, treasure_delta 100 -> +2.0
    reward += 0.02 * float(cur_total_score - prev_total_score)

    # ----------------------------
    # 2) Safety shaping (survival first)
    # ----------------------------
    prev_dist = float(remain_info.get("min_monster_dist", Config.MAP_DIAG))
    cur_dist = float(_remain_info.get("min_monster_dist", prev_dist))
    dist_delta = np.clip((cur_dist - prev_dist) / 20.0, -1.0, 1.0)
    reward += 0.18 * float(dist_delta)

    if cur_dist <= 1.5:
        reward -= 0.75
    elif cur_dist <= 3.0:
        reward -= 0.32

    # ----------------------------
    # 3) Event shaping: treasure, buff, flash quality
    # ----------------------------
    treasure_gain = int(_remain_info.get("treasure_cnt", 0) - remain_info.get("treasure_cnt", 0))
    buff_gain = int(_remain_info.get("buff_cnt", 0) - remain_info.get("buff_cnt", 0))
    if treasure_gain > 0:
        reward += 0.8 * treasure_gain
    if buff_gain > 0:
        reward += 0.25 * buff_gain

    flash_gain = int(_remain_info.get("flash_cnt", 0) - remain_info.get("flash_cnt", 0))
    if flash_gain > 0:
        # Reward defensive flash if danger is reduced
        if prev_dist <= 4.0 and cur_dist > prev_dist:
            reward += 0.30 * flash_gain
        else:
            reward -= 0.10 * flash_gain

    # Mild anti-stuck penalty
    prev_x = float(remain_info.get("hero_x", 0.0))
    prev_z = float(remain_info.get("hero_z", 0.0))
    cur_x = float(_remain_info.get("hero_x", prev_x))
    cur_z = float(_remain_info.get("hero_z", prev_z))
    moved = abs(cur_x - prev_x) + abs(cur_z - prev_z)
    if moved < 0.1:
        reward -= 0.02

    # ----------------------------
    # 4) Terminal shaping
    # ----------------------------
    if terminated:
        reward -= 8.0
    elif truncated:
        reward += 4.0

    return np.array([reward], dtype=np.float32)


def sample_process(list_sample_data):
    """Fill next_value then compute GAE.

    先填充 next_value，再计算 GAE。
    """
    if not list_sample_data:
        return list_sample_data

    for i in range(len(list_sample_data) - 1):
        done = _scalar(list_sample_data[i].done)
        if done >= 0.5:
            list_sample_data[i].next_value = np.zeros(1, dtype=np.float32)
        else:
            list_sample_data[i].next_value = np.array([_scalar(list_sample_data[i + 1].value)], dtype=np.float32)

    list_sample_data[-1].next_value = np.zeros(1, dtype=np.float32)

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        reward = _scalar(sample.reward)
        value = _scalar(sample.value)
        next_value = _scalar(sample.next_value)
        done = _scalar(sample.done)

        not_done = 1.0 - float(done >= 0.5)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lamda * not_done * gae

        sample.advantage = np.array([gae], dtype=np.float32)
        sample.reward_sum = np.array([gae + value], dtype=np.float32)
