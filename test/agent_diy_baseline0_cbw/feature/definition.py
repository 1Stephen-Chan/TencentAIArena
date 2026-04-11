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

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData用于在aisrv和learner之间传递训练样本
# 必须使用整数定义维度（不能用None），框架层会自动生成FIELD_DIMS并处理序列化
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


def reward_shaping(
    frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs
):
    reward = 0.0

    if remain_info is None or _remain_info is None:
        if terminated:
            reward -= 8.0
        elif truncated:
            reward += 4.0
        return np.array([reward], dtype=np.float32)

    speedup_active = bool(_remain_info.get("monster_speedup_seen", 0.0) >= 0.5)
    hero_buff_active = bool(_remain_info.get("hero_buff_active", 0.0) >= 0.5)
    visible_monster_cnt = int(_remain_info.get("visible_monster_cnt", 0))

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
    # 2) Phase-aware dense shaping
    # ----------------------------
    prev_dist = float(remain_info.get("min_monster_dist", Config.MAP_DIAG))
    cur_dist = float(_remain_info.get("min_monster_dist", prev_dist))
    dist_delta = np.clip((cur_dist - prev_dist) / 20.0, -1.0, 1.0)

    prev_treasure_dist = float(remain_info.get("nearest_treasure_dist_norm", 1.0))
    cur_treasure_dist = float(_remain_info.get("nearest_treasure_dist_norm", prev_treasure_dist))
    treasure_dist_delta = np.clip(prev_treasure_dist - cur_treasure_dist, -1.0, 1.0)

    prev_buff_dist = float(remain_info.get("nearest_buff_dist_norm", 1.0))
    cur_buff_dist = float(_remain_info.get("nearest_buff_dist_norm", prev_buff_dist))
    buff_dist_delta = np.clip(prev_buff_dist - cur_buff_dist, -1.0, 1.0)

    is_new_area = float(_remain_info.get("is_new_area", 0.0))

    if speedup_active:
        reward += 0.28 * float(dist_delta)
        reward += 0.04 * float(treasure_dist_delta)
        if not hero_buff_active:
            reward += 0.24 * float(buff_dist_delta)

        if cur_dist <= 1.5:
            reward -= 1.10
        elif cur_dist <= 3.0:
            reward -= 0.55
        elif cur_dist <= 5.0:
            reward -= 0.15
    else:
        reward += 0.08 * float(dist_delta)
        reward += 0.22 * float(treasure_dist_delta)
        if not hero_buff_active:
            reward += 0.12 * float(buff_dist_delta)
        reward += 0.18 * is_new_area

        if visible_monster_cnt <= 0 and is_new_area > 0.5:
            reward += 0.05

        if cur_dist <= 1.5:
            reward -= 0.60
        elif cur_dist <= 3.0:
            reward -= 0.24

    # ----------------------------
    # 3) Event shaping: treasure, buff, flash quality
    # ----------------------------
    treasure_gain = int(_remain_info.get("treasure_cnt", 0) - remain_info.get("treasure_cnt", 0))
    buff_gain = int(_remain_info.get("buff_cnt", 0) - remain_info.get("buff_cnt", 0))
    if treasure_gain > 0:
        reward += (1.15 if not speedup_active else 0.55) * treasure_gain
    if buff_gain > 0:
        reward += (0.60 if not speedup_active else 1.20) * buff_gain

    flash_gain = int(_remain_info.get("flash_cnt", 0) - remain_info.get("flash_cnt", 0))
    if flash_gain > 0:
        if speedup_active:
            if prev_dist <= 5.0 and cur_dist > prev_dist + 0.5:
                reward += 0.45 * flash_gain
            elif cur_dist <= prev_dist:
                reward -= 0.18 * flash_gain
            else:
                reward -= 0.05 * flash_gain
        else:
            if prev_dist <= 3.5 and cur_dist > prev_dist + 0.5:
                reward += 0.18 * flash_gain
            else:
                reward -= 0.45 * flash_gain

    # ----------------------------
    # 4) Anti-stuck / anti-loop shaping
    # ----------------------------
    prev_x = float(remain_info.get("hero_x", 0.0))
    prev_z = float(remain_info.get("hero_z", 0.0))
    cur_x = float(_remain_info.get("hero_x", prev_x))
    cur_z = float(_remain_info.get("hero_z", prev_z))
    moved = abs(cur_x - prev_x) + abs(cur_z - prev_z)
    if moved < 0.1:
        reward -= 0.08 if speedup_active else 0.06

    visit_count = int(_remain_info.get("visit_count", 1))
    stuck_steps = int(_remain_info.get("stuck_steps", 0))
    if visit_count >= 3 and is_new_area < 0.5 and not speedup_active:
        reward -= min(0.04 * (visit_count - 2), 0.18)
    if stuck_steps >= 2:
        reward -= min(0.05 * stuck_steps, 0.35 if speedup_active else 0.25)

    prev_action = int(remain_info.get("last_action", -1))
    cur_action = int(_remain_info.get("last_action", -1))
    if _is_opposite_action(prev_action, cur_action) and moved < 1.5:
        reward -= 0.12 if visible_monster_cnt <= 0 else 0.08
    if visible_monster_cnt <= 0 and moved < 1.0 and not speedup_active:
        reward -= 0.04

    # ----------------------------
    # 5) Terminal shaping
    # ----------------------------
    if terminated:
        reward -= 9.0 if speedup_active else 8.0
    elif truncated:
        reward += 4.0

    return np.array([reward], dtype=np.float32)


def _is_opposite_action(prev_action, cur_action):
    if prev_action < 0 or cur_action < 0:
        return False
    if prev_action >= Config.ACTION_NUM or cur_action >= Config.ACTION_NUM:
        return False
    return ((int(prev_action) % 8) - (int(cur_action) % 8)) % 8 == 4

def _scalar(v):
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return 0.0
        return float(v.reshape(-1)[0])
    try:
        return float(v)
    except Exception:
        return 0.0

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
            list_sample_data[i].next_value = np.array(
                [_scalar(list_sample_data[i + 1].value)], dtype=np.float32
            )

    list_sample_data[-1].next_value = np.zeros(1, dtype=np.float32)

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
