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
from agent_diy.conf.conf import Config


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
    # 2) Safety shaping (survival first, but reward escaping)
    # ----------------------------
    prev_dist = float(remain_info.get("min_monster_dist", Config.MAP_DIAG))
    cur_dist = float(_remain_info.get("min_monster_dist", prev_dist))
    dist_delta = np.clip((cur_dist - prev_dist) / 20.0, -1.0, 1.0)

    # 改进：只在安全区奖励拉开距离，危险区鼓励逃跑
    if cur_dist > 3.0:
        reward += 0.18 * float(dist_delta)
    elif cur_dist <= 1.5:
        reward -= 0.75
        # 危险区内逃跑给奖励，鼓励边逃边吃
        if dist_delta > 0:
            reward += 0.15
    elif cur_dist <= 3.0:
        reward -= 0.32
        # 中危险区逃跑也给奖励
        if dist_delta > 0:
            reward += 0.10

    # ----------------------------
    # 3) Event shaping: treasure, buff, flash quality
    # ----------------------------
    treasure_gain = int(_remain_info.get("treasure_cnt", 0) - remain_info.get("treasure_cnt", 0))
    buff_gain = int(_remain_info.get("buff_cnt", 0) - remain_info.get("buff_cnt", 0))
    entered_new_cell = float(_remain_info.get("entered_new_cell_flag", 0.0))
    current_cell_visits = int(_remain_info.get("current_cell_visits", 1))
    new_buff_discovery = int(_remain_info.get("new_buff_discovery", 0))
    new_treasure_discovery = int(_remain_info.get("new_treasure_discovery", 0))
    phase_enter_reward = float(_remain_info.get("phase_enter_reward", 0.0))
    phase_move_reward_cap = float(_remain_info.get("phase_move_reward_cap", 0.05))
    phase_repeat_penalty = float(_remain_info.get("phase_repeat_penalty", 0.0))
    phase_buff_discovery_reward = float(_remain_info.get("phase_buff_discovery_reward", 0.0))
    phase_treasure_discovery_reward = float(_remain_info.get("phase_treasure_discovery_reward", 0.0))

    # 宝箱奖励：危险时吃到宝箱给额外奖励（鼓励顺路吃）
    if treasure_gain > 0:
        if cur_dist <= 1.5:
            # 已处于贴脸危险区时，吃到宝箱不应鼓励“赌命换分”。
            reward -= 0.20 * treasure_gain
        elif cur_dist <= 3.0:
            # 中危险区不再额外奖励，保留官方分数增量即可。
            reward += 0.0
        else:
            # 安全区吃宝箱：正常奖励
            reward += 0.35 * treasure_gain

    # Buff奖励
    if buff_gain > 0:
        reward += 0.25 * buff_gain

    if new_buff_discovery > 0:
        reward += phase_buff_discovery_reward * new_buff_discovery

    if new_treasure_discovery > 0:
        reward += phase_treasure_discovery_reward * new_treasure_discovery

    # ----------------------------
    # 4) Exploration reward: anti-stuck & anti-spinning
    # ----------------------------
    prev_x = float(remain_info.get("hero_x", 0.0))
    prev_z = float(remain_info.get("hero_z", 0.0))
    cur_x = float(_remain_info.get("hero_x", prev_x))
    cur_z = float(_remain_info.get("hero_z", prev_z))
    moved = abs(cur_x - prev_x) + abs(cur_z - prev_z)

    # 检测卡住状态：连续多步移动距离很小
    prev_prev_x = float(remain_info.get("prev_x", prev_x))
    prev_prev_z = float(remain_info.get("prev_z", prev_z))
    prev_moved = abs(prev_x - prev_prev_x) + abs(prev_z - prev_prev_z)

    # 困境检测：连续两步都几乎不动
    is_stuck = (moved < 0.2 and prev_moved < 0.2)

    # 闪现逻辑
    flash_gain = int(_remain_info.get("flash_cnt", 0) - remain_info.get("flash_cnt", 0))
    if flash_gain > 0:
        # 改进闪现逻辑：看结果而不是简单判断
        if prev_dist <= 4.0:
            # 危险时闪现
            if cur_dist > prev_dist + 3.0:
                reward += 0.50  # 成功逃脱（提高奖励）
            elif cur_dist > prev_dist:
                reward += 0.20  # 稍微拉开
            else:
                reward -= 0.30  # 闪现失败
        elif is_stuck:
            # 卡住时闪现脱困（新增）
            if moved > 3.0:
                reward += 0.60  # 成功脱困
            else:
                reward -= 0.20  # 闪现了但还是卡住
        elif treasure_gain > 0:
            if cur_dist > 3.0 and cur_dist >= prev_dist:
                reward += 0.05  # 安全地闪现顺路吃箱，给极小正反馈
            else:
                reward -= 0.12  # 抢箱但落点仍危险，轻惩罚
        else:
            reward -= 0.10  # 浪费闪现（降低惩罚）

        if cur_dist <= 1.5:
            reward -= 0.60  # 闪现后贴脸，显式打压高危落点

    # 原地不动惩罚（撞墙或卡住）
    if moved < 0.1:
        if is_stuck:
            # 连续卡住：重罚，逼迫使用闪现
            reward -= 0.50
        else:
            # 单次撞墙：轻罚
            reward -= 0.20

    # 检测打转行为：动作与上次相反或连续小幅移动
    cur_action = int(_remain_info.get("last_action", -1))
    prev_action = int(remain_info.get("last_action", -1))
    prev_prev_action = int(remain_info.get("prev_action", prev_action))

    if cur_action >= 0 and prev_action >= 0:
        # 检测反向动作（打转）：例如 0(E) 和 4(W)，1(NE) 和 5(SW)
        if (cur_action < 8 and prev_action < 8 and
            abs((cur_action % 8) - (prev_action % 8)) == 4):
            reward -= 0.40  # 提高惩罚

            # 如果连续打转（A->B->A），加倍惩罚
            if prev_prev_action >= 0 and cur_action == prev_prev_action:
                reward -= 0.30  # 额外惩罚

        # 小幅移动但频繁换方向（低效探索）
        # 改进：只在安全时惩罚，危险时允许灵活走位
        if moved < 0.3 and cur_action != prev_action and cur_dist > 4.0:
            reward -= 0.10

        # 保持方向奖励：连续朝同一方向移动
        if cur_action == prev_action and moved > 0.5:
            if entered_new_cell > 0.5 or current_cell_visits <= 2 or cur_dist <= 4.0:
                reward += 0.03  # 只在有效推进时保留小奖励

    if entered_new_cell > 0.5 and cur_dist > 3.5:
        reward += phase_enter_reward

    # 探索奖励：从“走得远”转向“走进新区域”
    if cur_dist > 4.0:
        if current_cell_visits <= 1:
            novelty_factor = 1.0
        elif current_cell_visits <= 2:
            novelty_factor = 0.55
        else:
            novelty_factor = 0.20

        exploration_bonus = np.clip(moved / 2.0, 0.0, phase_move_reward_cap) * novelty_factor
        reward += exploration_bonus

        if current_cell_visits >= 3 and treasure_gain <= 0 and buff_gain <= 0 and phase_repeat_penalty > 0.0:
            reward -= phase_repeat_penalty

    # 更新历史位置（用于下一步检测）
    _remain_info["prev_x"] = cur_x
    _remain_info["prev_z"] = cur_z
    _remain_info["prev_action"] = cur_action

    # ----------------------------
    # 5) Terminal shaping
    # ----------------------------
    if terminated:
        reward -= 8.0
    elif truncated:
        # 根据收集率给奖励，避免"苟活"策略
        treasures_collected = int(_remain_info.get("treasure_cnt", 0))
        total_treasure = 10.0  # 默认值
        collect_rate = treasures_collected / max(1.0, total_treasure)
        reward += 3.0 + 6.0 * collect_rate  # 3~9分

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
