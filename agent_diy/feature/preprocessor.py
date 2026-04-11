#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Structured Feature Preprocessor for Gorge Chase DIY Agent.
结构化特征预处理：
- 输出分组特征供网络处理
- 标量特征、实体特征、地图特征分离
"""

import numpy as np
from collections import deque
from agent_diy.conf.conf import Config


MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.map_info = None
        self.hero_pos = {"x": 0, "z": 0}
        self.last_min_monster_dist = 0.5
        self.last_treasure_dist = None
        self.last_buff_dist = None
        self.last_buff_count = 0
        self.history_positions = []
        self.visited_positions = set()

        self.monster_speedup_step = None
        self.second_monster_appear_step = None

        self.last_hero_pos = None
        self.last_flash_used = False
        self.stuck_counter = 0

        self.hero_center_x = 0
        self.hero_center_z = 0

        self.last_seen_min_dist = 1.0
        self.steps_since_last_seen = 0
        self.last_seen_pos = None
        self.estimated_threat_dist = 1.0
        self.last_estimated_threat_dist = 1.0

    def feature_process(self, env_obs, last_action):
        """
        Process env_obs into structured features.
        输出结构化特征字典供网络处理。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)
        self.map_info = map_info

        if map_info is not None:
            self.hero_center_x = len(map_info) // 2
            self.hero_center_z = len(map_info) // 2

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        self.hero_pos = hero_pos

        flash_cooldown = env_info.get("flash_cooldown", 2000)
        flash_count = env_info.get("flash_count", 0)

        # ========== 1. 英雄特征 [10维] ==========
        hero_feat = self._extract_hero_features(hero, hero_pos, flash_cooldown, flash_count)
        
        # ========== 2. 怪物特征 [2, 8维] ==========
        monsters = frame_state.get("monsters", [])
        monster_feats = self._extract_monster_features(monsters, hero_pos, map_info)
        
        # ========== 3. 宝箱特征 [4, 4维] ==========
        organs = frame_state.get("organs", [])
        treasures = [o for o in organs if o.get("sub_type") == 1]
        treasure_feats = self._extract_treasure_features(treasures, hero_pos, map_info)
        
        # ========== 4. Buff特征 [2, 4维] ==========
        buffs = [o for o in organs if o.get("sub_type") == 2]
        buff_feats = self._extract_buff_features(buffs, hero, hero_pos, map_info)
        
        # ========== 5. 进度特征 [6维] ==========
        progress_feat = self._extract_progress_features(env_info, monster_feats)
        
        # ========== 6. 地图特征 [1, 21, 21] ==========
        map_grid = self._extract_map_features(map_info, hero_pos)
        
        # ========== 7. 合法动作 [16维] ==========
        legal_action = self._process_legal_action(legal_act_raw)

        # 构建结构化特征字典
        structured_features = {
            'hero': hero_feat,
            'monsters': monster_feats,
            'treasures': treasure_feats,
            'buffs': buff_feats,
            'progress': progress_feat,
            'map': map_grid,
            'legal_action': legal_action,
        }

        # 计算奖励
        reward = self._compute_reward(
            env_obs, hero, monster_feats, treasure_feats,
            nearest_treasure=treasures[0] if treasures else None,
            nearest_buff=buffs[0] if buffs else None,
            buffs=buffs,
            current_action=last_action,
            terrain_feat=None,  # 地形信息已整合到map_grid
        )

        self.last_hero_pos = hero_pos.copy()
        return structured_features, legal_action, reward

    def _extract_hero_features(self, hero, hero_pos, flash_cooldown, flash_count):
        """提取英雄特征 [10维]."""
        return np.array([
            hero_pos["x"] / MAP_SIZE,                    # 位置X
            hero_pos["z"] / MAP_SIZE,                    # 位置Z
            hero["treasure_score"] / 1000.0,             # 宝箱得分
            hero["step_score"] / (self.max_step * 1.5),  # 步数得分
            1.0 if hero["flash_cooldown"] == 0 else 0.0, # 闪现是否可用
            hero["buff_remaining_time"] / MAX_BUFF_DURATION,  # buff剩余时间
            flash_cooldown / MAX_FLASH_CD,               # 闪现CD进度
            flash_count / 50.0,                          # 闪现次数
            self.stuck_counter / 10.0,                   # 卡住计数
            len(self.history_positions) / 100.0,         # 历史位置数
        ], dtype=np.float32)

    def _extract_monster_features(self, monsters, hero_pos, map_info):
        """提取怪物特征 [2, 8维]."""
        monster_feats = []
        for i in range(Config.MAX_MONSTERS):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                speed = m.get("speed", 1) / MAX_MONSTER_SPEED
                m_pos = m["pos"]
                
                real_dist_norm = self._compute_real_distance(hero_pos, m_pos, map_info, m)
                direction = m.get("hero_relative_direction", 0) / 8.0
                
                # 计算威胁度
                threat_level = self._compute_threat_level(real_dist_norm, speed, is_in_view)
                
                monster_feats.append([
                    real_dist_norm,      # 归一化距离
                    direction,           # 方向
                    speed,               # 速度
                    m_pos["x"] / MAP_SIZE,  # 位置X
                    m_pos["z"] / MAP_SIZE,  # 位置Z
                    is_in_view,          # 是否在视野
                    threat_level,        # 威胁度
                    1.0,                 # 存在标记
                ])
            else:
                # 填充空怪物
                monster_feats.append([0.0] * Config.MONSTER_FEATURE_DIM)
        
        return np.array(monster_feats, dtype=np.float32)

    def _extract_treasure_features(self, treasures, hero_pos, map_info):
        """提取宝箱特征 [4, 4维]."""
        treasure_feats = []
        for i in range(Config.MAX_TREASURES):
            if i < len(treasures):
                t = treasures[i]
                t_pos = t["pos"]
                real_dist_norm = self._compute_real_distance(hero_pos, t_pos, map_info, t)
                direction = t.get("hero_relative_direction", 0) / 8.0
                
                treasure_feats.append([
                    real_dist_norm,      # 归一化距离
                    direction,           # 方向
                    1.0,                 # 存在标记
                    1.0,                 # 价值（可扩展）
                ])
            else:
                treasure_feats.append([0.0] * Config.TREASURE_FEATURE_DIM)
        
        return np.array(treasure_feats, dtype=np.float32)

    def _extract_buff_features(self, buffs, hero, hero_pos, map_info):
        """提取Buff特征 [2, 4维]."""
        buff_feats = []
        for i in range(Config.MAX_BUFFS):
            if i < len(buffs):
                b = buffs[i]
                b_pos = b["pos"]
                real_dist_norm = self._compute_real_distance(hero_pos, b_pos, map_info, b)
                direction = b.get("hero_relative_direction", 0) / 8.0
                
                buff_feats.append([
                    real_dist_norm,      # 归一化距离
                    direction,           # 方向
                    1.0 if hero["buff_remaining_time"] > 0 else 0.0,  # 是否有buff
                    1.0,                 # 存在标记
                ])
            else:
                buff_feats.append([0.0] * Config.BUFF_FEATURE_DIM)
        
        return np.array(buff_feats, dtype=np.float32)

    def _extract_progress_features(self, env_info, monster_feats):
        """提取进度特征 [6维]."""
        monster_speedup_config = env_info.get("monster_speed", 500)
        monster_interval = env_info.get("monster_interval", 300)
        steps_until_speedup = max(0, monster_speedup_config - self.step_no) / monster_speedup_config
        second_monster_coming = 1.0 if self.step_no >= monster_interval else 0.0
        
        treasures_collected = env_info.get("treasures_collected", 0)
        collected_buff = env_info.get("collected_buff", 0)
        
        # 计算整体危险等级
        danger_level = self._compute_danger_level(monster_feats)
        
        return np.array([
            self.step_no / self.max_step,                # 进度
            1.0 if self.step_no > monster_speedup_config else 0.0,  # 是否加速
            danger_level,                                 # 危险等级
            steps_until_speedup,                          # 距离加速步数
            second_monster_coming,                        # 第二怪物是否出现
            (treasures_collected + collected_buff) / 15.0,  # 收集进度
        ], dtype=np.float32)

    def _extract_map_features(self, map_info, hero_pos):
        """提取局部地图特征（map_info已是以英雄为中心的视野栅格）."""
        if map_info is None:
            return np.zeros((Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        
        # map_info 本身就是以英雄为中心的视野范围
        # 直接转换为numpy数组，无需坐标转换
        map_h = len(map_info)
        map_w = len(map_info[0]) if map_info else 0
        
        # 创建固定大小的地图网格
        map_grid = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        
        # 将map_info复制到网格中心
        start_i = (Config.MAP_SIZE - map_h) // 2
        start_j = (Config.MAP_SIZE - map_w) // 2
        
        for i in range(min(map_h, Config.MAP_SIZE)):
            for j in range(min(map_w, Config.MAP_SIZE)):
                grid_i = start_i + i
                grid_j = start_j + j
                if 0 <= grid_i < Config.MAP_SIZE and 0 <= grid_j < Config.MAP_SIZE:
                    # 可行走=1.0, 障碍=0.0
                    map_grid[grid_i, grid_j] = 1.0 if map_info[i][j] != 0 else 0.0
        
        return map_grid.reshape(Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)

    def _compute_threat_level(self, dist_norm, speed, is_in_view):
        """计算怪物威胁度 [0, 1]."""
        if not is_in_view:
            return 0.0
        # 距离越近、速度越快，威胁越高
        threat = (1.0 - dist_norm) * 0.7 + speed * 0.3
        return min(threat, 1.0)

    def _compute_danger_level(self, monster_feats):
        """计算整体危险等级."""
        danger = 0.0
        for m in monster_feats:
            if len(m) >= 7:  # 确保有足够维度
                dist = m[0]
                threat = m[6] if len(m) > 6 else 0.0
                exists = m[7] if len(m) > 7 else 0.0
                if exists > 0.5:
                    danger += (1.0 - dist) * (1.0 + threat)
        return min(danger / 2.0, 1.0)  # 归一化到[0,1]

    def _compute_real_distance(self, hero_pos, target_pos, map_info, entity_data):
        """复合真实距离：BFS优先，视野外使用威胁估计."""
        if map_info is None:
            return 1.0

        is_in_view = entity_data.get("is_in_view", 0) if isinstance(entity_data, dict) else 1

        center = len(map_info) // 2
        hx = int(hero_pos["x"] - self.hero_center_x + center)
        hz = int(hero_pos["z"] - self.hero_center_z + center)
        tx = int(target_pos["x"] - self.hero_center_x + center)
        tz = int(target_pos["z"] - self.hero_center_z + center)

        if is_in_view:
            if (0 <= tx < len(map_info) and 0 <= tz < len(map_info[0])
                and 0 <= hx < len(map_info) and 0 <= hz < len(map_info[0])
                and map_info[tx][tz] != 0):
                bfs_dist = self._bfs((hx, hz), (tx, tz), map_info)
                if bfs_dist < float('inf'):
                    real_dist = min(bfs_dist / 42.0, 1.0)
                    self.last_seen_min_dist = real_dist
                    self.last_seen_pos = (tx, tz)
                    self.steps_since_last_seen = 0
                    return real_dist

        self.steps_since_last_seen += 1
        if hasattr(self, 'last_seen_pos') and self.last_seen_pos is not None:
            lx, lz = self.last_seen_pos
            if 0 <= lx < len(map_info) and 0 <= lz < len(map_info[0]):
                path_dist = self._bfs((hx, hz), (lx, lz), map_info)
                if path_dist < float('inf'):
                    self.estimated_threat_dist = min(path_dist / (len(map_info) * 0.5), 1.0)
                    return self.estimated_threat_dist
        self.estimated_threat_dist = 1.0
        return 1.0

    def _bfs(self, start, goal, map_info):
        """BFS计算最短路径."""
        from collections import deque
        if start == goal:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            (x, y), dist = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) == goal:
                    return dist + 1
                if (0 <= nx < len(map_info) and 0 <= ny < len(map_info[0])
                    and map_info[nx][ny] != 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        return float('inf')

    def _find_nearest_entity(self, entities, hero_pos):
        """找到最近的实体."""
        if not entities:
            return None
        nearest = min(entities, key=lambda e: e.get("hero_l2_distance", float('inf')))
        return nearest

    def _process_legal_action(self, legal_act_raw):
        """处理合法动作掩码."""
        legal_action = np.zeros(16, dtype=np.float32)
        for i in range(min(len(legal_act_raw), 16)):
            legal_action[i] = float(legal_act_raw[i])
        return legal_action

    def _is_exploring_repeatedly(self):
        """检测是否在重复探索."""
        if len(self.history_positions) < 10:
            return False
        recent = self.history_positions[-10:]
        if len(set(recent)) <= 3:
            return True
        return False

    def _compute_reward(self, env_obs, hero, monster_feats, treasure_feats,
                       nearest_treasure, nearest_buff, buffs, current_action, terrain_feat):
        """计算综合奖励."""
        rewards = []
        env_info = env_obs["observation"]["env_info"]

        # 生存奖励
        rewards.append(0.01)

        # 步数得分奖励
        current_step_score = hero["step_score"]
        if hasattr(self, 'last_step_score'):
            step_score_diff = current_step_score - self.last_step_score
            if step_score_diff > 0:
                rewards.append(step_score_diff * 0.01)
        self.last_step_score = current_step_score

        # 宝箱得分奖励
        current_treasure_score = hero["treasure_score"]
        if hasattr(self, 'last_treasure_score'):
            treasure_diff = current_treasure_score - self.last_treasure_score
            if treasure_diff > 0:
                rewards.append(treasure_diff * 1.0)
        self.last_treasure_score = current_treasure_score

        # 怪物距离奖励（使用第一只怪物的距离）
        if len(monster_feats) > 0 and len(monster_feats[0]) > 0:
            cur_min_dist = monster_feats[0][0]  # 第一只怪物的距离
            dist_delta = self.last_min_monster_dist - cur_min_dist
            dist_reward = 0.6 * dist_delta
            rewards.append(dist_reward)
            self.last_min_monster_dist = cur_min_dist

        # 加速阶段额外奖励
        monster_speedup_config = env_info.get("monster_speed", 500)
        is_speedup = self.step_no > monster_speedup_config
        if is_speedup and len(monster_feats) > 0:
            rewards.append(dist_reward * 0.2)

        # 视野外威胁奖励
        if self.steps_since_last_seen > 0:
            threat_delta = self.estimated_threat_dist - self.last_estimated_threat_dist
            if threat_delta > 0:
                rewards.append(threat_delta * 0.3)
        self.last_estimated_threat_dist = self.estimated_threat_dist

        # 宝箱接近奖励
        if len(treasure_feats) > 0 and treasure_feats[0][2] > 0:  # 存在标记
            current_treasure_dist = treasure_feats[0][0]
            if hasattr(self, 'last_treasure_dist') and self.last_treasure_dist is not None:
                treasure_delta = self.last_treasure_dist - current_treasure_dist
                if treasure_delta > 0:
                    rewards.append(treasure_delta * 0.15)
            self.last_treasure_dist = current_treasure_dist

        # 危险惩罚
        if len(monster_feats) > 0 and len(monster_feats[0]) > 0:
            cur_min_dist = monster_feats[0][0]
            if cur_min_dist < 0.2:
                danger_penalty = (0.2 - cur_min_dist) * 0.5
                if is_speedup:
                    danger_penalty *= 2.0
                rewards.append(-danger_penalty)

        # Buff收集奖励
        if nearest_buff and len(buffs) > self.last_buff_count:
            rewards.append(0.5)
        self.last_buff_count = len(buffs)

        # 重复探索惩罚
        if self._is_exploring_repeatedly():
            rewards.append(-0.1)

        # 终局奖励
        done = env_obs.get("done", False)
        if done:
            if hero["hp"] <= 0:
                rewards.append(-5.0)  # 死亡惩罚
            elif self.step_no >= self.max_step:
                rewards.append(3.0)   # 存活奖励

        return sum(rewards)
