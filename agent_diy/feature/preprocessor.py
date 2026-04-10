#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase DIY Agent.
峡谷追猎 DIY 智能体特征预处理与奖励设计。

全面考虑数据协议参数：
- BFS 真实路径距离
- 16维动作掩码（8移动+8闪现）
- 闪现相关参数（cooldown、count）
- 地形特征（8方向通路深度、死角、走廊）
- 物件类型区分（宝箱/buff）
"""

import numpy as np
from collections import deque


MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
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

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward."""
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

        hero_feat = np.array([
            hero_pos["x"] / MAP_SIZE,
            hero_pos["z"] / MAP_SIZE,
            hero["treasure_score"] / 1000.0,
            hero["step_score"] / (self.max_step * 1.5),
            1.0 if hero["flash_cooldown"] == 0 else 0.0,
            hero["buff_remaining_time"] / MAX_BUFF_DURATION,
        ], dtype=np.float32)

        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                speed = m.get("speed", 1) / MAX_MONSTER_SPEED
                m_pos = m["pos"]

                real_dist_norm = self._compute_real_distance(hero_pos, m_pos, map_info, m)
                direction = m.get("hero_relative_direction", 0) / 8.0

                monster_feats.append(np.array([
                    real_dist_norm,
                    direction,
                    speed,
                    m_pos["x"] / MAP_SIZE,
                    m_pos["z"] / MAP_SIZE,
                    is_in_view,
                ], dtype=np.float32))
            else:
                monster_feats.append(np.zeros(6, dtype=np.float32))

        organs = frame_state.get("organs", [])
        treasures = [o for o in organs if o.get("sub_type") == 1]
        buffs = [o for o in organs if o.get("sub_type") == 2]

        nearest_treasure = self._find_nearest_entity(treasures, hero_pos)
        nearest_buff = self._find_nearest_entity(buffs, hero_pos)

        treasure_feat = np.zeros(6, dtype=np.float32)
        if nearest_treasure:
            real_dist_norm = self._compute_real_distance(hero_pos, nearest_treasure["pos"], map_info, nearest_treasure)
            direction = nearest_treasure.get("hero_relative_direction", 0) / 8.0
            treasure_feat = np.array([
                real_dist_norm,
                direction,
                nearest_treasure["pos"]["x"] / MAP_SIZE,
                nearest_treasure["pos"]["z"] / MAP_SIZE,
                1.0,
                0.0,
            ], dtype=np.float32)

        buff_feat = np.zeros(6, dtype=np.float32)
        if nearest_buff:
            real_dist_norm = self._compute_real_distance(hero_pos, nearest_buff["pos"], map_info, nearest_buff)
            direction = nearest_buff.get("hero_relative_direction", 0) / 8.0
            buff_feat = np.array([
                1.0 if hero["buff_remaining_time"] > 0 else 0.0,
                hero["buff_remaining_time"] / MAX_BUFF_DURATION,
                direction,
                nearest_buff.get("hero_l2_distance", 5) / 5.0,
                nearest_buff["pos"]["x"] / MAP_SIZE,
                nearest_buff["pos"]["z"] / MAP_SIZE,
            ], dtype=np.float32)

        monster_speedup_config = env_info.get("monster_speed", 500)
        monster_interval = env_info.get("monster_interval", 300)
        steps_until_speedup = max(0, monster_speedup_config - self.step_no) / monster_speedup_config
        second_monster_coming = 1.0 if self.step_no >= monster_interval else 0.0

        skill_feat = np.array([
            flash_cooldown / MAX_FLASH_CD,
            1.0 if hero["flash_cooldown"] == 0 else 0.0,
            steps_until_speedup,
            flash_count / 50.0,
            second_monster_coming,
        ], dtype=np.float32)

        terrain_feat = self._compute_terrain_features(map_info, hero_pos)

        treasures_collected = env_info.get("treasures_collected", 0)
        collected_buff = env_info.get("collected_buff", 0)

        progress_feat = np.array([
            self.step_no / self.max_step,
            1.0 if self.step_no > monster_speedup_config else 0.0,
            self._compute_danger_level(monster_feats),
            self.stuck_counter / 10.0,
            treasures_collected / 10.0,
            collected_buff / 5.0,
        ], dtype=np.float32)

        legal_action = self._process_legal_action(legal_act_raw)

        feature = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            treasure_feat,
            buff_feat,
            skill_feat,
            terrain_feat,
            progress_feat,
            legal_action,
        ])

        reward = self._compute_reward(
            env_obs, hero, monster_feats, treasure_feat,
            nearest_treasure, buffs, last_action, terrain_feat
        )

        self.last_hero_pos = hero_pos.copy() if hero_pos else None
        self.history_positions.append((hero_pos["x"], hero_pos["z"]) if hero_pos else (0, 0))
        if len(self.history_positions) > 20:
            self.history_positions.pop(0)

        return feature, legal_action, reward

    def _process_legal_action(self, legal_act_raw):
        """处理16维合法动作掩码（8移动+8闪现）"""
        legal_action = [0] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]
        if sum(legal_action) == 0:
            legal_action = [1] * 16
        return np.array(legal_action, dtype=np.float32)

    def _compute_real_distance(self, hero_pos, target_pos, map_info, entity_data):
        """复合真实距离：BFS优先，不可达统一返回1.0"""
        if map_info is None:
            return 1.0

        is_in_view = entity_data.get("is_in_view", 0)

        center = len(map_info) // 2
        hx = int(hero_pos["x"] - self.hero_center_x + center)
        hz = int(hero_pos["z"] - self.hero_center_z + center)
        tx = int(target_pos["x"] - self.hero_center_x + center)
        tz = int(target_pos["z"] - self.hero_center_z + center)

        if is_in_view:
            if (0 <= tx < len(map_info) and 0 <= tz < len(map_info[0])
                and map_info[tx][tz] != 0):
                bfs_dist = self._bfs((hx, hz), (tx, tz), map_info)
                if bfs_dist < float('inf'):
                    return min(bfs_dist / 42.0, 1.0)

        return 1.0

    def _bfs(self, start, goal, map_info):
        """BFS 计算最短路径距离"""
        from collections import deque

        sx, sz = start
        gx, gz = goal

        if map_info[sx][sz] == 0:
            return float('inf')

        queue = deque([(sx, sz, 0)])
        visited = {(sx, sz)}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            x, z, dist = queue.popleft()
            if x == gx and z == gz:
                return dist

            for dx, dz in directions:
                nx, nz = x + dx, z + dz
                if (0 <= nx < len(map_info) and 0 <= nz < len(map_info[0])
                    and (nx, nz) not in visited
                    and map_info[nx][nz] != 0):
                    visited.add((nx, nz))
                    queue.append((nx, nz, dist + 1))

        return float('inf')

    def _compute_terrain_features(self, map_info, hero_pos):
        """计算地形特征：8方向通路深度 + 死角 + 走廊 + 开阔"""
        if map_info is None:
            return np.zeros(11, dtype=np.float32)

        direction_depths = []
        for direction in range(8):
            depth = self._trace_path(hero_pos, direction, map_info)
            direction_depths.append(depth / 10.0)

        escape_count = sum(1 for d in direction_depths if d >= 0.3)
        is_dead = 1.0 if escape_count <= 1 else 0.0

        vertical = min(direction_depths[2], direction_depths[6])
        horizontal = min(direction_depths[4], direction_depths[0])
        is_corridor = 1.0 if vertical >= 0.5 and horizontal <= 0.1 else 0.0

        total_depth = sum(direction_depths)
        is_open = 1.0 if total_depth >= 3.0 else 0.0

        return np.array(direction_depths + [is_dead, is_corridor, is_open], dtype=np.float32)

    def _trace_path(self, hero_pos, direction, map_info):
        """从起点沿指定方向发射射线，返回能走的最大深度"""
        direction_vectors = [
            (1, 0), (1, -1), (0, -1), (-1, -1),
            (-1, 0), (-1, 1), (0, 1), (1, 1),
        ]

        dx, dz = direction_vectors[direction]
        x, z = hero_pos["x"], hero_pos["z"]

        depth = 0
        max_depth = 20

        while depth < max_depth:
            next_x = x + dx * (depth + 1)
            next_z = z + dz * (depth + 1)
            if not self._is_passable(next_x, next_z, map_info):
                break
            depth += 1

        return float(depth)

    def _is_passable(self, x, z, map_info):
        """检查指定坐标是否可通行"""
        center = len(map_info) // 2
        local_x = int(x - self.hero_center_x + center)
        local_z = int(z - self.hero_center_z + center)

        if 0 <= local_x < len(map_info) and 0 <= local_z < len(map_info[0]):
            return map_info[local_x][local_z] != 0
        return False

    def _find_nearest_entity(self, entities, hero_pos):
        """找到最近的实体（宝箱或buff）"""
        if not entities:
            return None
        nearest = None
        min_dist = float('inf')
        for e in entities:
            if e.get("status") != 1:
                continue
            dist = abs(e["pos"]["x"] - hero_pos["x"]) + abs(e["pos"]["z"] - hero_pos["z"])
            if dist < min_dist:
                min_dist = dist
                nearest = e
        return nearest

    def _compute_danger_level(self, monster_feats):
        """计算当前危险等级"""
        danger = 0.0
        for m_feat in monster_feats:
            if m_feat[5] > 0:
                danger = max(danger, 1.0 - m_feat[0])
        return danger

    def _is_exploring_repeatedly(self):
        """检测是否在重复探索"""
        if len(self.history_positions) < 10:
            return False
        recent = self.history_positions[-10:]
        if len(set(recent)) <= 3:
            return True
        return False

    def _compute_reward(self, env_obs, hero, monster_feats, treasure_feat,
                       nearest_treasure, buffs, current_action, terrain_feat):
        """计算综合奖励"""
        rewards = []
        env_info = env_obs["observation"]["env_info"]

        rewards.append(0.01)

        current_step_score = hero["step_score"]
        if hasattr(self, 'last_step_score'):
            step_score_diff = current_step_score - self.last_step_score
            if step_score_diff > 0:
                rewards.append(step_score_diff * 0.1)
        self.last_step_score = current_step_score

        current_treasure_score = hero["treasure_score"]
        if hasattr(self, 'last_treasure_score'):
            treasure_diff = current_treasure_score - self.last_treasure_score
            if treasure_diff > 0:
                rewards.append(treasure_diff * 1.0)
        self.last_treasure_score = current_treasure_score

        current_buff_count = len([b for b in buffs if b.get("status") == 1])
        if current_buff_count > self.last_buff_count:
            rewards.append(0.5)
        self.last_buff_count = current_buff_count

        monsters = env_obs.get("observation", {}).get("frame_state", {}).get("organs", [])
        monsters = [m for m in monsters if m.get("sub_type") == 3]

        cur_min_dist = 1.0
        for m_feat in monster_feats:
            if m_feat[5] > 0:
                cur_min_dist = min(cur_min_dist, m_feat[0])

        for m in monsters:
            real_dist = self._compute_real_distance(self.hero_pos, m["pos"], self.map_info, m)
            cur_min_dist = min(cur_min_dist, real_dist)

        if hasattr(self, 'last_min_monster_dist'):
            dist_delta = self.last_min_monster_dist - cur_min_dist
            dist_reward = 0.1 * dist_delta
            rewards.append(dist_reward)

            monster_speedup_config = env_info.get("monster_speed", 500)
            if self.step_no > monster_speedup_config:
                rewards.append(dist_reward * 2.0)
        self.last_min_monster_dist = cur_min_dist

        if nearest_treasure and treasure_feat[4] > 0:
            current_treasure_dist = treasure_feat[0]
            if hasattr(self, 'last_treasure_dist') and self.last_treasure_dist is not None:
                treasure_delta = self.last_treasure_dist - current_treasure_dist
                if treasure_delta > 0:
                    rewards.append(treasure_delta * 0.5)
            self.last_treasure_dist = current_treasure_dist

        if nearest_buff and hero["buff_remaining_time"] == 0:
            current_buff_dist = self._compute_real_distance(
                self.hero_pos, nearest_buff["pos"], self.map_info, nearest_buff
            )
            if hasattr(self, 'last_buff_dist') and self.last_buff_dist is not None:
                buff_delta = self.last_buff_dist - current_buff_dist
                if buff_delta > 0:
                    rewards.append(buff_delta * 0.5)
            self.last_buff_dist = current_buff_dist

        monster_speedup_config = env_info.get("monster_speed", 500)
        is_speedup = self.step_no > monster_speedup_config
        steps_until_speedup = monster_speedup_config - self.step_no

        is_corridor = terrain_feat[9] > 0
        is_dead = terrain_feat[8] > 0
        is_open = terrain_feat[10] > 0

        if steps_until_speedup > 0 and steps_until_speedup < 100:
            buffer_reward = 0.05 * (1.0 - steps_until_speedup / 100.0)
            if cur_min_dist < 0.5:
                buffer_reward *= (1.0 + (0.5 - cur_min_dist))
            rewards.append(buffer_reward)

        corridor_penalty = -0.1 if not is_speedup else -0.2
        dead_penalty = -0.2 if not is_speedup else -0.4
        if is_corridor:
            rewards.append(corridor_penalty)
        if is_dead:
            rewards.append(dead_penalty)
        if is_open and cur_min_dist < 0.3:
            rewards.append(0.1)

        if cur_min_dist < 0.2:
            danger_penalty = (0.2 - cur_min_dist) * 0.5
            if is_speedup:
                danger_penalty *= 2.0
            rewards.append(-danger_penalty)
        elif cur_min_dist < 0.4 and is_speedup:
            danger_penalty = (0.4 - cur_min_dist) * 0.25
            rewards.append(-danger_penalty)

        is_flash = current_action >= 8
        gain = 0.0
        if is_flash:
            flash_dir = current_action - 8
            direction_vectors = [
                (1, 0), (1, -1), (0, -1), (-1, -1),
                (-1, 0), (-1, 1), (0, 1), (1, 1),
            ]
            dx, dz = direction_vectors[flash_dir]
            FLASH_DIST = 8.0
            new_x = self.hero_pos["x"] + dx * FLASH_DIST
            new_z = self.hero_pos["z"] + dz * FLASH_DIST
            new_hero_pos = {"x": new_x, "z": new_z}

            for m in monsters:
                new_entity_data = {"is_in_view": 1}
                new_dist = self._compute_real_distance(new_hero_pos, m["pos"], self.map_info, new_entity_data)
                gain = max(gain, new_dist - cur_min_dist)

            rewards.append(-0.05)

            if is_speedup:
                if gain <= 0.15:
                    rewards.append(-0.1)
            else:
                if gain <= 0.3:
                    rewards.append(-0.2)

        if self.step_no > monster_speedup_config:
            if hasattr(self, 'last_min_monster_dist'):
                if cur_min_dist > self.last_min_monster_dist:
                    rewards.append(0.2)

        if len(monsters) >= 2:
            m1_dir = monster_feats[0][1] if monster_feats[0][5] > 0 else -1
            m2_dir = monster_feats[1][1] if monster_feats[1][5] > 0 else -1
            if m1_dir >= 0 and m2_dir >= 0:
                dir_diff = abs(m1_dir - m2_dir)
                if 2 <= dir_diff <= 6:
                    rewards.append(-0.15)

        actual_dx = self.hero_pos["x"] - self.last_hero_pos["x"]
        actual_dz = self.hero_pos["z"] - self.last_hero_pos["z"]
        if abs(actual_dx) < 0.1 and abs(actual_dz) < 0.1:
            rewards.append(-0.05)
        else:
            current_pos = (int(self.hero_pos["x"]), int(self.hero_pos["z"]))
            if not hasattr(self, 'visited_positions'):
                self.visited_positions = set()
            if current_pos not in self.visited_positions:
                self.visited_positions.add(current_pos)
                rewards.append(0.02)

        if self._is_exploring_repeatedly():
            rewards.append(-0.1)

        if len(monsters) >= 2:
            second_dist = monster_feats[1][0] if monster_feats[1][5] > 0 else 1.0
            if second_dist < 0.3:
                rewards.append(-0.15)

        return [sum(rewards)]
