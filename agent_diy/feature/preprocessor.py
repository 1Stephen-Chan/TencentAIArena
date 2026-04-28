#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Gorge Chase DIY agent.
"""

import math
import numpy as np

from agent_diy.conf.conf import Config


ACTION_DIRS = [
    (1, 0),   # E
    (1, -1),  # NE
    (0, -1),  # N
    (-1, -1), # NW
    (-1, 0),  # W
    (-1, 1),  # SW
    (0, 1),   # S
    (1, 1),   # SE
]


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    denom = float(v_max - v_min)
    if abs(denom) < 1e-6:
        return 0.0
    return (v - v_min) / denom


def _dir_to_vec(direction):
    table = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, -1.0),
        3: (0.0, -1.0),
        4: (-1.0, -1.0),
        5: (-1.0, 0.0),
        6: (-1.0, 1.0),
        7: (0.0, 1.0),
        8: (1.0, 1.0),
    }
    x, z = table.get(int(direction), (0.0, 0.0))
    n = math.sqrt(x * x + z * z)
    if n < 1e-6:
        return 0.0, 0.0
    return x / n, z / n


def _action_vec(action_idx):
    dx, dz = ACTION_DIRS[int(action_idx) % 8]
    n = math.sqrt(dx * dx + dz * dz)
    return dx / n, dz / n


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = int(getattr(self, "configured_max_step", 1000))
        self.monster_interval = int(getattr(self, "configured_monster_interval", 200))
        self.monster_speedup = int(getattr(self, "configured_monster_speedup", 300))

        # 局内记忆：用于探索与反绕圈特征
        self.visited_cells = {}
        self.prev_cell = None
        self.prev_hero_x = None
        self.prev_hero_z = None
        self.prev_action = -1
        self.same_cell_streak = 0
        self.stuck_steps = 0
        self.lost_monster_steps = 0

        # 固定 buff 点记忆：记录冷却起点和最近一次已知位置
        self.buff_memory = {}
        self.known_buff_ids = set()
        self.known_treasure_cells = set()

    def set_episode_context(self, monster_interval=None, monster_speedup=None, max_step=None):
        max_step_val = int(max_step) if max_step is not None else int(getattr(self, "configured_max_step", 1000))
        max_step_val = max(1, max_step_val)

        interval = monster_interval if monster_interval is not None else getattr(self, "configured_monster_interval", 200)
        interval = int(interval)
        if interval < 0:
            interval = max(11, int(round(max_step_val * 0.22)))

        speedup = monster_speedup if monster_speedup is not None else getattr(self, "configured_monster_speedup", 300)
        speedup = int(speedup)
        if speedup < 0:
            speedup = max(interval + 1, int(round(max_step_val * 0.38)))
        speedup = max(interval + 1, speedup)

        self.configured_max_step = max_step_val
        self.configured_monster_interval = interval
        self.configured_monster_speedup = speedup

        self.max_step = max_step_val
        self.monster_interval = interval
        self.monster_speedup = speedup

    def _parse_legal_action(self, legal_act_raw):
        legal = np.ones(Config.ACTION_NUM, dtype=np.float32)
        if isinstance(legal_act_raw, (list, tuple, np.ndarray)) and len(legal_act_raw) > 0:
            first = legal_act_raw[0]
            if isinstance(first, (bool, np.bool_)):
                legal[:] = 1.0
                for i in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                    legal[i] = 1.0 if bool(legal_act_raw[i]) else 0.0
            else:
                legal[:] = 0.0
                for a in legal_act_raw:
                    try:
                        idx = int(a)
                        if 0 <= idx < Config.ACTION_NUM:
                            legal[idx] = 1.0
                    except Exception:
                        continue

        if float(np.sum(legal)) <= 0.0:
            legal[:] = 1.0
        return legal

    def _extract_hero(self, frame_state, env_info):
        heroes = frame_state.get("heroes", {})
        hero_pos = heroes.get("pos", env_info.get("pos", {}))

        hero_x = float(hero_pos.get("x", 0.0))
        hero_z = float(hero_pos.get("z", 0.0))
        return heroes, hero_x, hero_z

    def _bucket_center_dist(self, bucket):
        centers = [15.0, 45.0, 75.0, 105.0, 135.0, 165.0]
        b = int(np.clip(bucket, 0, 5))
        return centers[b]

    def _monster_sort_key(self, monster, hero_x, hero_z):
        if not isinstance(monster, dict):
            return Config.MAP_DIAG

        in_view = float(monster.get("is_in_view", 0.0))
        if in_view >= 0.5:
            m_pos = monster.get("pos", {})
            mx = float(m_pos.get("x", hero_x))
            mz = float(m_pos.get("z", hero_z))
            dist = math.sqrt((mx - hero_x) ** 2 + (mz - hero_z) ** 2)
        else:
            dist = self._bucket_center_dist(float(monster.get("hero_l2_distance", 5.0)))

        speed = float(monster.get("speed", 1.0))
        return dist - 0.5 * speed

    def _build_monster_features(self, monsters, hero_x, hero_z):
        """
        建立怪物特征，并按威胁排序以稳定槽位语义
        """

        feats = []
        dists = []
        monster_local = []

        monster_list = [m for m in monsters if isinstance(m, dict)]
        monster_list.sort(key=lambda m: self._monster_sort_key(m, hero_x, hero_z))

        for i in range(2):
            if i < len(monster_list):
                m = monster_list[i]
                m_pos = m.get("pos", {})
                in_view = float(m.get("is_in_view", 0))
                speed = float(m.get("speed", 1.0))
                bucket = float(m.get("hero_l2_distance", 5.0))
                rel_dir = int(m.get("hero_relative_direction", 0))

                if in_view >= 0.5:
                    mx = float(m_pos.get("x", hero_x))
                    mz = float(m_pos.get("z", hero_z))
                    dx = mx - hero_x
                    dz = mz - hero_z
                    dist = math.sqrt(dx * dx + dz * dz)
                    if dist > 1e-6:
                        dir_x = dx / dist
                        dir_z = dz / dist
                    else:
                        dir_x, dir_z = 0.0, 0.0
                else:
                    dir_x, dir_z = _dir_to_vec(rel_dir)
                    dist = self._bucket_center_dist(bucket)
                    dx = dir_x * dist
                    dz = dir_z * dist

                feats.extend(
                    [
                        1.0,                                   # 怪物存在标记
                        in_view,                               # 是否在视野内
                        _norm(dist, Config.MAP_DIAG),          # 归一化距离
                        _norm(bucket, 5.0),                    # 归一化桶距离
                        _norm(speed, Config.MAX_MONSTER_SPEED),# 归一化速度
                        dir_x,                                 # 朝向英雄的单位方向 x
                        dir_z,                                 # 朝向英雄的单位方向 z
                        float(np.clip(dx / 20.0, -1.0, 1.0)),  # 相对位移 x（局部裁剪）
                        float(np.clip(dz / 20.0, -1.0, 1.0)),  # 相对位移 z（局部裁剪）
                        math.exp(-dist / 20.0),                # 距离威胁值
                    ]
                )

                dists.append(dist)
                monster_local.append(
                    {
                        "dx": float(dx),
                        "dz": float(dz),
                        "speed": speed,
                        "dist": dist,
                    }
                )
            else:
                feats.extend([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                dists.append(Config.MAP_DIAG)

        return np.array(feats, dtype=np.float32), dists, monster_local

    def _build_target_features(self, organs, hero_x, hero_z, buff_refresh_time):
        """
        建立目标特征，补充位置不确定性，并固定 buff 槽位身份
        """

        treasures = []
        buffs_by_id = {}
        alive_buffs = []
        new_treasure_discovery = 0
        new_buff_discovery = 0

        for organ in organs:
            if not isinstance(organ, dict):
                continue

            status = int(organ.get("status", 1))
            sub_type = int(organ.get("sub_type", 0))
            config_id = int(organ.get("config_id", 0))
            m_pos = organ.get("pos", {})

            pos_x = float(m_pos.get("x", -1.0))
            pos_z = float(m_pos.get("z", -1.0))
            has_exact_pos = pos_x != -1.0 and pos_z != -1.0

            rel_dir = int(organ.get("hero_relative_direction", 0))
            dir_x, dir_z = _dir_to_vec(rel_dir)
            pos_certainty = 0.0

            if has_exact_pos:
                dx = pos_x - hero_x
                dz = pos_z - hero_z
                dist = math.sqrt(dx * dx + dz * dz)
                if dist > 1e-6:
                    dir_x = dx / dist
                    dir_z = dz / dist
                pos_certainty = 1.0
            else:
                bucket = float(organ.get("hero_l2_distance", 5.0))
                dist = self._bucket_center_dist(bucket)
                dx = dir_x * dist
                dz = dir_z * dist

            dist_norm = _norm(dist, Config.MAP_DIAG)

            if sub_type == 1:
                if has_exact_pos:
                    treasure_cell = self._to_visit_cell(pos_x, pos_z)
                    if treasure_cell not in self.known_treasure_cells:
                        self.known_treasure_cells.add(treasure_cell)
                        new_treasure_discovery += 1

                item = [
                    1.0,                                   # 宝箱存在标记
                    dist_norm,                             # 归一化距离
                    dir_x,                                 # 指向宝箱的方向 x
                    dir_z,                                 # 指向宝箱的方向 z
                    float(np.clip(dx / 20.0, -1.0, 1.0)),  # 相对位移 x（局部裁剪）
                    float(np.clip(dz / 20.0, -1.0, 1.0)),  # 相对位移 z（局部裁剪）
                    pos_certainty,                         # 位置确定性：1精确/0桶估计
                ]
                treasures.append((dist, item))

            elif sub_type == 2:
                if has_exact_pos:
                    buff_key = config_id if config_id > 0 else self._to_visit_cell(pos_x, pos_z)
                    if buff_key not in self.known_buff_ids:
                        self.known_buff_ids.add(buff_key)
                        new_buff_discovery += 1

                mem = self.buff_memory.setdefault(
                    config_id,
                    {"last_status": status, "start_step": -1, "last_pos": None},
                )

                if has_exact_pos:
                    mem["last_pos"] = (pos_x, pos_z)
                elif mem["last_pos"] is not None:
                    last_x, last_z = mem["last_pos"]
                    dx = float(last_x - hero_x)
                    dz = float(last_z - hero_z)
                    dist = math.sqrt(dx * dx + dz * dz)
                    dist_norm = _norm(dist, Config.MAP_DIAG)
                    if dist > 1e-6:
                        dir_x = dx / dist
                        dir_z = dz / dist
                    pos_certainty = 0.5

                if mem["last_status"] == 1 and status == 0:
                    mem["start_step"] = self.step_no
                elif status == 1:
                    mem["start_step"] = -1
                mem["last_status"] = status

                cd_remain = 0.0
                if status == 0 and mem["start_step"] != -1:
                    elapsed_steps = self.step_no - mem["start_step"]
                    cd_remain = max(0.0, buff_refresh_time - float(elapsed_steps))

                cd_norm = _norm(cd_remain, buff_refresh_time)

                item = [
                    float(status),                          # buff 当前是否可吃
                    cd_norm,                                # buff 剩余冷却归一化
                    dist_norm,                              # 归一化距离
                    dir_x,                                  # 指向 buff 的方向 x
                    dir_z,                                  # 指向 buff 的方向 z
                    float(np.clip(dx / 20.0, -1.0, 1.0)),   # 相对位移 x（局部裁剪）
                    float(np.clip(dz / 20.0, -1.0, 1.0)),   # 相对位移 z（局部裁剪）
                    pos_certainty,                          # 位置确定性：1精确/0.5记忆/0桶估计
                ]
                buffs_by_id[config_id] = (dist, item)
                if status == 1:
                    alive_buffs.append((dist, item))

        treasures.sort(key=lambda x: x[0])
        alive_buffs.sort(key=lambda x: x[0])

        feat = []

        for i in range(Config.TREASURE_SLOT_NUM):
            if i < len(treasures):
                feat.extend(treasures[i][1])
            else:
                feat.extend([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Buff 槽位按 config_id 绑定，保证冷却与地点语义稳定
        buff_ids = sorted(buffs_by_id.keys())
        for i in range(Config.BUFF_SLOT_NUM):
            if i < len(buff_ids):
                _, item = buffs_by_id[buff_ids[i]]
                feat.extend(item)
            else:
                feat.extend([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        nearest_treasure_vec = None
        nearest_treasure_dist_norm = 1.0
        if treasures:
            best_t_item = treasures[0][1]
            nearest_treasure_dist_norm = best_t_item[1]
            nearest_treasure_vec = (best_t_item[2], best_t_item[3])

        nearest_buff_vec = None
        nearest_buff_dist_norm = 1.0
        if alive_buffs:
            best_b_item = alive_buffs[0][1]
            nearest_buff_dist_norm = best_b_item[2]
            nearest_buff_vec = (best_b_item[3], best_b_item[4])

        target_memory = {
            "new_treasure_discovery": int(new_treasure_discovery),
            "new_buff_discovery": int(new_buff_discovery),
            "known_treasure_count": int(len(self.known_treasure_cells)),
            "known_buff_count": int(len(self.known_buff_ids)),
        }

        return (
            np.array(feat, dtype=np.float32),
            nearest_treasure_vec,
            nearest_treasure_dist_norm,
            nearest_buff_vec,
            nearest_buff_dist_norm,
            target_memory,
        )

    def _build_local_map_feature(self, map_info):
        view_size = int(Config.LOCAL_MAP_VIEW_SIZE)
        feat = np.zeros((view_size, view_size), dtype=np.float32)

        if not isinstance(map_info, list) or len(map_info) == 0:
            return feat.reshape(-1)

        rows = len(map_info)
        cols = len(map_info[0]) if isinstance(map_info[0], list) else 0
        if cols <= 0:
            return feat.reshape(-1)

        center_r = rows // 2
        center_c = cols // 2
        radius = view_size // 2

        for out_r, r in enumerate(range(center_r - radius, center_r + radius + 1)):
            for out_c, c in enumerate(range(center_c - radius, center_c + radius + 1)):
                if 0 <= r < rows and 0 <= c < cols and isinstance(map_info[r], list) and c < len(map_info[r]):
                    feat[out_r, out_c] = 1.0 if float(map_info[r][c]) != 0.0 else 0.0

        return feat.reshape(-1)


    def _local_passable(self, map_info, dx, dz):
        if not isinstance(map_info, list) or len(map_info) == 0:
            return True
        rows = len(map_info)
        cols = len(map_info[0]) if isinstance(map_info[0], list) else 0
        if cols <= 0:
            return True

        cr = rows // 2
        cc = cols // 2

        r = cr + int(round(dz))
        c = cc + int(round(dx))

        if 0 <= r < rows and 0 <= c < cols:
            return float(map_info[r][c]) != 0.0
        return True

    def _simulate_move(self, map_info, speed, dir_dx, dir_dz):
        cur_x, cur_z = 0.0, 0.0
        speed = max(1, int(round(speed)))

        for _ in range(speed):
            nx = cur_x + dir_dx
            nz = cur_z + dir_dz

            if abs(dir_dx) == 1 and abs(dir_dz) == 1:
                diag_ok = self._local_passable(map_info, nx, nz) and (
                    self._local_passable(map_info, cur_x + dir_dx, cur_z)
                    or self._local_passable(map_info, cur_x, cur_z + dir_dz)
                )
                if not diag_ok:
                    break
            else:
                if not self._local_passable(map_info, nx, nz):
                    break

            cur_x, cur_z = nx, nz

        return float(cur_x), float(cur_z)

    def _simulate_flash(self, map_info, dir_dx, dir_dz, max_len):
        for d in range(int(max_len), 0, -1):
            tx = dir_dx * d
            tz = dir_dz * d
            if self._local_passable(map_info, tx, tz):
                return float(tx), float(tz)
        return 0.0, 0.0

    def _to_visit_cell(self, x, z):
        cell_size = max(float(Config.VISIT_CELL_SIZE), 1.0)
        x = float(np.clip(x, 0.0, Config.MAP_SIZE))
        z = float(np.clip(z, 0.0, Config.MAP_SIZE))
        return int(math.floor(x / cell_size)), int(math.floor(z / cell_size))

    def _visit_count(self, cell):
        return float(self.visited_cells.get(cell, 0))

    def _visit_novelty(self, visits):
        return 1.0 - _norm(float(visits), Config.MAX_VISIT_COUNT)

    def _directional_visit_novelty(self, current_cell):
        scores = []
        for dx, dz in ACTION_DIRS:
            next_cell = (current_cell[0] + dx, current_cell[1] + dz)
            scores.append(self._visit_novelty(self._visit_count(next_cell)))
        return np.array(scores, dtype=np.float32)

    def _is_reverse_action(self, action_a, action_b):
        try:
            a = int(action_a)
            b = int(action_b)
        except Exception:
            return False
        if a < 0 or b < 0:
            return False
        return abs((a % 8) - (b % 8)) == 4

    def _update_history_memory(self, hero_x, hero_z, last_action):
        current_cell = self._to_visit_cell(hero_x, hero_z)
        previous_cell = self.prev_cell

        if self.prev_hero_x is None or self.prev_hero_z is None:
            move_dx = 0.0
            move_dz = 0.0
            self.stuck_steps = 0
        else:
            move_dx = hero_x - self.prev_hero_x
            move_dz = hero_z - self.prev_hero_z
            move_dist = math.sqrt(move_dx * move_dx + move_dz * move_dz)
            if move_dist < Config.STUCK_MOVE_THRESHOLD:
                self.stuck_steps = min(self.stuck_steps + 1, int(Config.MAX_STUCK_STEPS))
            else:
                self.stuck_steps = 0

        if previous_cell is not None and current_cell == previous_cell:
            self.same_cell_streak = min(self.same_cell_streak + 1, int(Config.MAX_REPEAT_STEPS))
        else:
            self.same_cell_streak = 0

        visits_before = self.visited_cells.get(current_cell, 0)
        entered_new_cell = 1.0 if visits_before == 0 else 0.0
        self.visited_cells[current_cell] = visits_before + 1
        current_cell_visits = self.visited_cells[current_cell]

        if last_action is not None and int(last_action) >= 0:
            last_dir_x, last_dir_z = _action_vec(last_action)
        else:
            last_dir_x, last_dir_z = 0.0, 0.0

        reverse_action_flag = 1.0 if self._is_reverse_action(last_action, self.prev_action) else 0.0

        history = {
            "current_cell": current_cell,
            "previous_cell": previous_cell,
            "current_cell_visits": current_cell_visits,
            "same_cell_streak": self.same_cell_streak,
            "entered_new_cell_flag": entered_new_cell,
            "prev_move_dx": float(np.clip(move_dx / Config.VISIT_CELL_SIZE, -1.0, 1.0)),
            "prev_move_dz": float(np.clip(move_dz / Config.VISIT_CELL_SIZE, -1.0, 1.0)),
            "last_action_dir_x": last_dir_x,
            "last_action_dir_z": last_dir_z,
            "reverse_action_flag": reverse_action_flag,
            "stuck_steps": self.stuck_steps,
            "directional_novelty": self._directional_visit_novelty(current_cell),
        }

        self.prev_hero_x = hero_x
        self.prev_hero_z = hero_z
        self.prev_cell = current_cell
        self.prev_action = int(last_action) if last_action is not None and int(last_action) >= 0 else -1

        return history

    def _get_phase_profile(self, min_dist, visible_monster_count):
        interval = max(1, int(self.monster_interval))
        speedup = max(interval + 1, int(self.monster_speedup))

        if self.step_no < interval:
            base_phase_id = 0
            phase_start, phase_end = 0, interval
        elif self.step_no < speedup:
            base_phase_id = 1
            phase_start, phase_end = interval, speedup
        else:
            base_phase_id = 2
            phase_start, phase_end = speedup, max(speedup + 1, int(self.max_step))

        phase_id = base_phase_id
        if min_dist <= 2.5 or visible_monster_count >= 2:
            phase_id = max(phase_id, 2)
        elif min_dist <= 4.0 or visible_monster_count >= 1:
            phase_id = max(phase_id, 1)

        profiles = [
            {
                "name": "stage1_explore",
                "explore_scale": 1.30,
                "new_cell_bonus": 0.24,
                "fresh_cell_bonus": 0.18,
                "repeat_penalty": 0.14,
                "search_bonus_scale": 1.20,
                "reward_enter_bonus": 0.18,
                "reward_move_cap": 0.08,
                "reward_repeat_penalty": 0.06,
                "reward_buff_discovery_bonus": 0.35,
                "reward_treasure_discovery_bonus": 0.10,
                "eval_explore_weight": 0.55,
                "eval_treasure_weight": 0.30,
            },
            {
                "name": "stage2_pressure",
                "explore_scale": 0.95,
                "new_cell_bonus": 0.16,
                "fresh_cell_bonus": 0.10,
                "repeat_penalty": 0.09,
                "search_bonus_scale": 1.05,
                "reward_enter_bonus": 0.10,
                "reward_move_cap": 0.05,
                "reward_repeat_penalty": 0.04,
                "reward_buff_discovery_bonus": 0.22,
                "reward_treasure_discovery_bonus": 0.05,
                "eval_explore_weight": 0.25,
                "eval_treasure_weight": 0.32,
            },
            {
                "name": "stage3_survival",
                "explore_scale": 0.35,
                "new_cell_bonus": 0.05,
                "fresh_cell_bonus": 0.02,
                "repeat_penalty": 0.02,
                "search_bonus_scale": 0.55,
                "reward_enter_bonus": 0.03,
                "reward_move_cap": 0.02,
                "reward_repeat_penalty": 0.00,
                "reward_buff_discovery_bonus": 0.08,
                "reward_treasure_discovery_bonus": 0.00,
                "eval_explore_weight": 0.05,
                "eval_treasure_weight": 0.18,
            },
        ]

        profile = dict(profiles[int(np.clip(phase_id, 0, 2))])
        profile["phase_id"] = int(np.clip(phase_id, 0, 2))
        profile["base_phase_id"] = int(base_phase_id)
        profile["monster_interval"] = int(interval)
        profile["monster_speedup"] = int(speedup)

        phase_span = max(1.0, float(phase_end - phase_start))
        profile["phase_progress"] = float(np.clip((self.step_no - phase_start) / phase_span, 0.0, 1.0))
        return profile

    def _estimate_action_eval(
        self,
        map_info,
        legal_action,
        hero_speed,
        monster_local,
        nearest_treasure_vec,
        nearest_treasure_dist_norm,
        nearest_buff_vec,
        nearest_buff_dist_norm,
        min_dist,
        hero_x,
        hero_z,
        history,
        visible_monster_count,
        lost_monster_steps,
        phase_profile,
    ):
        safety = np.full(Config.ACTION_NUM, -1.0, dtype=np.float32)
        treasure = np.full(Config.ACTION_NUM, -1.0, dtype=np.float32)
        explore = np.full(Config.ACTION_NUM, -1.0, dtype=np.float32)

        current_cell = history["current_cell"]
        previous_cell = history["previous_cell"]
        current_cell_visits = float(history["current_cell_visits"])
        repeat_pressure = _norm(float(history["same_cell_streak"]), Config.MAX_REPEAT_STEPS)
        safety_gate = 0.35 + 0.65 * _norm(min_dist, 6.0)
        lost_ratio = _norm(float(lost_monster_steps), Config.MAX_LOST_MONSTER_STEPS)
        search_mode = visible_monster_count <= 0 and min_dist > 4.0
        early_game = self.step_no <= 80
        explore_scale = float(phase_profile.get("explore_scale", 1.0))
        new_cell_bonus = float(phase_profile.get("new_cell_bonus", 0.0))
        fresh_cell_bonus = float(phase_profile.get("fresh_cell_bonus", 0.0))
        repeat_penalty = float(phase_profile.get("repeat_penalty", 0.0))
        search_bonus_scale = float(phase_profile.get("search_bonus_scale", 1.0))

        for act in range(Config.ACTION_NUM):
            if legal_action[act] < 0.5:
                continue

            dir_dx, dir_dz = ACTION_DIRS[act % 8]

            if act < 8:
                dst_x, dst_z = self._simulate_move(map_info, hero_speed, dir_dx, dir_dz)
                moved_dist = abs(dst_x) + abs(dst_z)
            else:
                max_len = 10 if (act % 2 == 0) else 8
                dst_x, dst_z = self._simulate_flash(map_info, dir_dx, dir_dz, max_len)
                moved_dist = abs(dst_x) + abs(dst_z)

            if monster_local:
                min_after = 1e9
                near_threat_count = 0
                for m in monster_local:
                    dx = m["dx"] - dst_x
                    dz = m["dz"] - dst_z
                    d = math.sqrt(dx * dx + dz * dz) - 0.5 * float(m.get("speed", 1.0))
                    min_after = min(min_after, d)
                    if d <= 3.0:
                        near_threat_count += 1
            else:
                min_after = Config.MAP_DIAG
                near_threat_count = 0

            s = float(np.clip((min_after - 2.0) / 8.0, -1.0, 1.0))

            # 对闪现动作额外收紧：落点贴怪、双怪夹击、开局高压时显式压低安全分。
            flash_hard_block = (
                act >= 8
                and (
                    min_after <= 1.6
                    or (near_threat_count >= 2 and min_after <= 3.0)
                    or (early_game and min_after <= 2.5)
                )
            )
            if act >= 8:
                if flash_hard_block:
                    s = -1.0
                elif min_after < min_dist - 0.5 and min_after <= 3.0:
                    s = min(s, -0.60)

            if act < 8 and moved_dist < 0.5:
                if s > 0.0:
                    s -= 0.35
                else:
                    s -= 0.05

            if act >= 8 and min_dist > 6.0:
                s -= 0.20

            ax, az = _action_vec(act)

            t_treasure = 0.0
            if nearest_treasure_vec is not None:
                tx, tz = nearest_treasure_vec
                align_t = float(ax * tx + az * tz)
                dist_weight_t = float(np.clip(1.2 - nearest_treasure_dist_norm, 0.0, 1.0))
                t_treasure = align_t * dist_weight_t

            t_buff = 0.0
            if nearest_buff_vec is not None:
                bx, bz = nearest_buff_vec
                align_b = float(ax * bx + az * bz)
                dist_weight_b = float(np.clip(1.2 - nearest_buff_dist_norm, 0.0, 1.0))
                t_buff = align_b * dist_weight_b * 1.5

            t_raw = max(t_treasure, t_buff)
            if act >= 8:
                t_raw *= 0.65

            safety_clipped = float(np.clip(s, -1.0, 1.0))
            t = t_raw * (0.65 + 0.35 * safety_clipped)

            # 危险动作上的逐利倾向要显著降温，避免“看到宝箱就冲”。
            if safety_clipped <= -0.30:
                t *= 0.15 if act >= 8 else 0.45
            elif safety_clipped <= 0.0:
                t *= 0.50 if act >= 8 else 0.75

            if flash_hard_block:
                t = -1.0

            target_cell = self._to_visit_cell(hero_x + dst_x, hero_z + dst_z)
            target_visits = self._visit_count(target_cell)
            novelty = self._visit_novelty(target_visits)

            e = 2.0 * novelty - 1.0
            if target_cell != current_cell:
                e += new_cell_bonus + 0.25 * repeat_pressure
            else:
                e -= 0.28 + 0.30 * repeat_pressure

            if target_visits <= 0:
                e += fresh_cell_bonus

            if target_visits < current_cell_visits:
                e += 0.10 + 0.06 * explore_scale
            elif target_visits >= current_cell_visits:
                e -= repeat_penalty

            if previous_cell is not None and target_cell == previous_cell and current_cell != previous_cell:
                e -= 0.20

            # 丢失怪物视野后，显式鼓励离开当前格并搜索低访问区域
            if search_mode:
                if target_cell != current_cell:
                    e += search_bonus_scale * (0.12 + 0.25 * lost_ratio)
                else:
                    e -= search_bonus_scale * (0.10 + 0.20 * lost_ratio)

                if target_visits <= 0:
                    e += search_bonus_scale * (0.10 + 0.15 * lost_ratio)
                elif target_visits >= current_cell_visits:
                    e -= search_bonus_scale * (0.05 + 0.10 * lost_ratio)

            if moved_dist < 0.5:
                e -= 0.35

            if act >= 8 and min_dist > 6.0:
                e -= 0.10

            if flash_hard_block:
                e = min(e, -0.80)

            e *= safety_gate * explore_scale

            safety[act] = float(np.clip(s, -1.0, 1.0))
            treasure[act] = float(np.clip(t, -1.0, 1.0))
            explore[act] = float(np.clip(e, -1.0, 1.0))

        return safety, treasure, explore

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        self.step_no = observation["step_no"]

        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act = observation["legal_action"]

        self.max_step = int(env_info.get("max_step", 1000))
        total_treasure = float(env_info.get("total_treasure", 10))
        treasures_collected = int(env_info.get("treasures_collected", 0))
        total_score = float(env_info.get("total_score", 0.0))
        step_score = float(env_info.get("step_score", 0.0))
        treasure_score = float(env_info.get("treasure_score", 0.0))
        flash_count = int(env_info.get("flash_count", 0))
        collected_buff = int(env_info.get("collected_buff", 0))
        buff_refresh_time = float(env_info.get("buff_refresh_time", 0.0))

        legal_action = self._parse_legal_action(legal_act)

        hero, hero_x, hero_z = self._extract_hero(frame_state, env_info)
        flash_cd = float(hero.get("flash_cooldown", 0.0))
        buff_remain = float(hero.get("buff_remaining_time", 0.0))
        hero_speed = 2.0 if buff_remain > 0.0 else 1.0

        history = self._update_history_memory(hero_x, hero_z, last_action)

        raw_monsters = frame_state.get("monsters", [])
        visible_monster_count = sum(
            1
            for m in raw_monsters
            if isinstance(m, dict) and float(m.get("is_in_view", 0.0)) >= 0.5
        )
        if visible_monster_count > 0:
            self.lost_monster_steps = 0
        else:
            self.lost_monster_steps = min(
                self.lost_monster_steps + 1,
                int(Config.MAX_LOST_MONSTER_STEPS),
            )

        monster_feat, monster_dists, monster_local = self._build_monster_features(
            raw_monsters,
            hero_x,
            hero_z,
        )
        min_dist = float(min(monster_dists)) if monster_dists else Config.MAP_DIAG
        second_dist = float(sorted(monster_dists)[1]) if len(monster_dists) > 1 else Config.MAP_DIAG

        (
            target_feat,
            nearest_treasure_vec,
            nearest_treasure_dist_norm,
            nearest_buff_vec,
            nearest_buff_dist_norm,
            target_memory,
        ) = self._build_target_features(
            frame_state.get("organs", []),
            hero_x,
            hero_z,
            buff_refresh_time,
        )

        map_feat = self._build_local_map_feature(map_info)
        phase_profile = self._get_phase_profile(
            min_dist=min_dist,
            visible_monster_count=visible_monster_count,
        )

        action_safety, action_treasure, action_explore = self._estimate_action_eval(
            map_info=map_info,
            legal_action=legal_action,
            hero_speed=hero_speed,
            monster_local=monster_local,
            nearest_treasure_vec=nearest_treasure_vec,
            nearest_treasure_dist_norm=nearest_treasure_dist_norm,
            nearest_buff_vec=nearest_buff_vec,
            nearest_buff_dist_norm=nearest_buff_dist_norm,
            min_dist=min_dist,
            hero_x=hero_x,
            hero_z=hero_z,
            history=history,
            visible_monster_count=visible_monster_count,
            lost_monster_steps=self.lost_monster_steps,
            phase_profile=phase_profile,
        )

        base_self_feat = np.array(
            [
                _norm(hero_x, Config.MAP_SIZE),                          # 英雄坐标 x
                _norm(hero_z, Config.MAP_SIZE),                          # 英雄坐标 z
                _norm(float(self.step_no), float(max(1, self.max_step))),# 当前步数进度
                _norm(flash_cd, Config.MAX_FLASH_CD),                    # 闪现冷却
                _norm(buff_remain, Config.MAX_BUFF_DURATION),            # buff 剩余持续时间
                _norm(hero_speed, Config.MAX_MONSTER_SPEED),             # 当前速度
                _norm(float(treasures_collected), max(1.0, total_treasure)), # 宝箱收集进度
                _norm(min_dist, Config.MAP_DIAG),                        # 最近怪物距离
                _norm(float(visible_monster_count), 2.0),               # 当前视野内怪物数量
                _norm(float(self.lost_monster_steps), Config.MAX_LOST_MONSTER_STEPS), # 连续丢视野步数
            ],
            dtype=np.float32,
        )

        # 历史与访问记忆特征：告诉 PPO 当前是否卡住、折返、进入新区域
        history_feat = np.array(
            [
                history["prev_move_dx"],                                  # 上一步实际位移 x
                history["prev_move_dz"],                                  # 上一步实际位移 z
                history["last_action_dir_x"],                             # 上一步动作方向 x
                history["last_action_dir_z"],                             # 上一步动作方向 z
                history["reverse_action_flag"],                           # 是否与再上一动作反向
                _norm(float(history["stuck_steps"]), Config.MAX_STUCK_STEPS), # 连续卡住步数
                _norm(float(history["same_cell_streak"]), Config.MAX_REPEAT_STEPS), # 当前格连续停留步数
                _norm(float(history["current_cell_visits"]), Config.MAX_VISIT_COUNT), # 当前格访问次数
                history["entered_new_cell_flag"],                         # 本步是否首次进入新格子
            ],
            dtype=np.float32,
        )

        self_feat = np.concatenate(
            [
                base_self_feat,
                history_feat,
                # 8个方向访问新颖度，顺序与 ACTION_DIRS 一致：
                # [E, NE, N, NW, W, SW, S, SE]
                history["directional_novelty"],
            ]
        ).astype(np.float32)

        action_eval_feat = np.concatenate(
            [action_safety, action_treasure, action_explore]
        ).astype(np.float32)
        # action_eval_feat = [16维动作安全先验, 16维动作逐利先验, 16维动作探索先验]

        feature = np.concatenate(
            [
                self_feat,
                monster_feat,
                target_feat,
                map_feat,
                action_eval_feat,
            ]
        ).astype(np.float32)

        # 防止后续调整特征块时出现维度漂移
        if feature.shape[0] != Config.DIM_OF_OBSERVATION:
            raise ValueError(
                f"feature dim mismatch: got {feature.shape[0]}, expect {Config.DIM_OF_OBSERVATION}"
            )

        remain_info = {
            "step_no": int(self.step_no),
            "hero_x": hero_x,
            "hero_z": hero_z,
            "min_monster_dist": min_dist,
            "second_monster_dist": second_dist,
            "treasure_cnt": treasures_collected,
            "buff_cnt": collected_buff,
            "flash_cnt": flash_count,
            "step_score": step_score,
            "treasure_score": treasure_score,
            "total_score": total_score,
            "action_safety": action_safety.tolist(),
            "action_treasure": action_treasure.tolist(),
            "action_explore": action_explore.tolist(),
            "same_cell_streak": int(history["same_cell_streak"]),
            "current_cell_visits": int(history["current_cell_visits"]),
            "entered_new_cell_flag": float(history["entered_new_cell_flag"]),
            "unique_visit_count": int(len(self.visited_cells)),
            "unique_visit_ratio": float(
                np.clip(
                    len(self.visited_cells)
                    / max(1.0, (Config.MAP_SIZE / Config.VISIT_CELL_SIZE) ** 2),
                    0.0,
                    1.0,
                )
            ),
            "visible_monster_count": int(visible_monster_count),
            "lost_monster_steps": int(self.lost_monster_steps),
            "monster_interval": int(phase_profile["monster_interval"]),
            "monster_speedup": int(phase_profile["monster_speedup"]),
            "phase_id": int(phase_profile["phase_id"]),
            "base_phase_id": int(phase_profile["base_phase_id"]),
            "phase_name": phase_profile["name"],
            "phase_progress": float(phase_profile["phase_progress"]),
            "phase_enter_reward": float(phase_profile["reward_enter_bonus"]),
            "phase_move_reward_cap": float(phase_profile["reward_move_cap"]),
            "phase_repeat_penalty": float(phase_profile["reward_repeat_penalty"]),
            "phase_buff_discovery_reward": float(phase_profile["reward_buff_discovery_bonus"]),
            "phase_treasure_discovery_reward": float(phase_profile["reward_treasure_discovery_bonus"]),
            "eval_explore_weight": float(phase_profile["eval_explore_weight"]),
            "eval_treasure_weight": float(phase_profile["eval_treasure_weight"]),
            "new_treasure_discovery": int(target_memory["new_treasure_discovery"]),
            "new_buff_discovery": int(target_memory["new_buff_discovery"]),
            "known_treasure_count": int(target_memory["known_treasure_count"]),
            "known_buff_count": int(target_memory["known_buff_count"]),
            "last_action": int(last_action) if last_action is not None else -1,
        }
        return feature, legal_action.tolist(), remain_info
