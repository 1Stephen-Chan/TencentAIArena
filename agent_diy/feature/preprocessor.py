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


# 方向映射,用于移动和闪现
ACTION_DIRS = [
    # 0: 东 E (Right)
    (1, 0),
    # 1: 东北 NE (Right-Up)
    (1, -1),
    # 2: 北 N (Up)
    (0, -1),
    # 3: 西北 NW (Left-Up)
    (-1, -1),
    # 4: 西 W (Left)
    (-1, 0),
    # 5: 西南 SW (Left-Down)
    (-1, 1),
    # 6: 南 S (Down)
    (0, 1),
    # 7: 东南 SE (Right-Down)
    (1, 1),
]


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    denom = float(v_max - v_min)
    if abs(denom) < 1e-6:
        return 0.0
    return (v - v_min) / denom


def _dir_to_vec(direction):
    """
    方向编号转向量
    0=overlap/invalid, 1=E, 2=NE, 3=N, 4=NW, 5=W, 6=SW, 7=S, 8=SE
    """
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
        self.max_step = 1000
        self.prev_pos = None
        self.prev_cell = None
        self.stuck_steps = 0
        self.visit_counter = {}
        self.monster_speedup_seen = False
        self.last_monster_vec = None  # 最后已知怪物方向
        self.last_monster_dist = Config.MAP_DIAG  # 最后已知怪物距离
        self.steps_since_last_seen = 0  # 失去视野的步数
        self.last_move_vec = None  # 上一步移动方向（用于保持惯性）

    def _parse_legal_action(self, legal_act_raw):
        legal = np.ones(Config.ACTION_NUM, dtype=np.float32)
        if (
            isinstance(legal_act_raw, (list, tuple, np.ndarray))
            and len(legal_act_raw) > 0
        ):
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
        if isinstance(heroes, list):
            hero = heroes[0] if heroes else {}
        elif isinstance(heroes, dict):
            hero = heroes
        else:
            hero = {}

        hero_pos = hero.get("pos", env_info.get("pos", {}))
        if not isinstance(hero_pos, dict):
            hero_pos = {}

        hero_x = float(hero_pos.get("x", 0.0))
        hero_z = float(hero_pos.get("z", 0.0))
        return hero, hero_x, hero_z

    def _bucket_center_dist(self, bucket):
        # bucket 0..5 over [0,180], use center value for approximation
        centers = [15.0, 45.0, 75.0, 105.0, 135.0, 165.0]
        b = int(np.clip(bucket, 0, 5))
        return centers[b]

    def _update_explore_state(self, hero_x, hero_z):
        cell_size = max(1.0, float(Config.EXPLORE_CELL_SIZE))
        cell = (
            int(math.floor(hero_x / cell_size)),
            int(math.floor(hero_z / cell_size)),
        )
        visit_count = int(self.visit_counter.get(cell, 0)) + 1
        is_new_area = 1.0 if visit_count == 1 else 0.0
        self.visit_counter[cell] = visit_count

        if self.prev_pos is None:
            move_l1 = 0.0
            self.stuck_steps = 0
        else:
            move_l1 = abs(hero_x - self.prev_pos[0]) + abs(hero_z - self.prev_pos[1])
            if move_l1 < Config.STUCK_MOVE_THRESHOLD:
                self.stuck_steps += 1
            elif cell == self.prev_cell and move_l1 < (2.0 * Config.STUCK_MOVE_THRESHOLD):
                self.stuck_steps += 1
            else:
                self.stuck_steps = max(0, self.stuck_steps - 2)

        # 更新上一步移动方向（用于保持惯性）
        if self.prev_pos is not None:
            dx = hero_x - self.prev_pos[0]
            dz = hero_z - self.prev_pos[1]
            dist = math.sqrt(dx * dx + dz * dz)
            if dist > 0.1:
                self.last_move_vec = (dx / dist, dz / dist)
        
        self.prev_pos = (hero_x, hero_z)
        self.prev_cell = cell
        return cell, visit_count, is_new_area, float(move_l1)

    def _build_monster_features(self, monsters, hero_x, hero_z):
        if isinstance(monsters, dict):
            monsters = [monsters]
        if not isinstance(monsters, list):
            monsters = []

        feats = []
        dists = []
        monster_local = []
        visible_monster_cnt = 0
        max_monster_speed = 1.0

        for i in range(2):
            if i < len(monsters) and isinstance(monsters[i], dict):
                m = monsters[i]
                m_pos = m.get("pos", {})
                has_pos = isinstance(m_pos, dict) and ("x" in m_pos) and ("z" in m_pos)

                in_view = float(m.get("is_in_view", 1.0 if has_pos else 0.0))
                speed = float(m.get("speed", 1.0))
                max_monster_speed = max(max_monster_speed, speed)
                if in_view > 0.5:
                    visible_monster_cnt += 1
                bucket = float(m.get("hero_l2_distance", 5.0))
                rel_dir = int(m.get("hero_relative_direction", 0))
                dir_x, dir_z = _dir_to_vec(rel_dir)

                if has_pos:
                    mx = float(m_pos.get("x", hero_x))
                    mz = float(m_pos.get("z", hero_z))
                    dx = mx - hero_x
                    dz = mz - hero_z
                    dist = math.sqrt(dx * dx + dz * dz)
                else:
                    dist = self._bucket_center_dist(bucket)
                    dx = dir_x * dist
                    dz = dir_z * dist

                threat = 1.0 - _norm(dist, Config.MAP_DIAG)

                feats.extend(
                    [
                        1.0,
                        in_view,
                        _norm(dist, Config.MAP_DIAG),
                        _norm(bucket, 5.0),
                        _norm(speed, Config.MAX_MONSTER_SPEED),
                        dir_x,
                        dir_z,
                        float(np.clip(dx / 30.0, -1.0, 1.0)),
                        float(np.clip(dz / 30.0, -1.0, 1.0)),
                        threat,
                    ]
                )

                dists.append(dist)
                monster_local.append(
                    {"dx": float(dx), "dz": float(dz), "speed": speed, "dist": dist}
                )
            else:
                feats.extend([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                dists.append(Config.MAP_DIAG)

        return (
            np.array(feats, dtype=np.float32),
            dists,
            monster_local,
            visible_monster_cnt,
            float(max_monster_speed),
        )

    def _build_target_features(self, organs):
        if isinstance(organs, dict):
            organs = [organs]
        if not isinstance(organs, list):
            organs = []

        treasures = []
        buffs = []

        for organ in organs:
            if not isinstance(organ, dict):
                continue

            status = int(organ.get("status", 1))
            if status != 1:
                continue

            sub_type = int(organ.get("sub_type", 0))
            dist_bucket = float(organ.get("hero_l2_distance", 5.0))
            rel_dir = int(organ.get("hero_relative_direction", 0))
            dir_x, dir_z = _dir_to_vec(rel_dir)
            dist_norm = _norm(dist_bucket, 5.0)

            item = [1.0, dist_norm, dir_x, dir_z]
            if sub_type == 1:
                treasures.append(item)
            elif sub_type == 2:
                buffs.append(item)

        treasures.sort(key=lambda x: x[1])
        buffs.sort(key=lambda x: x[1])

        feat = []
        for i in range(4):
            feat.extend(treasures[i] if i < len(treasures) else [0.0, 1.0, 0.0, 0.0])
        for i in range(4):
            feat.extend(buffs[i] if i < len(buffs) else [0.0, 1.0, 0.0, 0.0])

        nearest_treasure_vec = None
        nearest_treasure_dist_norm = 1.0
        if treasures:
            nearest_treasure_dist_norm = float(treasures[0][1])
            nearest_treasure_vec = (float(treasures[0][2]), float(treasures[0][3]))

        nearest_buff_vec = None
        nearest_buff_dist_norm = 1.0
        if buffs:
            nearest_buff_dist_norm = float(buffs[0][1])
            nearest_buff_vec = (float(buffs[0][2]), float(buffs[0][3]))

        return (
            np.array(feat, dtype=np.float32),
            treasures,
            buffs,
            nearest_treasure_vec,
            nearest_treasure_dist_norm,
            nearest_buff_vec,
            nearest_buff_dist_norm,
        )

    def _build_local_map_feature(self, map_info):
        # 提取完整的21×21视野地图 (441维)
        feat = np.zeros(441, dtype=np.float32)

        if not isinstance(map_info, list) or len(map_info) == 0:
            return feat

        rows = len(map_info)
        cols = len(map_info[0]) if isinstance(map_info[0], list) else 0
        if cols <= 0:
            return feat

        center_r = rows // 2
        center_c = cols // 2

        idx = 0
        for r in range(center_r - 10, center_r + 11):
            for c in range(center_c - 10, center_c + 11):
                if 0 <= r < rows and 0 <= c < cols:
                    feat[idx] = 1.0 if float(map_info[r][c]) != 0.0 else 0.0
                idx += 1

        return feat

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

    def _evaluate_away_direction(self, map_info, away_vec):
        """评估基于完整地图的最佳远离方向，返回每个动作的可行性分数。"""
        if not isinstance(map_info, list) or len(map_info) == 0 or away_vec is None:
            return None
        
        rows = len(map_info)
        cols = len(map_info[0]) if isinstance(map_info[0], list) else 0
        if cols <= 0:
            return None
        
        center_r = rows // 2
        center_c = cols // 2
        
        # 评估8个方向的远离可行性（使用与_action_vec一致的方向定义）
        # 方向: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
        away_scores = np.zeros(8, dtype=np.float32)
        
        for direction in range(8):
            # 使用与 _action_vec 一致的方向计算
            angle = direction * (np.pi / 4)
            dir_x = np.cos(angle)
            dir_z = -np.sin(angle)
            
            # 计算该方向与远离方向的对齐度
            align = dir_x * away_vec[0] + dir_z * away_vec[1]
            
            if align <= 0:
                away_scores[direction] = 0.0
                continue
            
            # 沿该方向检查可通行距离
            max_passable_dist = 0
            for dist in range(1, min(rows, cols) // 2):
                check_r = center_r + int(round(dir_z * dist))
                check_c = center_c + int(round(dir_x * dist))
                
                if 0 <= check_r < rows and 0 <= check_c < cols:
                    if float(map_info[check_r][check_c]) != 0.0:
                        max_passable_dist = dist
                    else:
                        break
                else:
                    break
            
            # 分数 = 方向对齐度 * 可通行距离
            away_scores[direction] = align * min(max_passable_dist / 10.0, 1.0)
        
        return away_scores

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
        speedup_active,
        hero_buff_active,
    ):
        safety = np.full(Config.ACTION_NUM, -1.0, dtype=np.float32)
        treasure = np.full(Config.ACTION_NUM, -1.0, dtype=np.float32)
        buff = np.full(Config.ACTION_NUM, -1.0, dtype=np.float32)

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
                for m in monster_local:
                    dx = m["dx"] - dst_x
                    dz = m["dz"] - dst_z
                    d = math.sqrt(dx * dx + dz * dz) - 0.5 * float(m.get("speed", 1.0))
                    min_after = min(min_after, d)
            else:
                min_after = Config.MAP_DIAG

            s = float(np.clip((min_after - 2.0) / 8.0, -1.0, 1.0))
            if act < 8 and moved_dist < 0.5:
                s -= 0.35
            if act >= 8 and min_dist > (5.0 if speedup_active else 6.0):
                # Keep flash for emergencies when current risk is low
                s -= 0.28 if not speedup_active else 0.12

            t = 0.0
            if nearest_treasure_vec is not None:
                ax, az = _action_vec(act)
                tx, tz = nearest_treasure_vec
                align = float(ax * tx + az * tz)
                dist_weight = float(np.clip(1.2 - nearest_treasure_dist_norm, 0.0, 1.0))
                t = align * dist_weight
                if act >= 8:
                    t *= 0.65
                if speedup_active:
                    t *= 0.55
                t *= 0.5 + 0.5 * float(np.clip(s, -1.0, 1.0))

            b = 0.0
            if nearest_buff_vec is not None and not hero_buff_active:
                ax, az = _action_vec(act)
                bx, bz = nearest_buff_vec
                align = float(ax * bx + az * bz)
                dist_weight = float(np.clip(1.2 - nearest_buff_dist_norm, 0.0, 1.0))
                b = align * dist_weight
                if act >= 8:
                    b *= 0.50 if not speedup_active else 0.75
                b *= 0.5 + 0.5 * float(np.clip(s, -1.0, 1.0))
                if speedup_active:
                    b *= 1.25

            safety[act] = float(np.clip(s, -1.0, 1.0))
            treasure[act] = float(np.clip(t, -1.0, 1.0))
            buff[act] = float(np.clip(b, -1.0, 1.0))

        return safety, treasure, buff

    def feature_process(self, env_obs, last_action):
        if not isinstance(env_obs, dict):
            env_obs = {}

        observation = env_obs.get("observation", {})
        if not isinstance(observation, dict):
            observation = {}

        frame_state = observation.get("frame_state", {})
        if not isinstance(frame_state, dict):
            frame_state = {}

        env_info = observation.get("env_info", {})
        if not isinstance(env_info, dict):
            env_info = {}

        self.step_no = int(
            observation.get("step_no", env_info.get("step_no", self.step_no))
        )
        self.max_step = int(env_info.get("max_step", self.max_step))

        legal_raw = observation.get(
            "legal_act", observation.get("legal_action", [True] * Config.ACTION_NUM)
        )
        legal_action = self._parse_legal_action(legal_raw)

        hero, hero_x, hero_z = self._extract_hero(frame_state, env_info)
        _, visit_count, is_new_area, move_l1 = self._update_explore_state(hero_x, hero_z)

        flash_cd = float(hero.get("flash_cooldown", 0.0))
        buff_remain = float(hero.get("buff_remaining_time", 0.0))
        hero_speed = float(hero.get("speed", 1.0))
        hero_buff_active = 1.0 if buff_remain > 0.5 else 0.0

        (
            monster_feat,
            monster_dists,
            monster_local,
            visible_monster_cnt,
            max_monster_speed,
        ) = self._build_monster_features(
            frame_state.get("monsters", []),
            hero_x,
            hero_z,
        )
        if max_monster_speed >= Config.MONSTER_SPEEDUP_THRESHOLD:
            self.monster_speedup_seen = True
        min_dist = float(min(monster_dists)) if monster_dists else Config.MAP_DIAG
        second_dist = (
            float(sorted(monster_dists)[1])
            if len(monster_dists) > 1
            else Config.MAP_DIAG
        )

        # 更新最后已知怪物位置记忆
        if visible_monster_cnt > 0 and monster_local:
            # 有怪物视野，更新最后已知位置
            nearest_monster = min(monster_local, key=lambda x: x["dist"])
            dx, dz = nearest_monster["dx"], nearest_monster["dz"]
            dist = nearest_monster["dist"]
            # 归一化方向向量
            if dist > 0.1:
                self.last_monster_vec = (dx / dist, dz / dist)
                self.last_monster_dist = dist
            self.steps_since_last_seen = 0
        elif min_dist <= 10.0 and monster_local:
            # 怪物很近但不在视野内（可能被墙挡住），也更新记忆
            nearest_monster = min(monster_local, key=lambda x: x["dist"])
            dx, dz = nearest_monster["dx"], nearest_monster["dz"]
            dist = nearest_monster["dist"]
            if dist > 0.1:
                self.last_monster_vec = (dx / dist, dz / dist)
                self.last_monster_dist = dist
            self.steps_since_last_seen = 0
        else:
            # 失去视野，增加计数
            self.steps_since_last_seen += 1

        (
            target_feat,
            treasures,
            buffs,
            nearest_treasure_vec,
            nearest_treasure_dist_norm,
            nearest_buff_vec,
            nearest_buff_dist_norm,
        ) = self._build_target_features(frame_state.get("organs", []))

        # 评估基于地图的远离方向（失去视野时使用）
        away_scores = None
        if visible_monster_cnt <= 0 and self.last_monster_vec is not None:
            map_info = observation.get("map_info", [])
            away_scores = self._evaluate_away_direction(map_info, self.last_monster_vec)

        map_info = observation.get("map_info", [])
        map_feat = self._build_local_map_feature(map_info)

        action_safety, action_treasure, action_buff = self._estimate_action_eval(
            map_info=map_info,
            legal_action=legal_action,
            hero_speed=hero_speed,
            monster_local=monster_local,
            nearest_treasure_vec=nearest_treasure_vec,
            nearest_treasure_dist_norm=nearest_treasure_dist_norm,
            nearest_buff_vec=nearest_buff_vec,
            nearest_buff_dist_norm=nearest_buff_dist_norm,
            min_dist=min_dist,
            speedup_active=self.monster_speedup_seen,
            hero_buff_active=bool(hero_buff_active),
        )

        total_treasure = float(env_info.get("total_treasure", max(1, len(treasures))))
        treasure_cnt = int(
            hero.get("treasure_collected_count", env_info.get("treasures_collected", 0))
        )

        total_buff = float(env_info.get("total_buff", 2.0))
        buff_cnt = int(env_info.get("collected_buff", 0))

        step_norm = _norm(float(self.step_no), float(max(1, self.max_step)))
        remain_step_norm = _norm(
            float(max(0, self.max_step - self.step_no)), float(max(1, self.max_step))
        )

        self_feat = np.array(
            [
                _norm(hero_x, Config.MAP_SIZE),
                _norm(hero_z, Config.MAP_SIZE),
                step_norm,
                remain_step_norm,
                _norm(flash_cd, Config.MAX_FLASH_CD),
                _norm(buff_remain, Config.MAX_BUFF_DURATION),
                _norm(hero_speed, Config.MAX_MONSTER_SPEED),
                _norm(min_dist, Config.MAP_DIAG),
                _norm(second_dist, Config.MAP_DIAG),
                _norm(float(treasure_cnt), max(1.0, total_treasure)),
            ],
            dtype=np.float32,
        )

        action_eval_feat = np.concatenate([action_safety, action_treasure]).astype(
            np.float32
        )
        risk_feat = np.array([1.0 - _norm(min_dist, Config.MAP_DIAG)], dtype=np.float32)

        feature = np.concatenate(
            [
                self_feat,
                monster_feat,
                target_feat,
                map_feat,
                legal_action.astype(np.float32),
                action_eval_feat,
                risk_feat,
            ]
        ).astype(np.float32)

        if feature.shape[0] != Config.FEATURE_LEN:
            if feature.shape[0] > Config.FEATURE_LEN:
                feature = feature[: Config.FEATURE_LEN]
            else:
                feature = np.concatenate(
                    [
                        feature,
                        np.zeros(
                            Config.FEATURE_LEN - feature.shape[0], dtype=np.float32
                        ),
                    ]
                )

        remain_info = {
            "step_no": int(self.step_no),
            "hero_x": hero_x,
            "hero_z": hero_z,
            "min_monster_dist": min_dist,
            "second_monster_dist": second_dist,
            "treasure_cnt": treasure_cnt,
            "buff_cnt": buff_cnt,
            "flash_cnt": int(env_info.get("flash_count", 0)),
            "step_score": float(
                hero.get("step_score", env_info.get("step_score", 0.0))
            ),
            "treasure_score": float(
                hero.get("treasure_score", env_info.get("treasure_score", 0.0))
            ),
            "total_score": float(env_info.get("total_score", 0.0)),
            "visible_monster_cnt": int(visible_monster_cnt),
            "max_monster_speed": float(max_monster_speed),
            "monster_speedup_seen": 1.0 if self.monster_speedup_seen else 0.0,
            "hero_buff_active": float(hero_buff_active),
            "nearest_treasure_dist_norm": float(nearest_treasure_dist_norm),
            "nearest_buff_dist_norm": float(nearest_buff_dist_norm),
            "visit_count": int(visit_count),
            "is_new_area": float(is_new_area),
            "stuck_steps": int(self.stuck_steps),
            "move_l1": float(move_l1),
            "action_safety": action_safety.tolist(),
            "action_treasure": action_treasure.tolist(),
            "action_buff": action_buff.tolist(),
            "last_action": int(last_action) if last_action is not None else -1,
            "last_monster_vec": self.last_monster_vec if self.last_monster_vec else (0.0, 0.0),
            "steps_since_last_seen": self.steps_since_last_seen,
            "away_scores": away_scores.tolist() if away_scores is not None else [0.0] * 8,
            "last_move_vec": self.last_move_vec if self.last_move_vec else None,
        }

        return feature, legal_action.tolist(), remain_info
