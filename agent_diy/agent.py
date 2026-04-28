#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase DIY PPO.
"""

import os
import numpy as np
import torch

from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)

        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.episode_context = None

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        if self.episode_context is not None:
            env_conf = self.episode_context.get("env_conf", {})
            if isinstance(env_conf, dict):
                self.preprocessor.set_episode_context(
                    monster_interval=env_conf.get("monster_interval"),
                    monster_speedup=env_conf.get("monster_speedup"),
                    max_step=env_conf.get("max_step"),
                )
        self.last_action = -1

    def set_episode_context(self, episode_context):
        self.episode_context = episode_context if isinstance(episode_context, dict) else None
        if self.episode_context is None:
            return

        env_conf = self.episode_context.get("env_conf", {})
        if isinstance(env_conf, dict):
            self.preprocessor.set_episode_context(
                monster_interval=env_conf.get("monster_interval"),
                monster_speedup=env_conf.get("monster_speedup"),
                max_step=env_conf.get("max_step"),
            )

    def predict(self, list_obs_data, list_state=None):
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value, prob = self._run_model(feature, legal_action)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        obs_data, remain_info = self.observation_process(env_obs)

        _, _, prob = self._run_model(obs_data.feature, obs_data.legal_action)
        legal = np.array(obs_data.legal_action, dtype=np.float32)
        final_prob = np.array(prob, dtype=np.float32)

        # Safety-aware policy fusion in evaluation:
        # blend model policy with action-level heuristic prior from feature engineering.
        safety = np.array(remain_info.get("action_safety", []), dtype=np.float32)
        treasure = np.array(remain_info.get("action_treasure", []), dtype=np.float32)
        explore = np.array(remain_info.get("action_explore", []), dtype=np.float32)
        if (
            safety.shape[0] == Config.ACTION_NUM
            and treasure.shape[0] == Config.ACTION_NUM
            and explore.shape[0] == Config.ACTION_NUM
        ):
            min_dist = float(remain_info.get("min_monster_dist", Config.MAP_DIAG))
            step_no = int(remain_info.get("step_no", 0))
            phase_id = int(remain_info.get("phase_id", 1))
            eval_treasure_weight = float(remain_info.get("eval_treasure_weight", 0.30))
            eval_explore_weight = float(remain_info.get("eval_explore_weight", 0.15))

            prior_logits = 2.05 * safety + eval_treasure_weight * treasure + eval_explore_weight * explore
            if min_dist <= 4.0:
                prior_logits[8:] += 0.40 * np.minimum(safety[8:], 0.0)
            elif phase_id == 0:
                prior_logits[:8] += 0.08 * np.maximum(explore[:8], 0.0)

            prior_prob = self._legal_soft_max(prior_logits, legal)
            if min_dist <= 4.0:
                mix = 0.45
            elif phase_id == 0:
                mix = 0.26
            elif phase_id == 1:
                mix = 0.20
            else:
                mix = 0.14
            final_prob = (1.0 - mix) * final_prob + mix * prior_prob

            # 评估阶段对显著不安全的闪现做软屏蔽，优先抑制开局贴脸暴毙。
            if min_dist <= 4.5 or step_no <= 80:
                unsafe_thresh = -0.45 if step_no <= 80 else -0.75
                unsafe_flash_mask = safety[8:] <= unsafe_thresh
                if np.any(unsafe_flash_mask):
                    unsafe_flash_idx = np.arange(8, Config.ACTION_NUM)[unsafe_flash_mask]
                    final_prob[unsafe_flash_idx] *= 0.02

        final_prob = final_prob * legal
        s = np.sum(final_prob)
        if s <= 1e-8:
            final_prob = legal / np.maximum(np.sum(legal), 1.0)
        else:
            final_prob = final_prob / s

        action = int(np.argmax(final_prob))
        self.last_action = action
        return action

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        if path is None:
            return

        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.logger.warning(f"正在调用-保存 {model_file_path}")
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)

        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        if path is None:
            return

        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.logger.warning(f"正在调用-加载 {model_file_path}")
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"model file {model_file_path} not found, skip loading")
            return

        state = torch.load(model_file_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        if not isinstance(state, dict):
            if self.logger:
                self.logger.warning(f"unexpected checkpoint format: {model_file_path}, skip loading")
            return

        current = self.model.state_dict()
        matched = {}
        skipped = []
        for k, v in state.items():
            if k in current and hasattr(v, "shape") and current[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)

        if not matched:
            if self.logger:
                self.logger.warning(
                    f"no compatible params in checkpoint {model_file_path}, "
                    "use random initialized model"
                )
            return

        current.update(matched)
        self.model.load_state_dict(current, strict=False)
        if self.logger:
            self.logger.info(
                f"load model {model_file_path} partially: "
                f"matched={len(matched)} skipped={len(skipped)}"
            )

    def observation_process(self, obs, preprocessor=None, extra_info=None):
        feature, legal_action, remain_info = self.preprocessor.feature_process(obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.detach().cpu().numpy()[0]
        value_np = value.detach().cpu().numpy()[0]

        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)
        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
