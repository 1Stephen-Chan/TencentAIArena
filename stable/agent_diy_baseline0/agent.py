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

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

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
        buff = np.array(remain_info.get("action_buff", []), dtype=np.float32)
        if (
            safety.shape[0] == Config.ACTION_NUM
            and treasure.shape[0] == Config.ACTION_NUM
            and buff.shape[0] == Config.ACTION_NUM
        ):
            min_dist = float(remain_info.get("min_monster_dist", Config.MAP_DIAG))
            speedup_active = bool(remain_info.get("monster_speedup_seen", 0.0) >= 0.5)
            hero_buff_active = bool(remain_info.get("hero_buff_active", 0.0) >= 0.5)
            visible_monster_cnt = int(remain_info.get("visible_monster_cnt", 0))

            if speedup_active:
                prior_logits = 2.0 * safety + 0.35 * treasure
                if not hero_buff_active:
                    prior_logits += 0.95 * buff
            else:
                prior_logits = 1.15 * safety + 1.00 * treasure
                if not hero_buff_active:
                    prior_logits += 0.45 * buff

            if min_dist <= 4.0:
                prior_logits[8:] += 0.35 * safety[8:]
            elif not speedup_active:
                prior_logits[8:] -= 0.30

            if visible_monster_cnt <= 0 and not speedup_active:
                prior_logits[8:] -= 0.10

            prior_prob = self._legal_soft_max(prior_logits, legal)
            mix = 0.65 if (speedup_active or min_dist <= 4.0) else 0.28
            final_prob = (1.0 - mix) * final_prob + mix * prior_prob

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
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)

        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        if path is None:
            return

        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"model file {model_file_path} not found, skip loading")
            return

        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

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
