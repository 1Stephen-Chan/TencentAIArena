#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

DIY Agent class for Gorge Chase PPO.
峡谷追猎 DIY Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.START_LR,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=feature,
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

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

    def exploit(self, list_obs_data):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, prob = self._run_model(feature, legal_action)

        action = self._legal_sample(prob, use_max=True)
        d_action = action

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model.

        保存模型。
        """
        if path is None:
            path = "agent_diy/model"
        torch.save(self.model.state_dict(), f"{path}/model_{id}.pt")

    def load_model(self, path=None, id="1"):
        """Load model.

        加载模型。
        """
        if path is None:
            path = "agent_diy/model"
        try:
            self.model.load_state_dict(torch.load(f"{path}/model_{id}.pt", map_location=self.device))
        except:
            self.logger.info(f"Failed to load model {path}/model_{id}.pt, use random init")

    def action_process(self, act_data, is_stochastic=True):
        """Process ActData to environment action.

        处理 ActData 转换为环境动作。
        """
        # 处理可能是标量的情况
        if is_stochastic:
            action_value = act_data.action[0] if isinstance(act_data.action, (list, np.ndarray, tuple)) else act_data.action
        else:
            action_value = act_data.d_action[0] if isinstance(act_data.d_action, (list, np.ndarray, tuple)) else act_data.d_action
        action = int(action_value)
        self.last_action = action
        return action

    def _run_model(self, feature, legal_action):
        """Run model forward.

        模型前向传播。
        """
        # 将特征转换为张量（如果是numpy数组）
        if isinstance(feature, dict):
            feature_dict = {
                'hero': torch.FloatTensor(feature['hero']).to(self.device) if 'hero' in feature else None,
                'monsters': torch.FloatTensor(feature['monsters']).to(self.device) if 'monsters' in feature else None,
                'treasures': torch.FloatTensor(feature['treasures']).to(self.device) if 'treasures' in feature else None,
                'buffs': torch.FloatTensor(feature['buffs']).to(self.device) if 'buffs' in feature else None,
                'progress': torch.FloatTensor(feature['progress']).to(self.device) if 'progress' in feature else None,
                'map': torch.FloatTensor(feature['map']).to(self.device) if 'map' in feature else None,
                'legal_action': torch.FloatTensor(legal_action).to(self.device),
            }
            # 添加batch维度
            for key in ['hero', 'monsters', 'treasures', 'buffs', 'progress', 'map']:
                if feature_dict[key] is not None:
                    feature_dict[key] = feature_dict[key].unsqueeze(0)
            feature_dict['legal_action'] = feature_dict['legal_action'].unsqueeze(0)
        else:
            # 兼容旧格式（展平向量）
            feature_dict = torch.FloatTensor(feature).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(feature_dict)
            logits = logits.squeeze(0)
            value = value.squeeze(0).item()

            legal_action_tensor = torch.FloatTensor(legal_action).to(self.device)
            masked_logits = logits + (1 - legal_action_tensor) * -1e10
            prob = torch.softmax(masked_logits, dim=-1)

        return logits, value, prob

    def _flatten_features(self, feature_dict):
        """将结构化特征字典展平为向量.
        
        Args:
            feature_dict: 包含结构化特征的字典
            
        Returns:
            numpy.ndarray: 展平后的特征向量
        """
        import numpy as np
        
        features = []
        
        # 按固定顺序拼接所有特征
        # 1. 英雄特征 [10]
        if 'hero' in feature_dict:
            features.append(np.array(feature_dict['hero']).flatten())
        
        # 2. 怪物特征 [2, 8] -> [16]
        if 'monsters' in feature_dict:
            features.append(np.array(feature_dict['monsters']).flatten())
        
        # 3. 宝箱特征 [4, 4] -> [16]
        if 'treasures' in feature_dict:
            features.append(np.array(feature_dict['treasures']).flatten())
        
        # 4. Buff特征 [2, 4] -> [8]
        if 'buffs' in feature_dict:
            features.append(np.array(feature_dict['buffs']).flatten())
        
        # 5. 进度特征 [6]
        if 'progress' in feature_dict:
            features.append(np.array(feature_dict['progress']).flatten())
        
        # 6. 地图特征 [1, 21, 21] -> [441]
        if 'map' in feature_dict:
            features.append(np.array(feature_dict['map']).flatten())
        
        return np.concatenate(features)

    def _legal_sample(self, prob, use_max=False):
        """Sample action from legal action probability distribution.

        从合法动作概率分布中采样动作。
        """
        if use_max:
            action = torch.argmax(prob).item()
        else:
            action = torch.multinomial(prob, 1).item()
        return action
