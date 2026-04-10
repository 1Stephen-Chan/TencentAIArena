#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase DIY Agent.
峡谷追猎 DIY 智能体神经网络模型。
"""

import torch
import torch.nn as nn
import numpy as np

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Single MLP backbone + Actor/Critic dual heads.

    单 MLP 骨干 + Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy"
        self.device = device

        input_dim = Config.FEATURE_VECTOR_SHAPE[0]
        hidden_dim = 256
        mid_dim = 128
        action_num = Config.ACTION_SHAPE[0]
        value_num = Config.VALUE_SHAPE[0]

        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )

        self.actor_head = make_fc_layer(mid_dim, action_num)

        self.critic_head = make_fc_layer(mid_dim, value_num)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
