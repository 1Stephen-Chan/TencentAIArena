#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase DIY agent.
"""

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            make_fc_layer(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            make_fc_layer(128, 128),
            nn.GELU(),
        )

        self.actor_head = make_fc_layer(128, action_num)

        # Action-eval prior branch:
        # consume per-action safety/treasure features and provide actor bias logits.
        self.prior_head = nn.Sequential(
            make_fc_layer(Config.ACTION_EVAL_DIM, 64),
            nn.GELU(),
            make_fc_layer(64, action_num),
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.35))

        self.critic_head = make_fc_layer(128, value_num)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)

        action_eval = obs[:, Config.ACTION_EVAL_START : Config.ACTION_EVAL_END]
        prior_logits = self.prior_head(action_eval)
        logits = logits + torch.tanh(self.prior_scale) * prior_logits

        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
