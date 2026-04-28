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


def make_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    nn.init.orthogonal_(conv.weight.data)
    nn.init.zeros_(conv.bias.data)
    return conv


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy"
        self.device = device

        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        map_embed_dim = 96

        self.map_encoder = nn.Sequential(
            make_conv_layer(Config.LOCAL_MAP_CHANNELS, 8),
            nn.GELU(),
            make_conv_layer(8, 16, stride=2),
            nn.GELU(),
            make_conv_layer(16, 32, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            make_fc_layer(32 * 3 * 3, map_embed_dim),
            nn.LayerNorm(map_embed_dim),
            nn.GELU(),
        )

        self.backbone = nn.Sequential(
            make_fc_layer(Config.NON_SPATIAL_DIM + map_embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            make_fc_layer(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            make_fc_layer(128, 128),
            nn.GELU(),
        )

        self.actor_head = make_fc_layer(128, action_num)

        # 动作先验分支：
        # 消费 safety/treasure/explore 三组动作级特征，为 actor 提供偏置 logits。
        self.prior_head = nn.Sequential(
            make_fc_layer(Config.ACTION_EVAL_DIM, 64),
            nn.GELU(),
            make_fc_layer(64, action_num),
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.35))

        self.critic_head = make_fc_layer(128, value_num)

    def forward(self, obs, inference=False):
        state_feat = obs[:, : Config.MAP_START]
        map_feat = obs[:, Config.MAP_START : Config.MAP_END]
        action_eval = obs[:, Config.ACTION_EVAL_START : Config.ACTION_EVAL_END]

        map_feat = map_feat.view(
            obs.shape[0],
            Config.LOCAL_MAP_CHANNELS,
            Config.LOCAL_MAP_VIEW_SIZE,
            Config.LOCAL_MAP_VIEW_SIZE,
        )

        map_hidden = self.map_encoder(map_feat)
        hidden_input = torch.cat([state_feat, action_eval, map_hidden], dim=1)
        hidden = self.backbone(hidden_input)
        logits = self.actor_head(hidden)

        prior_logits = self.prior_head(action_eval)
        logits = logits + torch.tanh(self.prior_scale) * prior_logits

        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
