#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase DIY Agent.
峡谷追猎 DIY 智能体 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import torch
from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_SHAPE[0]
        self.value_num = Config.VALUE_SHAPE[0]
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        self.entropy_coef = Config.ENTROPY_COEFF

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.MAX_GRAD_NORM)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "reward": round(reward.mean().item(), 4),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
        """
        prob_dist = self._masked_softmax(logits, legal_action)

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        total_loss = self.vf_coef * value_loss + policy_loss - self.entropy_coef * entropy_loss

        return total_loss, [value_loss, policy_loss, entropy_loss]

    def _masked_softmax(self, logits, legal_action):
        """Masked softmax for legal actions with numerical stability.

        合法动作掩码下的 softmax（数值稳定版）。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
