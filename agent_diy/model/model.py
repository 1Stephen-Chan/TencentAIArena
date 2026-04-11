#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Structured Neural Network for Gorge Chase DIY Agent.
结构化神经网络模型：
- 标量特征 → MLP
- 地图特征 → CNN
- 多个实体 → 分别编码 → 融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization."""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class EntityEncoder(nn.Module):
    """实体编码器：将同类实体编码为统一维度."""
    
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc = make_fc_layer(input_dim, embed_dim)
        
    def forward(self, x):
        # x: [batch, num_entities, input_dim]
        return F.relu(self.fc(x))


class StructuredNetwork(nn.Module):
    """结构化网络：处理分组特征."""
    
    def __init__(self):
        super().__init__()
        
        # 实体编码器 / Entity encoders
        self.hero_encoder = make_fc_layer(Config.HERO_FEATURE_DIM, Config.EMBED_DIM)
        self.monster_encoder = EntityEncoder(Config.MONSTER_FEATURE_DIM, Config.EMBED_DIM)
        self.treasure_encoder = EntityEncoder(Config.TREASURE_FEATURE_DIM, Config.EMBED_DIM)
        self.buff_encoder = EntityEncoder(Config.BUFF_FEATURE_DIM, Config.EMBED_DIM)
        self.progress_encoder = make_fc_layer(Config.PROGRESS_FEATURE_DIM, Config.EMBED_DIM)
        
        # 实体融合：决定关注哪个实体 / Entity fusion
        # 使用简单的加权平均，权重通过网络学习
        self.fusion_attention = nn.Sequential(
            make_fc_layer(Config.EMBED_DIM * 5, 64),
            nn.ReLU(),
            make_fc_layer(64, 5),  # 5组实体的注意力权重
        )
        
        # 地图CNN / Map CNN
        self.map_cnn = nn.Sequential(
            # 输入: [batch, 1, 5, 5]
            nn.Conv2d(Config.MAP_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            make_fc_layer(32 * Config.MAP_SIZE * Config.MAP_SIZE, Config.MAP_CNN_HIDDEN),
            nn.ReLU(),
        )
        
        # 融合层 / Fusion layer
        fusion_input_dim = Config.EMBED_DIM + Config.MAP_CNN_HIDDEN + Config.ACTION_SHAPE[0]
        self.fusion = nn.Sequential(
            make_fc_layer(fusion_input_dim, Config.FUSION_HIDDEN),
            nn.ReLU(),
            make_fc_layer(Config.FUSION_HIDDEN, Config.FUSION_HIDDEN // 2),
            nn.ReLU(),
        )
        
        # Actor & Critic heads
        self.actor_head = make_fc_layer(Config.FUSION_HIDDEN // 2, Config.ACTION_SHAPE[0])
        self.critic_head = make_fc_layer(Config.FUSION_HIDDEN // 2, Config.VALUE_SHAPE[0])
        
    def encode_entities(self, hero, monsters, treasures, buffs, progress):
        """编码所有实体."""
        # Hero: [batch, HERO_FEATURE_DIM] -> [batch, EMBED_DIM]
        hero_emb = F.relu(self.hero_encoder(hero))
        
        # Monsters: [batch, MAX_MONSTERS, MONSTER_FEATURE_DIM] -> [batch, MAX_MONSTERS, EMBED_DIM]
        monster_emb = self.monster_encoder(monsters)
        monster_emb = monster_emb.mean(dim=1)  # 平均池化 [batch, EMBED_DIM]
        
        # Treasures: [batch, MAX_TREASURES, TREASURE_FEATURE_DIM] -> [batch, EMBED_DIM]
        treasure_emb = self.treasure_encoder(treasures)
        treasure_emb = treasure_emb.mean(dim=1)
        
        # Buffs: [batch, MAX_BUFFS, BUFF_FEATURE_DIM] -> [batch, EMBED_DIM]
        buff_emb = self.buff_encoder(buffs)
        buff_emb = buff_emb.mean(dim=1)
        
        # Progress: [batch, PROGRESS_FEATURE_DIM] -> [batch, EMBED_DIM]
        progress_emb = F.relu(self.progress_encoder(progress))
        
        return hero_emb, monster_emb, treasure_emb, buff_emb, progress_emb
        
    def fuse_entities(self, hero_emb, monster_emb, treasure_emb, buff_emb, progress_emb):
        """融合实体特征，使用注意力机制."""
        # 拼接所有实体特征
        all_entities = torch.cat([
            hero_emb,
            monster_emb,
            treasure_emb,
            buff_emb,
            progress_emb,
        ], dim=-1)  # [batch, EMBED_DIM * 5]
        
        # 计算注意力权重
        attention_weights = F.softmax(self.fusion_attention(all_entities), dim=-1)  # [batch, 5]
        
        # 加权融合
        entity_fused = (
            attention_weights[:, 0:1] * hero_emb +
            attention_weights[:, 1:2] * monster_emb +
            attention_weights[:, 2:3] * treasure_emb +
            attention_weights[:, 3:4] * buff_emb +
            attention_weights[:, 4:5] * progress_emb
        )  # [batch, EMBED_DIM]
        
        return entity_fused
        
    def forward(self, hero, monsters, treasures, buffs, progress, map_grid, legal_action):
        """
        前向传播.
        
        Args:
            hero: [batch, HERO_FEATURE_DIM]
            monsters: [batch, MAX_MONSTERS, MONSTER_FEATURE_DIM]
            treasures: [batch, MAX_TREASURES, TREASURE_FEATURE_DIM]
            buffs: [batch, MAX_BUFFS, BUFF_FEATURE_DIM]
            progress: [batch, PROGRESS_FEATURE_DIM]
            map_grid: [batch, MAP_CHANNELS, MAP_SIZE, MAP_SIZE]
            legal_action: [batch, ACTION_SHAPE[0]]
            
        Returns:
            logits: [batch, ACTION_SHAPE[0]]
            value: [batch, VALUE_SHAPE[0]]
        """
        # 编码实体
        hero_emb, monster_emb, treasure_emb, buff_emb, progress_emb = self.encode_entities(
            hero, monsters, treasures, buffs, progress
        )
        
        # 融合实体
        entity_fused = self.fuse_entities(
            hero_emb, monster_emb, treasure_emb, buff_emb, progress_emb
        )
        
        # 处理地图
        map_features = self.map_cnn(map_grid)  # [batch, MAP_CNN_HIDDEN]
        
        # 融合所有信息
        combined = torch.cat([entity_fused, map_features, legal_action], dim=-1)
        fused = self.fusion(combined)
        
        # 输出
        logits = self.actor_head(fused)
        value = self.critic_head(fused)
        
        return logits, value


class Model(nn.Module):
    """Wrapper model for compatibility."""
    
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy_structured"
        self.device = device
        self.network = StructuredNetwork()
        
    def forward(self, obs, legal_action=None, inference=False):
        """
        兼容旧接口，接收结构化输入或展平向量.
        
        Args:
            obs: dict with keys ['hero', 'monsters', 'treasures', 'buffs', 
                                 'progress', 'map', 'legal_action']
                 or tensor [batch, feature_dim] 展平向量
            legal_action: [batch, ACTION_SHAPE[0]] 合法动作掩码，训练时传入
        """
        if isinstance(obs, dict):
            return self.network(
                hero=obs['hero'],
                monsters=obs['monsters'],
                treasures=obs['treasures'],
                buffs=obs['buffs'],
                progress=obs['progress'],
                map_grid=obs['map'],
                legal_action=obs['legal_action']
            )
        else:
            # 兼容旧格式（展平向量）- 用于训练时从SampleData读取
            feature_dict = self._unflatten_obs(obs, legal_action)
            return self.network(
                hero=feature_dict['hero'],
                monsters=feature_dict['monsters'],
                treasures=feature_dict['treasures'],
                buffs=feature_dict['buffs'],
                progress=feature_dict['progress'],
                map_grid=feature_dict['map'],
                legal_action=feature_dict['legal_action']
            )
    
    def _unflatten_obs(self, obs_tensor, legal_action=None):
        """将展平的特征向量重构为字典格式.
        
        Args:
            obs_tensor: [batch, feature_dim] 展平的特征向量
            legal_action: [batch, ACTION_SHAPE[0]] 合法动作掩码，训练时传入
            
        Returns:
            dict: 重构后的结构化特征
        """
        batch_size = obs_tensor.shape[0]
        
        # 计算各特征的维度
        hero_dim = Config.HERO_FEATURE_DIM
        monster_dim = Config.MONSTER_FEATURE_DIM * Config.MAX_MONSTERS
        treasure_dim = Config.TREASURE_FEATURE_DIM * Config.MAX_TREASURES
        buff_dim = Config.BUFF_FEATURE_DIM * Config.MAX_BUFFS
        progress_dim = Config.PROGRESS_FEATURE_DIM
        map_dim = Config.MAP_CHANNELS * Config.MAP_SIZE * Config.MAP_SIZE
        
        # 按顺序切分特征
        idx = 0
        hero = obs_tensor[:, idx:idx+hero_dim]
        idx += hero_dim
        
        monsters = obs_tensor[:, idx:idx+monster_dim].view(batch_size, Config.MAX_MONSTERS, Config.MONSTER_FEATURE_DIM)
        idx += monster_dim
        
        treasures = obs_tensor[:, idx:idx+treasure_dim].view(batch_size, Config.MAX_TREASURES, Config.TREASURE_FEATURE_DIM)
        idx += treasure_dim
        
        buffs = obs_tensor[:, idx:idx+buff_dim].view(batch_size, Config.MAX_BUFFS, Config.BUFF_FEATURE_DIM)
        idx += buff_dim
        
        progress = obs_tensor[:, idx:idx+progress_dim]
        idx += progress_dim
        
        map_grid = obs_tensor[:, idx:idx+map_dim].view(batch_size, Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)
        idx += map_dim
        
        # 使用传入的 legal_action，如果没有则创建全1占位符
        if legal_action is None:
            legal_action = torch.ones(batch_size, Config.ACTION_SHAPE[0], device=obs_tensor.device)
        
        return {
            'hero': hero,
            'monsters': monsters,
            'treasures': treasures,
            'buffs': buffs,
            'progress': progress,
            'map': map_grid,
            'legal_action': legal_action,
        }
            
    def set_train_mode(self):
        self.train()
        
    def set_eval_mode(self):
        self.eval()
