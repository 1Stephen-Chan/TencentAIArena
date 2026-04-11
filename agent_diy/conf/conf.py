#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

DIY Agent Configuration with Structured Features.
结构化特征配置的DIY智能体。
"""


import numpy as np


class Config:

    # Structured feature dimensions / 结构化特征维度
    # 标量特征 - 交给MLP
    HERO_FEATURE_DIM = 10          # 英雄状态: 位置(x,z), 分数, 步数, 闪现CD, buff时间等
    MONSTER_FEATURE_DIM = 8        # 单只怪物: 距离, 方向, 速度, 威胁度等
    TREASURE_FEATURE_DIM = 4       # 单个宝箱: 距离, 方向, 是否存在, 价值
    BUFF_FEATURE_DIM = 4           # 单个buff: 距离, 方向, 是否存在, 类型
    PROGRESS_FEATURE_DIM = 6       # 进度特征: 阶段, 危险等级, 卡住计数等
    
    # 实体数量 / Entity counts
    MAX_MONSTERS = 2               # 最多2只怪物
    MAX_TREASURES = 4              # 最多4个宝箱
    MAX_BUFFS = 2                  # 最多2个buff
    
    # 地图特征 - 交给CNN / Map features for CNN
    MAP_SIZE = 21                  # 21x21局部地图（视野范围）
    MAP_CHANNELS = 1               # 单通道（可行走=1，障碍=0）
    
    # 总标量特征维度 / Total scalar features
    SCALAR_FEATURE_DIM = (
        HERO_FEATURE_DIM +
        MONSTER_FEATURE_DIM * MAX_MONSTERS +
        TREASURE_FEATURE_DIM * MAX_TREASURES +
        BUFF_FEATURE_DIM * MAX_BUFFS +
        PROGRESS_FEATURE_DIM
    )  # = 10 + 16 + 16 + 8 + 6 = 56

    # 地图特征维度 / Map feature dimensions
    MAP_FEATURE_DIM = MAP_CHANNELS * MAP_SIZE * MAP_SIZE  # 1 * 21 * 21 = 441
    
    # 总特征向量形状 - 用于SampleData / Total feature vector shape for SampleData
    # 标量特征(56) + 地图特征(441) = 497
    FEATURE_VECTOR_SHAPE = (SCALAR_FEATURE_DIM + MAP_FEATURE_DIM,)

    # 动作空间 / Action space
    ACTION_SHAPE = (16,)           # 8移动 + 8闪现
    VALUE_SHAPE = (1,)
    
    # 网络结构 / Network architecture
    EMBED_DIM = 32                 # 实体编码维度
    MAP_CNN_HIDDEN = 64            # CNN输出维度
    FUSION_HIDDEN = 128            # 融合层隐藏维度
    
    # PPO超参数 / PPO hyperparameters
    GAMMA = 0.95
    LAMDA = 0.95
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    ENTROPY_COEFF = 0.001
    MAX_GRAD_NORM = 0.5
    
    # 学习率 / Learning rate
    START_LR = 1e-4                # 降低学习率以适应复杂网络
    END_LR = 1e-5
    DECAY_TYPE = "cosine"          # 余弦衰减更平滑
    
    # 训练参数 / Training parameters
    VALUE_LOSS_COEFF = 0.5
    ENTROPY_LOSS_COEFF = 0.001
    EPSILON = 1e-8
    MAX_BUFFER_SIZE = 10000
    BATCH_SIZE = 512               # 增加批次大小
    EPOCH = 5                      # 减少epoch避免过拟合

