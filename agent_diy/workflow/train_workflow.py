#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase DIY Agent.
峡谷追猎 DIY 智能体训练工作流。

课程学习配置：
- warmup_stable (0-150): 资源多、压力低，学会稳定推进
- mid_pressure (151-500): 逐步增加难度
- late_speedup_survival (501-900): 高压存活
- hard_generalization (901+): 泛化挑战
"""

import os
import time

import numpy as np
from agent_diy.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


CURRICULUM_PHASES = [
    {
        "name": "warmup_stable",
        "max_episode": 150,
        "treasure_count": (9, 10),
        "buff_count": (2, 2),
        "monster_interval": (220, 300),
        "monster_speedup": (360, 460),
        "max_step": 2000,
    },
    {
        "name": "mid_pressure",
        "max_episode": 500,
        "treasure_count": (8, 10),
        "buff_count": (1, 2),
        "monster_interval": (160, 280),
        "monster_speedup": (240, 420),
        "max_step": 2000,
    },
    {
        "name": "late_speedup_survival",
        "max_episode": 900,
        "treasure_count": (7, 10),
        "buff_count": (1, 2),
        "monster_interval": (120, 220),
        "monster_speedup": (180, 320),
        "max_step": 2000,
    },
    {
        "name": "hard_generalization",
        "max_episode": float('inf'),
        "treasure_count": (6, 10),
        "buff_count": (0, 2),
        "monster_interval": (120, 320),
        "monster_speedup": (140, 420),
        "max_step": 2000,
    },
]


def _get_curriculum_phase(episode_cnt):
    """Get current curriculum phase based on episode count."""
    for phase in CURRICULUM_PHASES:
        if episode_cnt <= phase["max_episode"]:
            return phase
    return CURRICULUM_PHASES[-1]


def _sample_range(range_tuple):
    """Sample a random integer from a (min, max) tuple."""
    return np.random.randint(range_tuple[0], range_tuple[1] + 1)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    base_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if base_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        base_conf=base_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, base_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.base_conf = base_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.current_phase_index = 0

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            self.episode_cnt += 1
            phase = _get_curriculum_phase(self.episode_cnt)
            self.current_phase_index = CURRICULUM_PHASES.index(phase)
            self.logger.info(f"Episode {self.episode_cnt} - Phase: {phase['name']}")

            conf = self._build_curriculum_conf(phase)
            env_obs = self.env.reset(conf)

            if handle_disaster_recovery(env_obs, self.logger):
                self.episode_cnt -= 1
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 10.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "phase": self.current_phase_index,
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info

    def _build_curriculum_conf(self, phase):
        """Build environment config for current curriculum phase."""
        conf = self.base_conf.copy() if hasattr(self.base_conf, 'copy') else dict(self.base_conf)

        treasure_count = _sample_range(phase["treasure_count"])
        buff_count = _sample_range(phase["buff_count"])
        monster_interval = _sample_range(phase["monster_interval"])
        monster_speedup = _sample_range(phase["monster_speedup"])
        max_step = phase["max_step"]

        if hasattr(conf, 'update'):
            conf.update({
                "treasure_count": treasure_count,
                "buff_count": buff_count,
                "monster_interval": monster_interval,
                "monster_speedup": monster_speedup,
                "max_step": max_step,
            })
        else:
            conf["treasure_count"] = treasure_count
            conf["buff_count"] = buff_count
            conf["monster_interval"] = monster_interval
            conf["monster_speedup"] = monster_speedup
            conf["max_step"] = max_step

        return conf