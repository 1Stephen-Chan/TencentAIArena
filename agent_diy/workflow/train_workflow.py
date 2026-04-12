#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase DIY agent.
"""

import os
import time
from copy import deepcopy

import numpy as np

from agent_diy.feature.definition import SampleData, reward_shaping, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
import random

# 默认模型保存时间
DEFAULT_PERIODIC_SAVE_SECS = 1800
# 最优模型预热保存时间
DEFAULT_BEST_SAVE_WARMUP_SECS = 1800
DEFAULT_LATEST_MODEL_ID = "latest"
DEFAULT_BEST_MODEL_ID = "best"


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )
    best_model_selector = BestModelSelector(logger=logger)

    training_start_time = time.time()
    next_periodic_save_time = training_start_time + DEFAULT_PERIODIC_SAVE_SECS
    if logger:
        logger.info(
            f"best model saving will start after {DEFAULT_BEST_SAVE_WARMUP_SECS} seconds, "
            f"periodic save interval is {DEFAULT_PERIODIC_SAVE_SECS} seconds"
        )

    while True:
        for episode_result in episode_runner.run_episodes():
            g_data = episode_result["sample_data"]
            agent.send_sample_data(g_data)
            g_data.clear()

            episode_summary = episode_result["episode_summary"]
            now = time.time()

            if should_start_best_save(now, training_start_time):
                if best_model_selector.update_if_best(episode_summary):
                    agent.save_model(id=DEFAULT_BEST_MODEL_ID)
                    # 更新下次定期保存时间，避免短时间内重复保存
                    next_periodic_save_time = now + 60

            if now >= next_periodic_save_time:
                agent.save_model()
                next_periodic_save_time = get_next_periodic_save_time(
                    current_time=now,
                    previous_target_time=next_periodic_save_time,
                    interval_secs=DEFAULT_PERIODIC_SAVE_SECS,
                )


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor

        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # 课程学习
            episode_usr_conf = self._build_curriculum_conf()
            env_obs = self.env.reset(episode_usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id=DEFAULT_LATEST_MODEL_ID)

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, next_env_obs = self.env.step(act)
                if handle_disaster_recovery(next_env_obs, self.logger):
                    break

                terminated = bool(next_env_obs["terminated"])
                truncated = bool(next_env_obs["truncated"])
                done = terminated or truncated
                step += 1

                next_obs_data, next_remain_info = self.agent.observation_process(next_env_obs)


                # 奖励塑性
                reward = reward_shaping(
                    frame_no=next_env_obs.get("frame_no", step),
                    score=env_reward.get("reward", 0.0) if isinstance(env_reward, dict) else 0.0,
                    terminated=terminated,
                    truncated=truncated,
                    remain_info=remain_info,
                    _remain_info=next_remain_info,
                    obs=env_obs,
                    _obs=next_env_obs,
                )
                total_reward += float(reward[0])

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=np.array(reward, dtype=np.float32),
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    env_info = next_env_obs.get("observation", {}).get("env_info", {})
                    total_score = env_info.get("total_score", 0)
                    result_str = "FAIL" if terminated else "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                    if collector:
                        collector = sample_process(collector)

                        now = time.time()
                        if now - self.last_report_monitor_time >= 60 and self.monitor:
                            monitor_data = {
                                "reward": round(total_reward, 4),
                                "episode_steps": step,
                                "episode_cnt": self.episode_cnt,
                                "sim_score": float(total_score),
                            }
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                        yield {
                            "sample_data": collector,
                            "episode_summary": {
                                "episode_cnt": self.episode_cnt,
                                "episode_steps": step,
                                "episode_reward": float(total_reward),
                                "total_score": float(total_score),
                                "success": not terminated,
                            },
                        }
                    break

                obs_data = next_obs_data
                remain_info = next_remain_info
                env_obs = next_env_obs

    def _build_curriculum_conf(self):
        """
        课程学习 curriculum learning
        """
        conf = deepcopy(self.usr_conf)
        env_conf = conf.get("env_conf", {})
        if not isinstance(env_conf, dict):
            return conf

        ep = int(self.episode_cnt)

        if ep < 150:
            # warmup_stable: 简单地图，稳定环境，让智能体学习基础操作
            env_conf["map"] = [1, 3, 4, 5]
            env_conf["map_random"] = True
            env_conf["treasure_count"] = 2
            env_conf["buff_count"] = 2
            env_conf["monster_interval"] = random.randint(220, 300)
            env_conf["monster_speedup"] = random.randint(360, 460)
            env_conf["max_step"] = 2000

        elif ep < 500:
            # mid_pressure: 增加压力，怪物出现更快
            env_conf["map"] = [1, 3, 4, 5]
            env_conf["map_random"] = True
            env_conf["treasure_count"] = random.randint(8, 10)
            env_conf["buff_count"] = 2
            env_conf["monster_interval"] = random.randint(160, 280)
            env_conf["monster_speedup"] = random.randint(240, 420)
            env_conf["max_step"] = 2000

        elif ep < 900:
            # late_speedup_survival: 更快加速，考验生存能力
            env_conf["map"] = [1, 3, 4, 5, 6, 8, 9]
            env_conf["map_random"] = True
            env_conf["treasure_count"] = random.randint(7, 10)
            env_conf["buff_count"] = 2
            env_conf["monster_interval"] = random.randint(120, 220)
            env_conf["monster_speedup"] = random.randint(180, 320)
            env_conf["max_step"] = 2000

        else:
            # hard_generalization: 
            env_conf["map"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            env_conf["map_random"] = True
            env_conf["treasure_count"] = random.randint(6, 10)
            env_conf["buff_count"] = 2
            env_conf["monster_interval"] = random.randint(120, 320)
            env_conf["monster_speedup"] = random.randint(140, 420)
            env_conf["max_step"] = 2000

        conf["env_conf"] = env_conf
        return conf


class BestModelSelector:
    def __init__(self, logger=None):
        self.logger = logger
        self.best_priority = None

    def update_if_best(self, episode_summary):
        priority = self._build_priority(episode_summary)
        if self.best_priority is not None and priority <= self.best_priority:
            return False

        self.best_priority = priority
        if self.logger:
            self.logger.info(
                f"new best model selected, episode:{episode_summary['episode_cnt']} "
                f"reward:{episode_summary['episode_reward']:.3f} "
                f"score:{episode_summary['total_score']:.1f} "
                f"success:{episode_summary['success']}"
            )
        return True

    def _build_priority(self, episode_summary):
        return (
            int(bool(episode_summary.get("success", False))),
            float(episode_summary.get("episode_reward", float("-inf"))),
            float(episode_summary.get("total_score", float("-inf"))),
            -int(episode_summary.get("episode_steps", 0)),
        )


def should_start_best_save(current_time, training_start_time):
    return (current_time - training_start_time) >= DEFAULT_BEST_SAVE_WARMUP_SECS


def get_next_periodic_save_time(current_time, previous_target_time, interval_secs):
    next_target_time = previous_target_time
    while next_target_time <= current_time:
        next_target_time += interval_secs
    return next_target_time
