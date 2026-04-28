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
from agent_diy.workflow.best_model_tracker import BestModelTracker
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


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

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

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

        model_path = "/data/user_ckpt_dir/gorge_chase_diy"
        self.best_model_path = model_path
        self.best_tracker = BestModelTracker(
            check_interval=50,  # 每50局检查一次
            min_improvement=20,  # 至少提升0.5分才算新最优
            logger=logger
        )

    def _save_best_window_model(self, best_info):
        """
        保存“全局窗口平均分最高”的模型。

        """
        if not best_info:
            return

        self.agent.save_model()

        self.logger.info(
                f"[BEST WINDOW MODEL SAVED] window_avg:{best_info.get('window_avg_score', 0.0):.3f} "
                f"| episodes:[{best_info.get('window_start_episode', 0)}, "
                f"{best_info.get('window_end_episode', 0)}] "
                f"| save current model"
            )

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            episode_usr_conf = self._build_episode_usr_conf()
            if hasattr(self.agent, "set_episode_context"):
                self.agent.set_episode_context(episode_usr_conf)
            env_obs = self.env.reset(episode_usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            
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

                if reward is None:
                    reward = np.array([0.0], dtype=np.float32)
                elif isinstance(reward, (list, tuple)):
                    reward = np.array(reward, dtype=np.float32).reshape(-1)[:1]
                elif isinstance(reward, np.ndarray):
                    reward = reward.astype(np.float32).reshape(-1)[:1]
                else:
                    reward = np.array([float(reward)], dtype=np.float32)

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

                    # 记录得分到best_tracker
                    best_window_info = self.best_tracker.record_episode(
                        episode_num=self.episode_cnt,
                        score=total_score,
                        total_reward=total_reward,
                    )

                    if best_window_info != None:
                        self.logger.warning(f"保存最优模型:{best_window_info}")
                        self._save_best_window_model(best_window_info)

                    if collector:
                        collector = sample_process(collector)

                        now = time.time()
                        if now - self.last_report_monitor_time >= 60 and self.monitor:
                            monitor_data = {
                                "reward": round(total_reward, 4),
                                "episode_steps": step,
                                "episode_cnt": self.episode_cnt,
                                "sim_score": float(total_score),
                                "best_score": self.best_tracker.get_global_best_score(),
                            }
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                        yield collector
                    break

                obs_data = next_obs_data
                remain_info = next_remain_info
                env_obs = next_env_obs

    def _build_episode_usr_conf(self):
        """Curriculum over episodes: easy -> medium -> full difficulty."""
        conf = deepcopy(self.usr_conf)
        env_conf = conf.get("env_conf", {})
        if not isinstance(env_conf, dict):
            return conf

        ep = int(self.episode_cnt)

        # Stage 1: 第二次天梯榜配置
        if ep < 3000:
            env_conf["map"] = [1, 3, 4, 5]
            env_conf["map_random"] = True
            env_conf["buff_cooldown"] = 100
            env_conf["talent_cooldown"] = 100
            env_conf["monster_interval"] = 500
            env_conf["monster_speedup"] = 700
            env_conf["max_step"] = 2000

        # Stage 2: 第三次天梯榜配置
        elif ep < 7000:
            env_conf["map"] = [1, 2, 3, 4, 5, 6]
            env_conf["map_random"] = True
            env_conf["talent_cooldown"] = 100
            env_conf["monster_interval"] = 300
            env_conf["monster_speedup"] = 500
            env_conf["max_step"] = 1500

        # Stage 3: 第四次天梯榜配置
        elif ep < 11000:
            env_conf["map"] = [1, 2, 3, 4, 5, 6, 7, 8]
            env_conf["map_random"] = True
            env_conf["talent_cooldown"] = 150
            env_conf["monster_interval"] = 200
            env_conf["monster_speedup"] = 400
            env_conf["max_step"] = 1000

        # Stage 4: 最终天梯榜配置
        else:
            env_conf["map_random"] = True

        conf["env_conf"] = env_conf
        return conf
