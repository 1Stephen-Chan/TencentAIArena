#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Best model tracker for window-average score.
"""


class BestModelTracker:
    """
    纯内存版最优窗口追踪器。

    规则：
    1. 每 check_interval 局形成一个窗口
    2. 只看窗口平均分
    3. 若该窗口平均分刷新全局最高，则返回窗口摘要，供 workflow 立即保存当前模型
    """

    def __init__(self, check_interval=100, logger=None, min_improvement=0.0):
        self.check_interval = check_interval
        self.logger = logger
        self.min_improvement = min_improvement

        self.recent_scores = []
        self.recent_episodes = []

        self.global_best_score = -999999.0
        self.best_window_info = None

    def record_episode(self, episode_num, score, total_reward):
        """
        记录一局结果；窗口满时检查是否刷新全局最高窗口平均分。

        Returns:
            dict | None:
                若当前窗口刷新全局最高，返回窗口摘要；
                否则返回 None。
        """
        self.recent_scores.append(float(score))
        self.recent_episodes.append(
            {
                "episode": int(episode_num),
                "score": float(score),
                "total_reward": float(total_reward),
            }
        )

        if len(self.recent_scores) >= self.check_interval:
            return self._check_and_update_best()

        return None

    def _check_and_update_best(self):
        """检查当前窗口平均分是否刷新全局最高。"""
        if not self.recent_scores:
            return None

        window_size = len(self.recent_scores)
        avg_score = sum(self.recent_scores) / window_size
        avg_total_reward = sum(ep["total_reward"] for ep in self.recent_episodes) / window_size
        max_score_episode = max(self.recent_episodes, key=lambda x: x["score"])

        window_info = {
            "criterion": "global_best_window_avg",
            "window_avg_score": float(avg_score),
            "best_score": float(avg_score),
            "window_start_episode": int(self.recent_episodes[0]["episode"]),
            "window_end_episode": int(self.recent_episodes[-1]["episode"]),
            "window_size": window_size,
            "avg_total_reward": float(avg_total_reward),
            "max_score": float(max_score_episode["score"]),
            "max_score_episode": int(max_score_episode["episode"]),
        }

        should_save = avg_score > self.global_best_score + self.min_improvement
        if should_save:
            self.global_best_score = avg_score
            self.best_window_info = dict(window_info)

            if self.logger:
                self.logger.info(
                    "[GLOBAL BEST WINDOW] "
                    f"window_avg:{avg_score:.3f} | "
                    f"episodes:[{window_info['window_start_episode']}, {window_info['window_end_episode']}] | "
                    f"window_max:{window_info['max_score']:.1f}"
                )

        self.recent_scores.clear()
        self.recent_episodes.clear()
        return dict(window_info) if should_save else None

    def get_best_score(self):
        """获取全局最高窗口平均分"""
        return self.global_best_score

    def get_global_best_score(self):
        """获取全局最高窗口平均分"""
        return self.global_best_score

    def get_best_window_info(self):
        """获取当前保存的最优窗口摘要"""
        return None if self.best_window_info is None else dict(self.best_window_info)
