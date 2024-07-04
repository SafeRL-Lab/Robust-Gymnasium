"""A point mass environment with Gymnasium API for the Gymnasium-Robotics PointMaze environments.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

from os import path
from typing import Optional

import numpy as np
from robust_gymnasium import spaces
from robust_gymnasium.envs.robust_mujoco.mujoco_env import MujocoEnv

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

class PointEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, xml_file: Optional[str] = None, **kwargs):

        if xml_file is None:
            xml_file = path.join(
                path.dirname(path.realpath(__file__)), "../assets/point/point.xml"
            )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        super().__init__(
            model_path=xml_file,
            frame_skip=1,
            observation_space=observation_space,
            **kwargs
        )

    def reset_model(self) -> np.ndarray:
        self.set_state(self.init_qpos, self.init_qvel)
        obs, _ = self._get_obs()

        return obs

    def step(self, action):
        if args.noise_factor == "action":
            if args.noise_type == "gauss":
                action = action + random.gauss(args.noise_mu, args.noise_sigma)  # robust setting
            elif args.noise_type == "shift":
                action = action + args.noise_shift
            else:
                action = action
                print('\033[0;31m "No action entropy learning! Using the original action" \033[0m')
        else:
            action = action
        action = np.clip(action, -1.0, 1.0)
        self._clip_velocity()
        self.do_simulation(action, self.frame_skip)
        obs, info = self._get_obs()
        # This environment class has no intrinsic task, thus episodes don't end and there is no reward
        reward = 0
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()

        if args.noise_factor == "state":
            if args.noise_type == "gauss":
                obs = obs + random.gauss(args.noise_mu, args.noise_sigma)  # robust setting
            elif args.noise_type == "shift":
                obs = obs + args.noise_shift
            else:
                obs = obs
                print('\033[0;31m "No action entropy learning! Using the original action" \033[0m')
        else:
            obs = obs

        if args.noise_factor == "reward":
            if args.noise_type == "gauss":
                reward = reward + random.gauss(args.noise_mu, args.noise_sigma)  # robust setting
            elif args.noise_type == "shift":
                reward = reward + args.noise_shift
            else:
                reward = reward
                print('\033[0;31m "No action entropy learning! Using the original action" \033[0m')
        else:
            reward = reward

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel(), {}

    def _clip_velocity(self):
        """The velocity needs to be limited because the ball is
        force actuated and the velocity can grow unbounded."""
        qvel = np.clip(self.data.qvel, -5.0, 5.0)
        self.set_state(self.data.qpos, qvel)
