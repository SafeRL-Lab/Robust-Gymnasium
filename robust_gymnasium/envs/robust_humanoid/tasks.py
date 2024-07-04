import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards
import mujoco

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

class Task:
    qpos0_robot = {}
    dof = 0
    frame_skip = 10
    camera_name = "cam_default"
    max_episode_steps = 1000
    kwargs = {}  # Default kwargs for a task

    def __init__(self, robot=None, env=None, **kwargs):
        self.robot = robot
        if env:
            self._env = env

    @property
    def observation_space(self):
        return None

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        state = np.concatenate((position, velocity))
        return state

    def get_reward(self):
        return 0, {}

    def get_terminated(self):
        return False, {}

    def reset_model(self):
        return self.get_obs()

    def normalize_action(self, action):
        return (
            2
            * (action - self._env.action_low)
            / (self._env.action_high - self._env.action_low)
            - 1
        )

    def unnormalize_action(self, action):
        return (action + 1) / 2 * (
            self._env.action_high - self._env.action_low
        ) + self._env.action_low

    def step(self, robust_input):
        action = robust_input["action"]
        args = robust_input["robust_config"]
        mu = args.noise_mu
        sigma = args.noise_sigma

        if args.noise_factor == "action":
            if args.noise_type == "gauss":
                action = action + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                action = action + args.noise_shift
            else:
                action = action
                print('\033[0;31m "No action entropy learning! Using the original action" \033[0m')
        else:
            action = action

        action = self.unnormalize_action(action)
        self._env.do_simulation(action, self._env.frame_skip)

        obs = self.get_obs()
        reward, reward_info = self.get_reward()
        terminated, terminated_info = self.get_terminated()
        # print("obs---------:", obs)
        if args.noise_factor == "state":
            if args.noise_type == "gauss":
                obs = obs + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                obs = obs + args.noise_shift
            else:
                obs = obs
                print('\033[0;31m "No state entropy learning! Using the original state" \033[0m')
        else:
            obs = obs

        info = {"per_timestep_reward": reward, **reward_info, **terminated_info}

        if args.noise_factor == "reward":
            if args.noise_type == "gauss":
                reward = reward + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                reward = reward + args.noise_shift
            else:
                reward = reward
                print('\033[0;31m "No reward entropy learning! Using the original reward" \033[0m')
        else:
            reward = reward

        return obs, reward, terminated, False, info

    def render(self):
        # , self._env.camera_id, self._env.camera_name
        return self._env.mujoco_renderer.render(
            self._env.render_mode, self._env.camera_id, self._env.camera_name
        )
