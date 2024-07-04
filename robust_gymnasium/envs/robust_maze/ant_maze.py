import sys
from os import path
from typing import Dict, List, Optional, Union

import numpy as np
from robust_gymnasium import spaces
from robust_gymnasium.envs.robust_mujoco.ant_v4 import AntEnv
from robust_gymnasium.utils.ezpickle import EzPickle

from robust_gymnasium.envs.robust_maze.maps import U_MAZE
from robust_gymnasium.envs.robust_maze.maze import MazeEnv
from robust_gymnasium.envs.utils.mujoco_utils import MujocoModelNames

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

class AntMazeEnv(MazeEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        **kwargs,
    ):
        # Get the ant.xml path from the Gymnasium package
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )
        super().__init__(
            agent_xml_path=ant_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=4,
            maze_height=0.5,
            reward_type=reward_type,
            continuing_task=continuing_task,
            **kwargs,
        )
        # Create the MuJoCo environment, include position observation of the Ant for GoalEnv
        self.ant_env = AntEnv(
            xml_file=self.tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            reset_noise_scale=0.0,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.ant_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.ant_env.action_space
        obs_shape: tuple = self.ant_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape[0] - 2,), dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            render_mode,
            maze_map,
            reward_type,
            continuing_task,
            **kwargs,
        )

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.ant_env.init_qpos[:2] = self.reset_pos

        obs, info = self.ant_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)

        return obs_dict, info

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
        ant_obs, _, _, _, info = self.ant_env.step(action)
        obs = self._get_obs(ant_obs)

        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

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

    def _get_obs(self, ant_obs: np.ndarray) -> Dict[str, np.ndarray]:
        achieved_goal = ant_obs[:2]
        observation = ant_obs[2:]

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def update_target_site_pos(self):
        self.ant_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def render(self):
        return self.ant_env.render()

    def close(self):
        super().close()
        self.ant_env.close()

    @property
    def model(self):
        return self.ant_env.model

    @property
    def data(self):
        return self.ant_env.data
