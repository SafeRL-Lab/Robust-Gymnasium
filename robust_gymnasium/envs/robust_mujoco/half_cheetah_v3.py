__credits__ = ["Rushiv Arora"]

import numpy as np

from robust_gymnasium import utils
from robust_gymnasium.envs.robust_mujoco.mujoco_py_env import MuJocoPyEnv
from robust_gymnasium.spaces import Box

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )

        MuJocoPyEnv.__init__(
            self, xml_file, 5, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, robust_input):
        # action = action["action"]
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
                print('\033[0;31m "No action robust learning! Using the original action" \033[0m')
        else:
            action = action
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        if args.noise_factor == "state":
            if args.noise_type == "gauss":
                observation = self._get_obs() + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                observation = self._get_obs() + args.noise_shift
            else:
                observation = self._get_obs()
                print('\033[0;31m "No state robust learning! Using the original state" \033[0m')
        else:
            observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        if args.noise_factor == "reward":
            if args.noise_type == "gauss":
                reward = reward + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                reward = reward + args.noise_shift
            else:
                reward = reward
                print('\033[0;31m "No reward robust learning! Using the original reward" \033[0m')
        else:
            reward = reward
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
