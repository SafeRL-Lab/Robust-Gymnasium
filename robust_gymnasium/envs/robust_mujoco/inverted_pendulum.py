import numpy as np

from robust_gymnasium import utils
from robust_gymnasium.envs.robust_mujoco.mujoco_py_env import MuJocoPyEnv
from robust_gymnasium.spaces import Box

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

class InvertedPendulumEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self,
            "inverted_pendulum.xml",
            2,
            observation_space=observation_space,
            **kwargs,
        )

    def step(self, robust_input):
        a = robust_input["action"]
        # action = robust_input["action"]
        args = robust_input["robust_config"]
        mu = args.noise_mu
        sigma = args.noise_sigma
        if args.noise_factor == "action":
            if args.noise_type == "gauss":
                a = a + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                a = a + args.noise_shift
            else:
                a = a
                print('\033[0;31m "No action robust learning! Using the original action" \033[0m')
        else:
            a = action
        reward = 1.0
        self.do_simulation(a, self.frame_skip)

        if args.noise_factor == "state":
            if args.noise_type == "gauss":
                ob = self._get_obs() + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                ob = self._get_obs() + args.noise_shift
            else:
                ob = self._get_obs()
                print('\033[0;31m "No state robust learning! Using the original state" \033[0m')
        else:
            ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2))

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
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent
