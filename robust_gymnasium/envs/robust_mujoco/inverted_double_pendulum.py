import numpy as np

from robust_gymnasium import utils
from robust_gymnasium.envs.robust_mujoco.mujoco_py_env import MuJocoPyEnv
from robust_gymnasium.spaces import Box

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

class InvertedDoublePendulumEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self,
            "inverted_double_pendulum.xml",
            5,
            observation_space=observation_space,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

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
        self.do_simulation(action, self.frame_skip)

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
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        terminated = bool(y <= 1)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        if args.noise_factor == "reward":
            if args.noise_type == "gauss":
                r = r + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                r = r + args.noise_shift
            else:
                r = r
                print('\033[0;31m "No reward robust learning! Using the original reward" \033[0m')
        else:
            r = r
        return ob, r, terminated, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos[:1],  # cart x pos
                np.sin(self.sim.data.qpos[1:]),  # link angles
                np.cos(self.sim.data.qpos[1:]),
                np.clip(self.sim.data.qvel, -10, 10),
                np.clip(self.sim.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
