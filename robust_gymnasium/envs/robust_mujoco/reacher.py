import numpy as np

from robust_gymnasium import utils
from robust_gymnasium.envs.robust_mujoco.mujoco_py_env import MuJocoPyEnv
from robust_gymnasium.spaces import Box

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

class ReacherEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "reacher.xml", 2, observation_space=observation_space, **kwargs
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
            a = a
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

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
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )
