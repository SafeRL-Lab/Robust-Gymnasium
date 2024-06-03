import numpy as np

from robust_gymnasium import utils
from robust_gymnasium.envs.mujoco.mujoco_py_env import MuJocoPyEnv
from robust_gymnasium.spaces import Box

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

class SwimmerEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "swimmer.xml", 4, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, a):
        mu = args.noise_mu
        sigma = args.noise_sigma
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs() + random.gauss(mu, sigma)  # robust setting

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl),
        )

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()
