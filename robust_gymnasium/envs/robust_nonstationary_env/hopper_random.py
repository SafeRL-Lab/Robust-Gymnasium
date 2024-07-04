import mujoco_py
import gymnasium as gym
import numpy as np
import random

from gym.envs.mujoco.hopper_v3 import HopperEnv

class HopperTransferEnv(HopperEnv):
    def __init__(self, torso_len: float = 0.2, foot_len: float = 0.195, **kwargs):
        super().__init__(**kwargs)
        self.model.body_pos[1][2] = 1.05 + torso_len
        self.model.body_pos[2][2] = -torso_len
        self.model.geom_size[1][1] = torso_len

        self.model.geom_size[4][1] = foot_len



class HopperRandomEnv(HopperEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, torso_len: float = 0.2, foot_len: float = 0.195):
        '''Must called like env.reset(body_len = XXX)'''
        
        self.model.body_pos[1][2] = 1.05 + torso_len
        self.model.body_pos[2][2] = -torso_len
        self.model.geom_size[1][1] = torso_len

        self.model.geom_size[4][1] = foot_len

        return super().reset()

if __name__ == "__main__":
    env = gym.make("HopperRandom-v0", torso_len = 0.2, foot_len = 0.195)
    env.reset(torso_len = 0.2, foot_len = 0.195)
    
    for _ in range(10):
        for i in range(50):
            env.step(np.random.rand(3))
            env.render()

        l1 = 0.2 + 0.15 * random.random()
        l2 = 0.195 + 0.1 * random.random()
        env.reset(torso_len = l1, foot_len = l2)
    
    env.close()