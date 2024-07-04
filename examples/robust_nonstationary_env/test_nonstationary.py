import numpy as np

import datetime
import itertools
from copy import copy

import shutil
# import Non_stationary_env
import robust_gymnasium as gym
# import gym
import numpy as np
import json
import os
import time
from datetime import datetime
currentDateAndTime = datetime.now()
start_run_date_and_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()
args.env_name = "HumanoidRandom-v5"
# 'AntTransfer-v5',
# 'AntRandom-v5',
# 'HumanoidTransfer-v5',
# 'HumanoidRandom-v5',
# 'Walker2dTransfer-v5',
# 'Walker2dRandom-v5',
# 'HopperTransfer-v5',
# 'HopperRandom-v5',

args.noise_factor = "dynamics"
args.noise_type = "Non_stationary"

folder = os.getcwd()[:0] + 'data/' + str(args.env_name)+'/'+ str(args.noise_type)+'/'+ str(
    start_run_date_and_time) + '/'
print("folder---:", folder)
# folder = os.getcwd()[:-4] + 'runs\\test\\'
if not os.path.exists(folder):
    os.makedirs(folder)
json_path = folder + '/config.json'
argsDict = args.__dict__
with open(json_path, 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')

# env = gym.make(args.env_name)
env = gym.make(args.env_name, render_mode="human") # render environments: human, rgb_array, or depth_array.
# env.seed(args.env_seed)
env.action_space.seed(args.env_seed)
# torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)




# next_state, reward, done, _ = env.step(action) # Step
observation, info = env.reset(seed=args.env_seed)
env.reset()
for i in range(10000):
    action = env.action_space.sample()
    robust_input = {
        "action": action,
        "robust_type": "action",
        "robust_config": args,
    }

    observation, reward, terminated, truncated, info = env.step(robust_input)
    # env.step_with_random(robust_input)
    # env.reset_robust()
    # print("observation-------:", observation)
    # print("reward----------:", reward)
    env.render()  # render environments
    # if terminated or truncated:
    #     if "Hopper" in args.env_name:
    #         torso_len = np.random.uniform(0.3, 0.5)
    #         foot_len = np.random.uniform(0.29, 0.49)
    #         state, info = env.reset()
    #     elif "Walker" in args.env_name:
    #         torso_len = np.random.uniform(0.1, 0.3)
    #         foot_len = np.random.uniform(0.05, 0.15)
    #         state, info = env.reset_robust(torso_len=torso_len, foot_len=foot_len)
    #     elif "Ant" in args.env_name:
    #         gravity = np.random.uniform(9.81, 19.82)
    #         wind = np.random.uniform(0.8, 1.2)
    #         state, info = env.reset_robust(gravity=gravity, wind=wind)
    #     elif "Humanoid" in args.env_name:
    #         gravity = np.random.uniform(9.81, 19.82)
    #         wind = np.random.uniform(0.5, 1.5)
    #         state, info = env.reset_robust(gravity=gravity, wind=wind)
    #     else:
    #         state = env.reset()
    #     # observation, info = env.reset()
