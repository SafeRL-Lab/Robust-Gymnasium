import robust_gymnasium as gym

import json
import os
import time
from datetime import datetime
currentDateAndTime = datetime.now()
start_run_date_and_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()
args.env_name = "SafetyReacher-v4" # "SafetyAnt-v4" # "SafetyReacher-v4" # "Pusher-v4" # "HumanoidStandup-v4" # "Humanoid-v4" # "Swimmer-v4" # "Hopper-v4" # "Walker2d-v4" # "HalfCheetah-v4" # "Ant-v4"
args.noise_factor = "state"

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

# env = gym.make("Ant-v4")
env = gym.make(args.env_name, render_mode="human") # render environments
# env = gym.make(args.env_name, args)
print("type-----------args:", args)

observation, info = env.reset(seed=42)
for i in range(10000):
    action = env.action_space.sample()
    robust_input = {
        "action": action,
        "robust_type": "state",
        "robust_config": args,
    }
    observation, reward, terminated, truncated, info = env.step(robust_input)
    # print("reward----------:", info["reward"])
    # print("cost----------:", info["cost"])
    env.render() # render environments
    if terminated or truncated:
        observation, info = env.reset()
env.close()
