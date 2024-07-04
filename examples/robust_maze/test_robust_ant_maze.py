import robust_gymnasium as gym
import numpy as np
import json
import os
import time
from datetime import datetime
currentDateAndTime = datetime.now()
start_run_date_and_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()
args.env_name = "AntMaze_UMaze-v4" # "InvertedDoublePendulum-v4" # "Reacher-v4" # "Pusher-v4" # "HumanoidStandup-v4" # "Humanoid-v4" # "Swimmer-v4" # "Hopper-v4" # "Walker2d-v4" # "HalfCheetah-v4" # "Ant-v4"
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


def test_reset():
    """Check that AntMaze does not reset into a success state."""
    env = gym.make("AntMaze_UMaze-v4", continuing_task=True, render_mode="human")

    for _ in range(10000):
        obs, info = env.reset()
        assert not info["success"]
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        env.render()  # render environments
        assert dist > 0.45, f"dist={dist} < 0.45"

if __name__=='__main__':
    test_reset()