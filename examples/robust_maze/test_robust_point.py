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
args.env_name = "PointMaze_UMaze-v3" # "InvertedDoublePendulum-v4" # "Reacher-v4" # "Pusher-v4" # "HumanoidStandup-v4" # "Humanoid-v4" # "Swimmer-v4" # "Hopper-v4" # "Walker2d-v4" # "HalfCheetah-v4" # "Ant-v4"
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
    """Check that PointMaze does not reset into a success state."""
    env = gym.make("PointMaze_UMaze-v3", continuing_task=True)

    for _ in range(1000):
        obs, info = env.reset()
        assert not info["success"]
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"


def test_reset_cell():
    """Check that passing the reset_cell location ensures that the agent resets in the right cell."""
    map = [
        [1, 1, 1, 1],
        [1, "r", "r", 1],
        [1, "r", "g", 1],
        [1, 1, 1, 1],
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=map, render_mode="human")
    obs = env.reset(options={"reset_cell": [1, 2]}, seed=42)[0]
    desired_obs = np.array([0.67929896, 0.59868401, 0, 0])
    np.testing.assert_almost_equal(desired_obs, obs["observation"], decimal=4)


def test_goal_cell():
    """Check that passing the goal_cell location ensures that the goal spawns in the right cell."""
    map = [
        [1, 1, 1, 1],
        [1, "r", "g", 1],
        [1, "g", "g", 1],
        [1, 1, 1, 1],
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=map, render_mode="human")
    obs = env.reset(options={"goal_cell": [2, 1]}, seed=42)[0]
    desired_goal = np.array([-0.36302198, -0.53056078])
    np.testing.assert_almost_equal(desired_goal, obs["desired_goal"], decimal=4)

def test_exp():
    map = [
        [1, 1, 1, 1],
        [1, "r", "g", 1],
        [1, "g", "g", 1],
        [1, 1, 1, 1],
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=map, render_mode="human")

    for _ in range(10000):
        obs = env.reset(options={"goal_cell": [2, 1]}, seed=42)[0]
        # obs, info = env.reset()
        desired_goal = np.array([-0.36302198, -0.53056078])
        np.testing.assert_almost_equal(desired_goal, obs["desired_goal"], decimal=4)



if __name__ == '__main__':
    test_exp()
    # test_goal_cell()
