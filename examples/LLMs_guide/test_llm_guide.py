import robust_gymnasium as gym
from os import path
import json
import os
import time
from datetime import datetime
currentDateAndTime = datetime.now()
start_run_date_and_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()
args.env_name = "Ant-v4"
# "InvertedDoublePendulum-v4"
# "Reacher-v4"
# "Pusher-v4"
# "HumanoidStandup-v4" #
# "Humanoid-v4"
# "Swimmer-v4"
# "Hopper-v4"
# "Walker2d-v4"
# "HalfCheetah-v4"
# "Ant-v4"
# "Pusher-v4"
# InvertedPendulum
args.noise_factor = "robust_force - llm"
# "robust_force"
# "robust_shape"
# "action"
args.noise_type = "gauss---llm"
# "uniform"
args.llm_guide = "adversary"

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


env = gym.make(args.env_name) # render_mode="human",  render environments: human, rgb_array, or depth_array.


def replace_xml_content(source_file_path, target_file_path):
    # read data from source file
    with open(source_file_path, 'r', encoding='utf-8') as file:
        source_content = file.read()

    # write the data into the target file
    with open(target_file_path, 'w', encoding='utf-8') as file:
        file.write(source_content)

observation, info = env.reset(seed=42)

try:
    for i in range(10000):
        action = env.action_space.sample()
        robust_input = {
            "action": action,
            "robust_type": "action",
            "robust_config": args,
        }

        observation, reward, terminated, truncated, info = env.step(robust_input)       
        if terminated or truncated:
            observation, info = env.reset()

        if i > 9999:
            replace_xml_content(info["source_file_path"], info["target_file_path"])
finally:  #except KeyboardInterrupt:
    replace_xml_content(info["source_file_path"], info["target_file_path"])
    print('\033[0;31m "Program was terminated by user (Ctrl+C) or finished!" \033[0m')

env.close()
