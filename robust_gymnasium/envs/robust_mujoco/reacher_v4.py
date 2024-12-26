import numpy as np

from robust_gymnasium import utils
from robust_gymnasium.envs.robust_mujoco import MujocoEnv
from robust_gymnasium.spaces import Box

import xml.etree.ElementTree as ET
from os import path

import random
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

from robust_gymnasium.envs.llm_guide_robust.gpt_collect import gpt_call

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ReacherEnv(MujocoEnv, utils.EzPickle):
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
        MujocoEnv.__init__(
            self,
            "reacher.xml",
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.xml_file = "reacher.xml"
        self.xml_file_original = "reacher_original.xml"
        self.previous_reward = 0
        self.llm_disturb_iteration = 0

    def step(self, robust_input):
        a = robust_input["action"]
        # action = robust_input["action"]
        args = robust_input["robust_config"]
        mu = args.noise_mu
        sigma = args.noise_sigma

        if args.noise_factor == "robust_force":
            self.modify_xml(self.fullpath, args)
        if args.noise_factor == "robust_shape":
            self.modify_xml(self.fullpath, args)

        if args.noise_factor == "action":
            if args.noise_type == "gauss":
                a = a + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                a = a + args.noise_shift
            else:
                a = a
                print('\033[0;31m "No action entropy learning! Using the original action" \033[0m')
        else:
            a = a
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        # ob = self._get_obs() + random.gauss(mu, sigma)  # robust setting
        if args.noise_factor == "state":
            if args.noise_type == "gauss":
                observation = self._get_obs() + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                observation = self._get_obs() + args.noise_shift
            else:
                observation = self._get_obs()
                print('\033[0;31m "No state entropy learning! Using the original state" \033[0m')
        else:
            observation = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        if args.noise_factor == "reward":
            if args.noise_type == "gauss":
                reward = reward + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                reward = reward + args.noise_shift
            else:
                reward = reward
                print('\033[0;31m "No reward entropy learning! Using the original reward" \033[0m')
        else:
            reward = reward
        fullpath_original = self.expand_model_path(self.xml_file_original)
        info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "source_file_path": fullpath_original,
            "target_file_path": self.fullpath,
        }
        if args.noise_factor == "robust_force" or args.noise_factor == "robust_shape":
            self.replace_xml_content(fullpath_original, self.fullpath)
        
        if args.llm_guide == "adversary":
            self.llm_disturb_iteration += 1
            if args.llm_guide_type == "stochastic":
                if self.llm_disturb_iteration % args.llm_disturb_interval == 0:
                    prompt = "This is about a robust reinforcement learning setting; we want you as an adversary policy. If the current reward exceeds the previous reward value, please input some observation noises to disturb the environment and improve the learning algorithm's robustness. " \
                         "the current reward:" + str(reward) + ", the previous reward is" + str(self.previous_reward) \
                         + "please slightly revise the current environment state values:" + str(
                    observation) + ", just output the revised state with its original format" \
                                   "do not output any other things."
                    prompt_state = gpt_call(prompt)
                    observation = prompt_state
            elif args.llm_guide_type == "uniform":
                if self.llm_disturb_iteration % args.llm_disturb_interval == 0:
                    prompt = "This is about a robust reinforcement learning setting; we want you as an adversary policy. If the current reward exceeds the previous reward value, please input some observation noises to disturb the environment and improve the learning algorithm's robustness. " \
                         "The noises should subject the uniform distribution:" +str((args.uniform_low, args.uniform_high))+ ", the current reward:" + str(reward) + ", the previous reward is" + str(self.previous_reward) \
                         + "please slightly revise the current environment state values:" + str(
                    observation) + ", just output the revised state with its original format" \
                                   "do not output any other things."
                    prompt_state = gpt_call(prompt)
                    observation = prompt_state
            elif args.llm_guide_type == "region_constraint":
                if self.llm_disturb_iteration % args.llm_disturb_interval == 0:
                    prompt = "This is about a robust reinforcement learning setting; we want you as an adversary policy. If the current reward exceeds the previous reward value, please input some observation noises to disturb the environment and improve the learning algorithm's robustness. " \
                         "The noises should be in this area:" +str((args.region_low, args.region_high))+ ", the current reward:" + str(reward) + ", the previous reward is" + str(self.previous_reward) \
                         + "please slightly revise the current environment state values:" + str(
                    observation) + ", just output the revised state with its original format" \
                                   "do not output any other things."
                    prompt_state = gpt_call(prompt)
                    observation = prompt_state

            self.previous_reward = reward
            
        return (
            observation,
            reward,
            False,
            False,
            info,
        )

    def expand_model_path(self, model_path):
        """Expands the `model_path` to a full path if it starts with '~' or '.' or '/'."""
        if model_path.startswith(".") or model_path.startswith("/"):
            fullpath = model_path
        elif model_path.startswith("~"):
            fullpath = path.expanduser(model_path)
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        return fullpath

    def modify_xml(self, file_name, args):
        tree = ET.parse(file_name)
        root = tree.getroot()
        if args.noise_factor == "robust_force":
            hip_4_motor = root.find('.//motor[@joint="joint0"]')
            if hip_4_motor is not None:
                gear_value = hip_4_motor.get('gear')
                gear_value = float(gear_value)
                if args.noise_type == "gauss":
                    gear_value = gear_value + random.gauss(args.robust_force_mu, args.robust_force_sigma)  # robust setting
                elif args.noise_type == "shift":
                    gear_value = gear_value + args.noise_shift
                else:
                    gear_value = gear_value
                    print('\033[0;31m "No robust_force entropy learning! Using the original action" \033[0m')
                # print(f"The gear value for joint 'hip_4' is: {gear_value}") # gear="150"
                hip_4_motor.set('gear', str(gear_value))
                tree.write(file_name)
            else:
                print("No motor found for joint 'joint0'")

        if args.noise_factor == "robust_shape":
            # find right_back_leg's geom
            change_shape_mass = root.find(".//body[@name='body1']/geom")
            if change_shape_mass is not None:
                size_value = change_shape_mass.get('size')  # "0.046 .145"
                size_value = float(size_value)
                
                # size_floats = [float(x) for x in size_value.split()]
                # size_value = size_floats[0]
                size_value = float(size_value)
                if args.noise_type == "gauss":
                    size_value = size_value + random.gauss(args.robust_shape_mu,
                                                           args.robust_shape_sigma)  # robust setting
                elif args.noise_type == "shift":
                    size_value = size_value + args.noise_shift
                elif args.noise_type == "uniform":
                    size_value = np.random.uniform(args.uniform_min_val, args.uniform_max_val)
                    pass
                else:
                    size_value = size_value
                change_shape_mass.set('size', str(size_value))
                # print(f"The size of geom for 'right_back_leg' is: ", change_shape_mass.get('size'))
                tree.write(file_name)
            else:
                print("No geom found for body 'body1'")

    def replace_xml_content(self, source_file_path, target_file_path):
        # read data from source file
        with open(source_file_path, 'r', encoding='utf-8') as file:
            source_content = file.read()

        # write the data into the target file
        with open(target_file_path, 'w', encoding='utf-8') as file:
            file.write(source_content)

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
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )
