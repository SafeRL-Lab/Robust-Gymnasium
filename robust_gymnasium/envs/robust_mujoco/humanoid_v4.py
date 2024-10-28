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

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.xml_file = "humanoid.xml"
        self.xml_file_original = "humanoid_original.xml"
        self.previous_reward = 0
        self.llm_disturb_iteration = 0

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, robust_input):
        # action = action["action"]
        action = robust_input["action"]
        args = robust_input["robust_config"]
        mu = args.noise_mu
        sigma = args.noise_sigma

        if args.noise_factor == "robust_force":
            self.modify_xml(self.fullpath, args)
        if args.noise_factor == "robust_shape":
            self.modify_xml(self.fullpath, args)

        if args.noise_factor == "action":
            if args.noise_type == "gauss":
                action = action + random.gauss(mu, sigma)  # robust setting
            elif args.noise_type == "shift":
                action = action + args.noise_shift
            else:
                action = action
                print('\033[0;31m "No action entropy learning! Using the original action" \033[0m')
        else:
            action = action
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

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
        reward = rewards - ctrl_cost
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
        terminated = self.terminated
        fullpath_original = self.expand_model_path(self.xml_file_original)
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "source_file_path": fullpath_original,
            "target_file_path": self.fullpath,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
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
            
        return observation, reward, terminated, False, info

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
            hip_4_motor = root.find('.//motor[@joint="abdomen_y"]')
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
                print("No motor found for joint 'hip_4'")

        if args.noise_factor == "robust_shape":
            # find right_back_leg's geom
            change_shape_mass = root.find(".//body[@name='left_upper_arm']/geom")
            if change_shape_mass is not None:
                size_value = change_shape_mass.get('size')  # "0.046 .145"
                # print("size_value------:", size_value)
                size_floats = [float(x) for x in size_value.split()]
                size_value = size_floats[0]

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
                change_shape_mass.set('size', str(size_value) + str(size_floats[1]))
                # print(f"The size of geom for 'right_back_leg' is: ", change_shape_mass.get('size'))
                tree.write(file_name)
            else:
                print("No geom found for the robot")

    def replace_xml_content(self, source_file_path, target_file_path):
        # read data from source file
        with open(source_file_path, 'r', encoding='utf-8') as file:
            source_content = file.read()

        # write the data into the target file
        with open(target_file_path, 'w', encoding='utf-8') as file:
            file.write(source_content)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
