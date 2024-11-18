.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Observation Disruptor
--------------------------------

A Simple Example: Ant Environment
=============================================

This module implements the `AntEnv` class for robust reinforcement learning using the MuJoCo physics engine.

Imports
-------
.. code-block:: python

    import numpy as np
    from os import path
    from robust_gymnasium import utils
    from robust_gymnasium.envs.robust_mujoco import MujocoEnv
    from robust_gymnasium.spaces import Box
    import xml.etree.ElementTree as ET
    import random
    from robust_gymnasium.envs.llm_guide_robust.gpt_collect import gpt_call
    DEFAULT_CAMERA_CONFIG = {
        "distance": 4.0,
    }

   

Classes
-------
AntEnv
~~~~~~

.. code-block:: python

    class AntEnv(MujocoEnv, utils.EzPickle):
        metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 20,
        }

        def __init__(
            self,
            xml_file="ant.xml",
            ctrl_cost_weight=0.5,
            use_contact_forces=False,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            **kwargs,
        ):
            utils.EzPickle.__init__(
                self,
                xml_file,
                ctrl_cost_weight,
                use_contact_forces,
                contact_cost_weight,
                healthy_reward,
                terminate_when_unhealthy,
                healthy_z_range,
                contact_force_range,
                reset_noise_scale,
                exclude_current_positions_from_observation,
                **kwargs,
            )
            self._ctrl_cost_weight = ctrl_cost_weight
            self._contact_cost_weight = contact_cost_weight
            self._healthy_reward = healthy_reward
            self._terminate_when_unhealthy = terminate_when_unhealthy
            self._healthy_z_range = healthy_z_range
            self._contact_force_range = contact_force_range
            self._reset_noise_scale = reset_noise_scale
            self._use_contact_forces = use_contact_forces
            self._exclude_current_positions_from_observation = (
                exclude_current_positions_from_observation
            )

            obs_shape = 27
            if not exclude_current_positions_from_observation:
                obs_shape += 2
            if use_contact_forces:
                obs_shape += 84

            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
            )

            MujocoEnv.__init__(
                self,
                xml_file,
                5,
                observation_space=observation_space,
                default_camera_config=DEFAULT_CAMERA_CONFIG,
                **kwargs,
            )
            self.xml_file = xml_file
            self.xml_file_original = "ant_original.xml"
            self.previous_reward = 0
            self.llm_disturb_iteration = 0

        @property
        def healthy_reward(self):
            return (
                float(self.is_healthy or self._terminate_when_unhealthy)
                * self._healthy_reward
            )

        def control_cost(self, action):
            control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
            return control_cost

        @property
        def contact_forces(self):
            raw_contact_forces = self.data.cfrc_ext
            min_value, max_value = self._contact_force_range
            contact_forces = np.clip(raw_contact_forces, min_value, max_value)
            return contact_forces

        @property
        def contact_cost(self):
            contact_cost = self._contact_cost_weight * np.sum(
                np.square(self.contact_forces)
            )
            return contact_cost

        @property
        def is_healthy(self):
            state = self.state_vector()
            min_z, max_z = self._healthy_z_range
            is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
            return is_healthy

        @property
        def terminated(self):
            terminated = not self.is_healthy if self._terminate_when_unhealthy else False
            return terminated

        def step(self, robust_input):
            action = robust_input["action"]
            args = robust_input["robust_config"]
            mu = args.noise_mu
            sigma = args.noise_sigma

            if args.noise_factor == "robust_force":
                self.modify_xml(self.fullpath, args)
            if args.noise_factor == "robust_shape":
                self.modify_xml(self.fullpath, args)

            if args.noise_factor == "action":
                self.llm_disturb_iteration += 1
                if self.llm_disturb_iteration % args.llm_disturb_interval == 0:
                    if args.noise_type == "gauss":
                        action = action + random.gauss(mu, sigma)  # robust setting
                    elif args.noise_type == "shift":
                        action = action + args.noise_shift  
                    elif args.noise_type =="uniform":
                        action = action + random.uniform(args.uniform_low, args.uniform_high)
                else:
                    action = action
            else:
                action = action

            xy_position_before = self.get_body_com("torso")[:2].copy()
            self.do_simulation(action, self.frame_skip)
            xy_position_after = self.get_body_com("torso")[:2].copy()

            xy_velocity = (xy_position_after - xy_position_before) / self.dt
            x_velocity, y_velocity = xy_velocity

            forward_reward = x_velocity
            healthy_reward = self.healthy_reward

            rewards = forward_reward + healthy_reward

            costs = ctrl_cost = self.control_cost(action)

            terminated = self.terminated

            if args.noise_factor == "state":
                self.llm_disturb_iteration += 1
                if self.llm_disturb_iteration % args.llm_disturb_interval == 0:
                    if args.noise_type == "gauss":
                        observation = self._get_obs() + random.gauss(mu, sigma)  # robust setting
                    elif args.noise_type == "shift":
                        observation = self._get_obs() + args.noise_shift
                    elif args.noise_type =="uniform":
                        observation = observation + random.uniform(args.uniform_low, args.uniform_high)
                else:
                    observation = self._get_obs()          
                
            else:
                observation = self._get_obs()

            fullpath_original = self.expand_model_path(self.xml_file_original)
            info = {
                "reward_forward": forward_reward,
                "reward_ctrl": -ctrl_cost,
                "reward_survive": healthy_reward,
                "x_position": xy_position_after[0],
                "y_position": xy_position_after[1],
                "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
                "x_velocity": x_velocity,
                "y_velocity": y_velocity,
                "forward_reward": forward_reward,
                "source_file_path": fullpath_original,
                "target_file_path": self.fullpath,
            }
            if self._use_contact_forces:
                contact_cost = self.contact_cost
                costs += contact_cost
                info["reward_ctrl"] = -contact_cost

            reward = rewards - costs

            if args.noise_factor == "reward":
                self.llm_disturb_iteration += 1
                if self.llm_disturb_iteration % args.llm_disturb_interval == 0:
                    if args.noise_type == "gauss":
                        reward = reward + random.gauss(mu, sigma)  # robust setting
                    elif args.noise_type == "shift":
                        reward = reward + args.noise_shift
                    elif args.noise_type =="uniform":
                        reward = reward + random.uniform(args.uniform_low, args.uniform_high)
                else:
                    reward = reward
            else:
                reward = reward

            if self.render_mode == "human":
                self.render()
            # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

            # self.modify_xml(xml_path_completion(xml_path), str(handle_width_left_door) + ' 0 ' + str(handle_height_door))  # <body name="latch" pos="-0.175 0 -0.025">

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
                        observation = gpt_call("the current observation is"+ str(observation))
            
            self.previous_reward = reward

            return observation, reward, terminated, False, info

        def expand_model_path(self, file_name):
            fullpath = path.join(self.model_dir, file_name)
            return fullpath

        def modify_xml(self, file_path, args):
            tree = ET.parse(file_path)
            root = tree.getroot()

            if args.noise_factor == "robust_force":
                for joint in root.iter("joint"):
                    joint.set("damping", str(random.uniform(args.damping_low, args.damping_high)))
                    joint.set("frictionloss", str(random.uniform(args.frictionloss_low, args.frictionloss_high)))
            if args.noise_factor == "robust_shape":
                for body in root.iter("body"):
                    body.set("pos", str(
                        random.uniform(args.pos_low, args.pos_high)) + ' ' + str(
                        random.uniform(args.pos_low, args.pos_high)) + ' ' + str(
                        random.uniform(args.pos_low, args.pos_high)))

            tree.write(file_path)

        def replace_xml_content(self, original_file_path, new_file_path):
            tree = ET.parse(original_file_path)
            root = tree.getroot()

            tree.write(new_file_path)

        def _get_obs(self):
            return self.simulation_state()

        def reset_model(self):
            self.previous_reward = 0
            return self._get_obs()



`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__