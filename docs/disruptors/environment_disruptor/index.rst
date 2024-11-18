.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Environment Disruptor
--------------------------------

**Environment-disruptor**: 
    Recall that a task environment consists of both the internal dynamic model and the external workspace it interacts with, characterized by its transition dynamics :math:`P` and reward function :math:`r`. 
    The environment during training can differ from the real-world environment due to factors such as the sim-to-real gap, human and natural variability, external disturbances, and more. 
    We attribute this potential nonstationarity to an environment-disruptor, which determines the actual environment :math:`(P, r)` the agent is interacting with at any given moment. 
    These dynamics may differ from the nominal environment :math:`(P^0, r^0)` that the agent was originally expected to interact with.



.. code-block:: python

         def step(self, robust_input):
            action = robust_input["action"]
            args = robust_input["robust_config"]
            mu = args.noise_mu
            sigma = args.noise_sigma

            if args.noise_factor == "robust_force":
                self.modify_xml(self.fullpath, args)
            if args.noise_factor == "robust_shape":
                self.modify_xml(self.fullpath, args)                  

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

            if self.render_mode == "human":
                self.render()
            # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

            # self.modify_xml(xml_path_completion(xml_path), str(handle_width_left_door) + ' 0 ' + str(handle_height_door))  # <body name="latch" pos="-0.175 0 -0.025">         
            
            
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