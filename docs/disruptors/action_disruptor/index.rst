.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Action Disruptor
--------------------------------

**Action-disruptor**: 
    The real action :math:`a_t` chosen by the agent may be altered before or during execution in the environment due to implementation inaccuracies or system malfunctions. 
    The action-disruptor models this perturbation, outputting a perturbed action :math:`\tilde{a}_t = D_a(a_t)`, which is then executed in the environment for the next step.



.. code-block:: python

         def step(self, robust_input):
            action = robust_input["action"]
            args = robust_input["robust_config"]
            mu = args.noise_mu
            sigma = args.noise_sigma           

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

            if self.render_mode == "human":
                self.render()
            
            
            self.previous_reward = reward

            return observation, reward, terminated, False, info       

        def _get_obs(self):
            return self.simulation_state()

        def reset_model(self):
            self.previous_reward = 0
            return self._get_obs()


`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__