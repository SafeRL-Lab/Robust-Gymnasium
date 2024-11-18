.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Observation Disruptor
--------------------------------

**Observation-disruptor**: An agent's observations may not perfectly reflect the true status of the environment due to factors like sensor noise and time delays.

To model this sensing inaccuracy, we introduce an additional module—the observation-disruptor—which determines the agent's observations from the environment:

- **Agents' observed state** :math:`\tilde{s}_t`
    The observation-disruptor takes the true current state :math:`s_t` as input and outputs a perturbed state :math:`\tilde{s}_t = D_s(s_t)`. 
    The agent uses :math:`\tilde{s}_t` as input to its policy to select an action.

- **Agents' observed reward** :math:`\tilde{r}_t`
    The observation-disruptor takes the real immediate reward :math:`r_t` as input and outputs a perturbed reward :math:`\tilde{r}_t = D_r(r_t)`. 
    The agent observes :math:`\tilde{r}_t` and updates its policy accordingly.



.. code-block:: python

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
        elif args.llm_guide_type == "constraint":
            if self.llm_disturb_iteration % args.llm_disturb_interval == 0:
                prompt = "This is about a robust reinforcement learning setting; we want you as an adversary policy. If the current reward exceeds the previous reward value, please input some observation noises to disturb the environment and improve the learning algorithm's robustness. " \
                         "The noises should should be in this area:" +str((args.uniform_low, args.uniform_high))+ ", the current reward:" + str(reward) + ", the previous reward is" + str(self.previous_reward) \
                         + "please slightly revise the current environment state values:" + str(
                observation) + ", just output the revised state with its original format" \
                                   "do not output any other things."
                prompt_state = gpt_call(prompt)
                observation = prompt_state 
    

    

        def step(self, robust_input):
            action = robust_input["action"]
            args = robust_input["robust_config"]
            mu = args.noise_mu
            sigma = args.noise_sigma                     

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

        def _get_obs(self):
            return self.simulation_state()

        def reset_model(self):
            self.previous_reward = 0
            return self._get_obs()



`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__