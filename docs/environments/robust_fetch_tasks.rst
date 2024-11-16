.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Robust Fetch Tasks
--------------------------------

.. list-table:: Robust Fetch Manipulation Tasks
   :widths: 30 20 20 20
   :header-rows: 1

   * - Tasks\Robust type
     - Robust State
     - Robust Action
     - Robust Reward
   * - RobustFetchPush-v3
     - ✅
     - ✅
     - ✅
   * - RobustFetchReach-v3
     - ✅
     - ✅
     - ✅
   * - RobustFetchSlide-v3
     - ✅
     - ✅
     - ✅
   * - RobustFetchPickAndPlace-v3
     - ✅
     - ✅
     - ✅

**A Simple Example**

.. code-block:: python

    import robust_gymnasium as gym
    import json
    import os
    import time
    from datetime import datetime

    # Get current date and time for folder naming
    currentDateAndTime = datetime.now()
    start_run_date_and_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # Import configuration settings
    from robust_gymnasium.configs.robust_setting import get_config
    args = get_config().parse_args()

    # Set environment and noise factor
    args.env_name = "FetchReach-v3"  # Options: FetchPush-v3, FetchSlide-v3, FetchPickAndPlace-v3, etc.
    args.noise_factor = "state"

    # Define folder path for storing data
    folder = os.getcwd()[:0] + 'data/' + str(args.env_name) + '/' + str(args.noise_type) + '/' + str(
        start_run_date_and_time) + '/'
    print("folder---:", folder)

    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save configuration settings to a JSON file
    json_path = folder + '/config.json'
    argsDict = args.__dict__
    with open(json_path, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    # Initialize environment
    env = gym.make("FetchSlide-v3", render_mode="human")
    # Options: FetchPush-v3, FetchReach-v3, FetchSlide-v3, FetchPickAndPlace-v3
    env.reset()

    # Function to test the environment with robust configurations
    def test_robust():
        observation, info = env.reset(seed=42)
        for i in range(10000):
            action = env.action_space.sample()
            robust_input = {
                "action": action,
                "robust_type": "state",
                "robust_config": args,
            }
            obs, reward, terminated, truncated, info = env.step(robust_input)
            env.render()  # Render environment
            if terminated or truncated:
                observation, info = env.reset()
        env.close()

    # # Assertions for goal-based environments
    # assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
    # assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
    # assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)
    #
    # # Example of substituting goals:
    # substitute_goal = obs["achieved_goal"].copy()
    # substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
    # substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
    # substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)

    if __name__ == '__main__':
        test_robust()


`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__