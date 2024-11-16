.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Robust Dexterous Tasks
--------------------------------

.. list-table:: Robust Dexterous Tasks
   :widths: 30 20 20 20
   :header-rows: 1

   * - Tasks\Robust type
     - Robust State
     - Robust Action
     - Robust Reward
   * - RobustHandManipulateEgg_BooleanTouchSensors-v1
     - ✅
     - ✅
     - ✅
   * - RobustHandReach-v2
     - ✅
     - ✅
     - ✅
   * - RobustHandManipulateBlock-v1
     - ✅
     - ✅
     - ✅
   * - RobustHandManipulateEgg-v1
     - ✅
     - ✅
     - ✅
   * - RobustHandManipulatePen-v1
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
    args.env_name = "AdroitHandDoor-v1"  # Other options: "InvertedDoublePendulum-v4", "Reacher-v4", etc.
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
    env = gym.make('HandManipulateEgg_BooleanTouchSensors-v1', max_episode_steps=400, render_mode="human")

    """
    Available Tasks:
    - HandManipulateEgg_BooleanTouchSensors-v1
    - HandReach-v2
    - HandManipulateBlock-v1
    - HandManipulateEgg-v1
    - HandManipulatePen-v1
    """

    # Reset environment and run simulation
    observation, info = env.reset(seed=42)
    for i in range(10000):
        action = env.action_space.sample()
        robust_input = {
            "action": action,
            "robust_type": "state",
            "robust_config": args,
        }
        observation, reward, terminated, truncated, info = env.step(robust_input)
        
        # Render environment
        env.render()
        if terminated or truncated:
            observation, info = env.reset()

    # Close environment
    env.close()


`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__