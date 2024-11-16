.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Robust Maze Tasks
--------------------------------

.. list-table:: Robust Maze Tasks
   :widths: 30 20 20 20
   :header-rows: 1

   * - Tasks\Robust type
     - Robust State
     - Robust Action
     - Robust Reward
   * - AntMaze_UMaze-v4
     - ✅
     - ✅
     - ✅
   * - PointMaze_UMaze-v3
     - ✅
     - ✅
     - ✅

**A Simple Example**

.. code-block:: python

    import robust_gymnasium as gym
    import numpy as np
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
    args.env_name = "AntMaze_UMaze-v4"
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

    # Function to test AntMaze environment reset behavior
    def test_reset():
        """Check that AntMaze does not reset into a success state."""
        env = gym.make("AntMaze_UMaze-v4", continuing_task=True, render_mode="human")

        for _ in range(10000):
            obs, info = env.reset()
            assert not info["success"]
            dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
            env.render()  # Render environment
            assert dist > 0.45, f"dist={dist} < 0.45"

    if __name__ == '__main__':
        test_reset()


.. `Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

.. `Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__