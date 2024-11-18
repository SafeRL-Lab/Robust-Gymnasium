.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Quick Start
--------------------------------

Installation of the Environments
**********************************

1. Create an environment (requires `Conda installation <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_):

   Use the following command to create a new Conda environment named ``robustgymnasium`` with Python 3.11:

   .. code-block:: bash

      conda create -n robustgymnasium python=3.11

   Activate the newly created environment:

   .. code-block:: bash

      conda activate robustgymnasium

2. Install dependency packages:

   Install the necessary packages using pip. Make sure you are in the project directory where the ``setup.py`` file is located:

   .. code-block:: bash

      pip install -r requirements.txt
      pip install -e .

Testing the Tasks
**********************

To run the tests, navigate to the ``examples`` directory and execute the test script, e.g.,

.. code-block:: bash

   cd examples/robust_action/mujoco/
   chmod +x test.sh
   ./test.sh

Ensure you follow these steps to set up and test the environment properly. Adjust paths and versions as necessary based on your specific setup requirements.

If you encounter any issues, please check the `existing solutions for reported issues <https://github.com/SafeRL-Lab/Robust-Gymnasium/issues?q=is%3Aissue+is%3Aclosed>`_, which could help you address your issue.


A Simple Example
**********************

.. code-block:: python

   import robust_gymnasium
   from robust_gymnasium.configs.robust_setting import get_config   

   env = robust_gymnasium.vector.make("Ant-v4", render_mode="human")
   observation, info = env.reset(seed=0)

   for _ in range(1000):
      args = get_config().parse_args()
      action = env.action_space.sample()
      robust_input = {"action": action, "robust_config": args}
      observation, reward, terminated, truncated, info = env.step(robust_input)

      if terminated or truncated:
         observation, info = env.reset()

   env.close()

Step-by-Step Introduction
=========================

This introduction demonstrates how to set up and use the `robust_gymnasium` library for robust reinforcement learning environments. We'll cover configuring tasks, selecting attack modes, managing attack frequencies, and running simulations.

Contents
--------

1. Importing Required Packages
2. Configuring the Environment and Robust Settings
3. Setting Up Experiment Data Recording
4. Running the Environment with Robust Attacks
5. Handling XML File Content Replacement
6. Wrapping Up and Closing the Environment

.. _import_packages:

1. Importing Required Packages
------------------------------

The following code imports all the necessary packages:

.. code-block:: python

    # Import packages
    import robust_gymnasium as gym
    from os import path
    import json
    import os
    import time
    from datetime import datetime

**Explanation**:

- `robust_gymnasium as gym`: The `robust_gymnasium` library is used for robust RL simulations.
- `os`, `path`, and `json`: Handle file operations and JSON configuration.
- `time` and `datetime`: Manage timestamps for recording experiments.

.. _configuring_environment:

2. Configuring the Environment and Robust Settings
--------------------------------------------------

We define the environment, attack settings, and other parameters:

.. code-block:: python

    from robust_gymnasium.configs.robust_setting import get_config
    args = get_config().parse_args()

    # choose robust task
    args.env_name = "Humanoid-v5"

    # choose attack type
    args.noise_factor = "state"

    # choose attack mode
    args.noise_type = "gauss"

    # attack frequency
    args.llm_disturb_interval = 500

**Explanation**:

- `get_config()`: Loads default configuration for robust settings.
- `args.env_name`: Specifies the simulation environment, e.g., `Humanoid-v5`.
- `args.noise_factor`: Specifies the aspect of the environment to attack, e.g., `state`.
- `args.noise_type`: Specifies the type of noise for the attack, e.g., Gaussian.
- `args.llm_disturb_interval`: Sets the interval (in steps) for attacks.

.. _recording_experiment:

3. Setting Up Experiment Data Recording
---------------------------------------

We set up a directory structure to save experiment logs:

.. code-block:: python

    folder = os.getcwd()[:0] + 'data/' + str(args.env_name) + '/' + str(args.noise_type) + '/' + str(
        start_run_date_and_time) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    json_path = folder + '/config.json'
    argsDict = args.__dict__

    with open(json_path, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

**Explanation**:

- Creates a folder structure to store data based on environment and attack settings.
- Saves the configuration in a `config.json` file for reproducibility.

.. _running_environment:

4. Running the Environment with Robust Attacks
----------------------------------------------

Set up the environment and run it with the specified robust settings:

.. code-block:: python

    # Create and render the environment
    env = gym.make(args.env_name, render_mode="human")

    # Reset environment
    observation, info = env.reset(seed=42)

    # Simulation loop
    try:
        for i in range(1000):
            action = env.action_space.sample()
            robust_input = {
                "action": action,
                "robust_type": "action",
                "robust_config": args,
            }

            observation, reward, terminated, truncated, info = env.step(robust_input)
            env.render()  # Render environment
            if terminated or truncated:
                observation, info = env.reset()

    finally:
        print('\033[0;31m "Program was terminated by user (Ctrl+C) or finished!" \033[0m')

**Explanation**:

- `env = gym.make()`: Creates the environment with rendering enabled.
- `env.reset()`: Initializes the environment.
- `env.step(robust_input)`: Executes an action with robust input parameters.
- `env.render()`: Renders the environment for visualization.
- The loop runs for 1000 iterations or until termination.

.. _xml_content:

5. Handling XML File Content Replacement
----------------------------------------

If XML file updates are required during the simulation, use the following function:

.. code-block:: python

    def replace_xml_content(source_file_path, target_file_path):
        # read data from source file
        with open(source_file_path, 'r', encoding='utf-8') as file:
            source_content = file.read()
        # write the data into the target file
        with open(target_file_path, 'w', encoding='utf-8') as file:
            file.write(source_content)

**Explanation**:

- Reads the content of a source XML file and writes it to a target XML file.
- Useful for updating environment configurations dynamically.

.. _wrapping_up:

6. Wrapping Up and Closing the Environment
------------------------------------------

Ensure the environment is closed properly to release resources:

.. code-block:: python

    env.close()

**Explanation**:

- Ensures all resources are released after the simulation.

**Conclusion**:

This tutorial provides a step-by-step guide to using the `robust_gymnasium` library for robust RL tasks. By following these sections, you can configure, run, and customize robust simulations efficiently.


A Simple Complete Example
**************************

.. code-block:: python

   # Import packages
    import robust_gymnasium as gym
    from os import path
    import json
    import os
    import time
    from datetime import datetime
    currentDateAndTime = datetime.now()
    start_run_date_and_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    from robust_gymnasium.configs.robust_setting import get_config
    args = get_config().parse_args()
    # choose robust task: choose any tasks that are listed in our benchmark, e.g., "InvertedDoublePendulum-v4",
    # "Reacher-v4", "Pusher-v4", "Ant-v4", etc.
    args.env_name = "Humanoid-v5"
    # choose attack type: choose any robust type that are list in our benchmark, such as state, reward, action, robust force (internal attack), wind (external attack)
    args.noise_factor = "state"
    # choose attack mode: we provide diverse attack modes, such as gaussian distribution attack, uniform 
    # distribution attack, LLM as adversary policy attack, etc.
    args.noise_type = "gauss"    
    # attack frequency: Different attack frequency settings are available. You can choose to perform an attack every 500 steps, 
    # every 100 steps, or customize it to any desired number of steps.
    args.llm_disturb_interval = 500
    # record experiment data
    folder = os.getcwd()[:0] + 'data/' + str(args.env_name) + '/' + str(args.noise_type) + '/' + str(
        start_run_date_and_time) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_path = folder + '/config.json'
    argsDict = args.__dict__
    with open(json_path, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    # env = gym.make("Ant-v4") # without render environments
    env = gym.make(args.env_name, render_mode="human")  # render environments: human, rgb_array, or depth_array.
      
    def replace_xml_content(source_file_path, target_file_path):
        # read data from source file
        with open(source_file_path, 'r', encoding='utf-8') as file:
            source_content = file.read()
        # write the data into the target file
        with open(target_file_path, 'w', encoding='utf-8') as file:
            file.write(source_content)

    observation, info = env.reset(seed=42)
    try:
        for i in range(1000):
            action = env.action_space.sample()
            robust_input = {
                "action": action,
                "robust_type": "action",
                "robust_config": args,
            }

            observation, reward, terminated, truncated, info = env.step(robust_input)            
            env.render()  # render environments
            if terminated or truncated:
                observation, info = env.reset()

            if i > 999:
                replace_xml_content(info["source_file_path"], info["target_file_path"])
    finally:  # except KeyboardInterrupt:
        replace_xml_content(info["source_file_path"], info["target_file_path"])
        print('\033[0;31m "Program was terminated by user (Ctrl+C) or finished!" \033[0m')

    env.close()
   


.. `Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

.. `Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__