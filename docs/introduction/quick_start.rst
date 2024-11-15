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


`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__