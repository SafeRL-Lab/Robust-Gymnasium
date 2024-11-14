.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Robust Gymnasium documentation
==============================


.. toctree::
   :maxdepth: 4
   :caption: Contents:


Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning.

.. image:: _static/images/gif-edit-overview.gif
   :alt: racecar
   :width: 500
   :align: center

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


.. toctree::
   :hidden:
   :caption: Introduction

   introduction/overview
   introduction/quick_start


.. toctree::
   :hidden:
   :caption: Environments

   environments/robust_mujoco
   environments/robust_safe_agent
   environments/robust_multi_agent


.. toctree::
   :hidden:
   :caption: Disruptor

   disruptors/Gaussian_attack/index
   disruptors/Uniform_attack/index
   disruptors/LLM_attack/index


.. toctree::
   :hidden:
   :caption: API

   api/utils/index


.. toctree::
   :hidden:
   :caption: Development

`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/SafeRL-Lab/Robust-Gymnasium/tree/main/docs>`__