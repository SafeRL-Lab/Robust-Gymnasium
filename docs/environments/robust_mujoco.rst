.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Robust MuJoCo Tasks
--------------------------------

.. list-table:: Robust MuJoCo Tasks
   :widths: 30 20 20 20 20
   :header-rows: 1

   * - Tasks\Robust type
     - Robust State
     - Robust Action
     - Robust Reward
     - Robust Dynamics
   * - Ant-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - HalfCheetah-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - Hopper-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - Walker2d-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - Swimmer-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - Humanoid-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - HumanoidStandup-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - Pusher-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - Reacher-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - InvertedDoublePendulum-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅
   * - InvertedPendulum-v2-v3-v4-v5
     - ✅
     - ✅
     - ✅
     - ✅


.. Robust Ant-v4
.. ++++++++++++++

.. Robust Hopper-v4
.. ++++++++++++++++++++++++++++

At the initial and training steps, if we choose non-stationary attack as deterministic noise,

.. math::

   \text{Ant deterministic noise} = 
   \begin{cases}
      \text{Gravity} = 14.715, \\
      \text{Wind} = 1.0.
   \end{cases}

If we choose non-stationary attack as stochastic noise,

.. math::

   \text{Ant and Humanoid stochastic noise at initial steps} = 
   \begin{cases}
      \text{Gravity} \sim \text{Uniform}(9.81, 19.82), \\
      \text{Wind} \sim \text{Uniform}(0.8, 1.2).
   \end{cases}

During training steps, if we choose non-stationary attack as stochastic noise, where :math:`i_{\text{episode}}` denotes the training step number,

.. math::

   \text{Ant and Humanoid noise during training} = 
   \begin{cases}
      \text{Gravity} = 14.715 + 4.905 \cdot \sin(0.5 \cdot i_{\text{episode}}), \\
      \text{Wind} = 1.0 + 0.2 \cdot \sin(0.5 \cdot i_{\text{episode}}).
   \end{cases}

.. math::

   \text{Walker stochastic noise at initial steps} = 
   \begin{cases}
      \text{Torso Length} \sim \text{Uniform}(0.1, 0.3), \\
      \text{Foot Length} \sim \text{Uniform}(0.05, 0.15).
   \end{cases}

.. math::

   \text{Walker Stochastic noise} = 
   \begin{cases}
      \text{Torso Length} = 0.2 + 0.1 \cdot \sin(0.3 \cdot i_{\text{episode}}), \\
      \text{Foot Length} = 0.1 + 0.05 \cdot \sin(0.3 \cdot i_{\text{episode}}).
   \end{cases}

.. math::

   \text{Hopper stochastic noise at initial steps} = 
   \begin{cases}
      \text{Torso Length} \sim \text{Uniform}(0.3, 0.5), \\
      \text{Foot Length} \sim \text{Uniform}(0.29, 0.49).
   \end{cases}

.. math::

   \text{Walker Stochastic noise} = 
   \begin{cases}
      \text{Torso Length} = 0.4 + 0.1 \cdot \sin(0.2 \cdot i_{\text{episode}}), \\
      \text{Foot Length} = 0.39 + 0.1 \cdot \sin(0.2 \cdot i_{\text{episode}}).
   \end{cases}

Python Code Examples
--------------------

Example 1: Non-stationary Ant python code for initial steps.

.. code-block:: python

   if config.deter_noise:
       gravity = 14.715
       wind = 1.
   else:
       gravity = np.random.uniform(9.81, 19.82)
       wind = np.random.uniform(0.8, 1.2)

Example 2: Non-stationary Ant python code for training steps.

.. code-block:: python

   if config.deter_noise:
       gravity = 14.715
       wind = 1.
   else:
       gravity = 14.715 + 4.905 * np.sin(0.5 * i_episode)
       wind = 1. + 0.2 * np.sin(0.5 * i_episode)

Example 3: Non-stationary Walker python code for training steps.

.. code-block:: python

   if config.deter_noise:
       torso_len = 0.2
       foot_len = 0.1
   else:
       torso_len = 0.2 + 0.1 * np.sin(0.3 * i_episode)
       foot_len = 0.1 + 0.05 * np.sin(0.3 * i_episode)




`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__