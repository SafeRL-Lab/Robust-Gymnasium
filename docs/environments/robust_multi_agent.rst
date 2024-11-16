.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Robust Multi-Agent Tasks
--------------------------------

.. list-table:: Robust Multi-Agent Tasks
   :widths: 30 20 20 20
   :header-rows: 1

   * - Tasks\Robust type
     - Robust State
     - Robust Action
     - Robust Reward
   * - MA-Ant-2x4, 2x4d, 4x2, 4x1
     - ✅
     - ✅
     - ✅
   * - MA-HalfCheetah-2x3, 6x1
     - ✅
     - ✅
     - ✅
   * - MA-Hopper-3x1
     - ✅
     - ✅
     - ✅
   * - MA-Walker2d-2x3
     - ✅
     - ✅
     - ✅
   * - MA-Swimmer-2x1
     - ✅
     - ✅
     - ✅
   * - MA-Humanoid-9|8
     - ✅
     - ✅
     - ✅
   * - MA-HumanoidStandup-v4
     - ✅
     - ✅
     - ✅
   * - MA-Pusher-3p
     - ✅
     - ✅
     - ✅
   * - MA-Reacher-2x1
     - ✅
     - ✅
     - ✅
   * - Many-MA-Swimmer-10x2, 5x4, 6x1, 1x2
     - ✅
     - ✅
     - ✅
   * - Many-MA-Ant-2x3, 3x1
     - ✅
     - ✅
     - ✅
   * - CoupledHalfCheetah-p1p
     - ✅
     - ✅
     - ✅

**A Simple Example**

.. code-block:: python

    import numpy as np
    from robust_gymnasium.envs.robust_ma_mujoco.mamujoco_v1 import get_parts_and_edges
    from robust_gymnasium.envs.robust_ma_mujoco import mujoco_multi
    from robust_gymnasium.configs.robust_setting import get_config

    # Parse configuration
    args = get_config().parse_args()

    # Get nodes and edges for partitioning
    unpartioned_nodes, edges, global_nodes = get_parts_and_edges('Ant', None)
    print("unpartioned_nodes---:", unpartioned_nodes)

    # Define partitioned nodes and agent factorization
    partioned_nodes = [
        (unpartioned_nodes[0][0],), (unpartioned_nodes[0][1],),
        (unpartioned_nodes[0][2],), (unpartioned_nodes[0][3],),
        (unpartioned_nodes[0][4],), (unpartioned_nodes[0][5],),
        (unpartioned_nodes[0][6],), (unpartioned_nodes[0][7],)
    ]
    my_agent_factorization = {
        "partition": partioned_nodes,
        "edges": edges,
        "globals": global_nodes
    }

    # Initialize environment
    env = mujoco_multi.MultiAgentMujocoEnv(
        'Ant', '8x1', agent_factorization=my_agent_factorization, render_mode="human"
    )

    # Supported environments:
    # - MA-Ant-2x4, 2x4d, 4x2, 4x1
    # - MA-HalfCheetah-2x3, 6x1
    # - MA-Hopper-3x1
    # - MA-Walker2d-2x3
    # - MA-Swimmer-2x1
    # - MA-Humanoid-9|8
    # - MA-HumanoidStandup-v4
    # - MA-Pusher-3p
    # - MA-Reacher-2x1
    # - Many-MA-Swimmer-10x2, 5x4, 6x1, 1x2
    # - Many-MA-Ant-2x3, 3x1
    # - CoupledHalfCheetah-p1p

    # Reset environment and verify action mapping
    observation, info = env.reset(seed=42)
    action = env.single_agent_env.action_space.sample()
    assert (
        action == env.map_local_actions_to_global_action(
            env.map_global_action_to_local_actions(action)
        )
    ).all()

    # Generate actions for each agent
    actions = []
    n_actions = max([len(l) for l in my_agent_factorization['partition']])
    for agent_id in range(8):
        avail_actions = [1.0]  # Example available actions
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = np.random.uniform(-1.0, 1.0, n_actions)
        actions.append(action)

    # Predefined agent actions
    re_actions = {
        'agent_0': [-0.34627497], 'agent_1': [0.13449878], 'agent_2': [-0.02019556],
        'agent_3': [0.5044606], 'agent_4': [0.8911963], 'agent_5': [-0.91603947],
        'agent_6': [-0.8271199], 'agent_7': [0.7132667]
    }

    # Run simulation
    for i in range(1000):
        robust_input = {
            "action": re_actions,
            "robust_type": "state",
            "robust_config": args,
        }
        observation, reward, terminated, truncated, info = env.step(re_actions)
        env.render()
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__