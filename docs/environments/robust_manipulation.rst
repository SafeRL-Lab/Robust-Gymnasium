.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Robust Robot Manipulation Tasks
--------------------------------

.. list-table:: Robust Robot Manipulation Tasks
   :widths: 30 20 20 20
   :header-rows: 1

   * - Tasks\Robust type
     - Robust State
     - Robust Action
     - Robust Reward
   * - RobustLift
     - ✅
     - ✅
     - ✅
   * - RobustDoor
     - ✅
     - ✅
     - ✅
   * - RobustNutAssembly
     - ✅
     - ✅
     - ✅
   * - RobustPickPlace
     - ✅
     - ✅
     - ✅
   * - RobustStack
     - ✅
     - ✅
     - ✅
   * - RobustWipe
     - ✅
     - ✅
     - ✅
   * - RobustToolHang
     - ✅
     - ✅
     - ✅
   * - RobustTwoArmLift
     - ✅
     - ✅
     - ✅
   * - RobustTwoArmPegInHole
     - ✅
     - ✅
     - ✅
   * - RobustTwoArmHandover
     - ✅
     - ✅
     - ✅
   * - RobustTwoArmTransport
     - ✅
     - ✅
     - ✅
   * - MultiRobustDoor
     - ✅
     - ✅
     - ✅

**A Simple Example**

This demonstrates the various functionalities of each controller available within ``robust_gymnasium.envs.robosuite``.

For a given controller, the script runs through each dimension and executes a perturbation ``test_value`` from its neutral (stationary) value for a certain amount of time ``steps_per_action``, then returns to all neutral values for ``steps_per_rest`` before proceeding with the next action dimension.

Example of the testing sequence of actions over time for the Pos / Ori (OSC_POSE) controller (without a gripper):

Expected sequential behavior:

* OSC_POSE: Gripper moves sequentially and linearly in x, y, z direction, then rotates in x, y, z axes relative to the global coordinate frame.
* OSC_POSITION: Gripper moves sequentially in x, y, z direction relative to the global frame.
* IK_POSE: Gripper moves in x, y, z direction, then rotates in x, y, z axes relative to the local robot end effector frame.
* JOINT_POSITION: Robot joints move sequentially in a controlled fashion.
* JOINT_VELOCITY: Robot joints move sequentially in a controlled fashion.
* JOINT_TORQUE: Joint torque controller acts sluggishly, as 0 torque will not guarantee a stable robot if it has non-zero velocity.

Code:

.. code-block:: python

    import robust_gymnasium.envs.robosuite as suite
    from robust_gymnasium.envs.robosuite.controllers import load_controller_config
    from robust_gymnasium.envs.robosuite.robots import Bimanual
    from robust_gymnasium.envs.robosuite.utils.input_utils import *
    from robust_gymnasium.configs.robust_setting import get_config
    args = get_config().parse_args()

    if __name__ == "__main__":
        options = {}
        print("Welcome to robust_gymnasium.envs.robosuite v{}!".format(suite.__version__))
        options["env_name"] = "Door"  # Choose environment

        if "TwoArm" in options["env_name"]:
            options["env_configuration"] = choose_multi_arm_config()
            options["robots"] = "Baxter" if options["env_configuration"] == "bimanual" else [choose_robots(exclude_bimanual=True) for _ in range(2)]
        else:
            options["robots"] = 'IIWA'  # Choose robot

        joint_dim = 6 if options["robots"] == "UR5e" else 7
        controller_name = 'JOINT_VELOCITY'
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

        controller_settings = {
            "OSC_POSE": [6, 6, 0.1],
            "OSC_POSITION": [3, 3, 0.1],
            "IK_POSE": [6, 6, 0.01],
            "JOINT_POSITION": [joint_dim, joint_dim, 0.2],
            "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
            "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
        }

        action_dim = controller_settings[controller_name][0]
        num_test_steps = controller_settings[controller_name][1]
        test_value = controller_settings[controller_name][2]

        steps_per_action = 750
        steps_per_rest = 750
        robust_setting = {"robust_type": "state", "robust_config": args}
        env = suite.make(
            **options,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            horizon=(steps_per_action + steps_per_rest) * num_test_steps,
            control_freq=20,
        )
        env.reset()
        env.viewer.set_camera(camera_id=0)

        n = sum(int(robot.action_dim / (action_dim + robot.gripper.dof)) for robot in env.robots)
        neutral = np.zeros(action_dim + robot.gripper.dof)

        count = 0
        while count < num_test_steps:
            action = neutral.copy()
            for i in range(steps_per_action):
                if controller_name in {"IK_POSE", "OSC_POSE"} and count > 2:
                    vec = np.zeros(3)
                    vec[count - 3] = test_value
                    action[3:6] = vec
                else:
                    action[count] = test_value
                total_action = np.tile(action, n)
                robust_input = {"action": total_action, "robust_type": "state", "robust_config": args}
                env.step(robust_input)
                env.render()
            for i in range(steps_per_rest):
                total_action = np.tile(neutral, n)
                robust_input = {"action": total_action, "robust_type": "state", "robust_config": args}
                env.step(robust_input)
                env.render()
            count += 1

        env.close()


.. `Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

.. `Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__