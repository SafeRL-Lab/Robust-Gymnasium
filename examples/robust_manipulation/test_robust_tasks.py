"""
This demo script demonstrates the various functionalities of each controller available within robust_gymnasium.envs.robosuite.

For a given controller, runs through each dimension and executes a perturbation "test_value" from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time "steps_per_rest" before proceeding with the next action dim.

    E.g.: Given that the expected action space of the Pos / Ori (OSC_POSE) controller (without a gripper) is
    (dx, dy, dz, droll, dpitch, dyaw), the testing sequence of actions over time will be:

        ***START OF DEMO***
        ( dx,  0,  0,  0,  0,  0, grip)     <-- Translation in x-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0, dy,  0,  0,  0,  0, grip)     <-- Translation in y-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0, dz,  0,  0,  0, grip)     <-- Translation in z-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0, dr,  0,  0, grip)     <-- Rotation in roll (x) axis       for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0, dp,  0, grip)     <-- Rotation in pitch (y) axis      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0,  0, dy, grip)     <-- Rotation in yaw (z) axis        for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        ***END OF DEMO***

    Thus the OSC_POSE controller should be expected to sequentially move linearly in the x direction first,
        then the y direction, then the z direction, and then begin sequentially rotating about its x-axis,
        then y-axis, then z-axis.

Please reference the documentation of Controllers in the Modules section for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space. The expected
sequential qualitative behavior during the test is described below for each controller:

* OSC_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the global coordinate frame
* OSC_POSITION: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* IK_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the local robot end effector frame
* JOINT_POSITION: Robot Joints move sequentially in a controlled fashion
* JOINT_VELOCITY: Robot Joints move sequentially in a controlled fashion
* JOINT_TORQUE: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the
            "controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a
            "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

"""

import robust_gymnasium.envs.robosuite as suite
from robust_gymnasium.envs.robosuite.controllers import load_controller_config
from robust_gymnasium.envs.robosuite.robots import Bimanual
from robust_gymnasium.envs.robosuite.utils.input_utils import *
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robust_gymnasium.envs.robosuite v{}!".format(suite.__version__))
    # print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "Lift"  # 0 # door # choose_environment() # TODO: choose environments
    """
    Here is a list of environments in the suite:
    [0] Door
    [1] Lift
    [2] NutAssembly
    [3] NutAssemblyRound
    [4] NutAssemblySingle
    [5] NutAssemblySquare
    [6] PickPlace
    [7] PickPlaceBread
    [8] PickPlaceCan
    [9] PickPlaceCereal
    [10] PickPlaceMilk
    [11] PickPlaceSingle
    [12] Stack
    [13] ToolHang
    [14] TwoArmHandover
    [15] TwoArmLift
    [16] TwoArmPegInHole
    [17] TwoArmTransport
    [18] Wipe
    [19] MultiRobustDoor
    """

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        # options["robots"] = []
        # # Have user choose two robots
        # print("A multiple single-arm configuration was chosen.\n")
        # for i in range(2):
        #     print("Please choose Robot {}...\n".format(i))
        #     options["robots"].append(choose_robots(exclude_bimanual=True))
        #     print("options------:", options)

        # options["robots"] = ['IIWA', 'Panda']
        options["robots"] = 'IIWA'  #  # choose_robots(exclude_bimanual=True) # TODO: choose robots

    """
    Here is a list of available robots:
    [0] IIWA
    [1] Jaco
    [2] Kinova3
    [3] Panda
    [4] Sawyer
    [5] UR5e
    """

    # Hacky way to grab joint dimension for now
    joint_dim = 6 if options["robots"] == "UR5e" else 7

    # Choose controller
    controller_name = 'JOINT_VELOCITY'  # JOINT_VELOCITY - Joint Velocity # choose_controller() # TODO: Choose control methods

    """"
    Here is a list of controllers in the suite:
    [0] JOINT_VELOCITY - Joint Velocity
    [1] JOINT_TORQUE - Joint Torque
    [2] JOINT_POSITION - Joint Position
    [3] OSC_POSITION - Operational Space Control (Position Only)
    [4] OSC_POSE - Operational Space Control (Position + Orientation)
    [5] IK_POSE - Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)
    """

    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

    # Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value)
    controller_settings = {
        "OSC_POSE": [6, 6, 0.1],
        "OSC_POSITION": [3, 3, 0.1],
        "IK_POSE": [6, 6, 0.01],
        "JOINT_POSITION": [joint_dim, joint_dim, 0.2],
        "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
        "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
    }

    # Define variables for each controller test
    action_dim = controller_settings[controller_name][0]
    num_test_steps = controller_settings[controller_name][1]
    test_value = controller_settings[controller_name][2]

    # Define the number of timesteps to use per controller action as well as timesteps in between actions
    steps_per_action = 750
    steps_per_rest = 750

    # initialize the task
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

    # To accommodate for multi-arm settings (e.g.: Baxter), we need to make sure to fill any extra action space
    # Get total number of arms being controlled
    n = 0
    gripper_dim = 0
    for robot in env.robots:
        gripper_dim = robot.gripper["right"].dof if isinstance(robot, Bimanual) else robot.gripper.dof
        n += int(robot.action_dim / (action_dim + gripper_dim))

    # Define neutral value
    neutral = np.zeros(action_dim + gripper_dim)

    # Keep track of done variable to know when to break loop
    count = 0
    # Loop through controller space
    while count < num_test_steps:
        action = neutral.copy()
        for i in range(steps_per_action):
            if controller_name in {"IK_POSE", "OSC_POSE"} and count > 2:
                # Set this value to be the scaled axis angle vector
                vec = np.zeros(3)
                vec[count - 3] = test_value
                action[3:6] = vec
            else:
                action[count] = test_value
            total_action = np.tile(action, n)
            robust_input = {
                "action": total_action,
                "robust_type": "state",
                "robust_config": args,
            }
            env.step(robust_input)
            env.render()
        for i in range(steps_per_rest):
            total_action = np.tile(neutral, n)
            robust_input = {
                "action": total_action,
                "robust_type": "state",
                "robust_config": args,
            }
            env.step(robust_input)
            env.render()
        count += 1

    # Shut down this env before starting the next test
    env.close()
