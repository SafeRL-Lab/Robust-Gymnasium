from robust_gymnasium.envs.robosuite.controllers import load_controller_config
from robust_gymnasium.envs.robosuite.utils.input_utils import *

from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robust_gymnasium.envs.robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    # options["env_name"] = choose_environment()
    options["env_name"] = "MultiRobustDoor"  # 0 # door # choose_environment() # TODO: choose environments

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
        options["robots"] = ['Panda', 'IIWA']
        # options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    # controller_name = choose_controller()
    controller_name = 'JOINT_VELOCITY'

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0) # 0

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(2000):
        action = np.random.uniform(low, high)
        robust_input = {
            "action": action,
            "robust_type": "state",
            "robust_config": args,
        }
        # env.step(robust_input)
        obs, reward, done, _ = env.step(robust_input)
        # print("obs-------:", obs)
        env.render()
