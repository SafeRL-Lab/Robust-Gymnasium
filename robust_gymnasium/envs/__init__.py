"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from robust_gymnasium.envs.registration import make, pprint_registry, register, registry, spec
from robust_gymnasium.envs import robosuite

from robust_gymnasium.envs.robust_maze import maps
from robust_gymnasium.envs import robust_ma_mujoco
from robust_gymnasium.envs.gymnasium_robotics_core import GoalEnv
from robust_gymnasium.envs.robust_ma_mujoco import mamujoco_v1

# Classic
# ----------------------------------------

register(
    id="CartPole-v0",
    entry_point="robust_gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    vector_entry_point="robust_gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="CartPole-v1",
    entry_point="robust_gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    vector_entry_point="robust_gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="MountainCar-v0",
    entry_point="robust_gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="MountainCarContinuous-v0",
    entry_point="robust_gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id="Pendulum-v1",
    entry_point="robust_gymnasium.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="Acrobot-v1",
    entry_point="robust_gymnasium.envs.classic_control.acrobot:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)


# Phys2d (jax classic control)
# ----------------------------------------

register(
    id="phys2d/CartPole-v0",
    entry_point="robust_gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
    vector_entry_point="robust_gymnasium.envs.phys2d.cartpole:CartPoleJaxVectorEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
    disable_env_checker=True,
)

register(
    id="phys2d/CartPole-v1",
    entry_point="robust_gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
    vector_entry_point="robust_gymnasium.envs.phys2d.cartpole:CartPoleJaxVectorEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
    disable_env_checker=True,
)

register(
    id="phys2d/Pendulum-v0",
    entry_point="robust_gymnasium.envs.phys2d.pendulum:PendulumJaxEnv",
    vector_entry_point="robust_gymnasium.envs.phys2d.pendulum:PendulumJaxVectorEnv",
    max_episode_steps=200,
    disable_env_checker=True,
)

# Robust Box2d
# ----------------------------------------

register(
    id="LunarLander-v3",
    entry_point="robust_gymnasium.envs.robust_box2d.lunar_lander:LunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuous-v3",
    entry_point="robust_gymnasium.envs.robust_box2d.lunar_lander:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="BipedalWalker-v3",
    entry_point="robust_gymnasium.envs.robust_box2d.bipedal_walker:BipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id="BipedalWalkerHardcore-v3",
    entry_point="robust_gymnasium.envs.robust_box2d.bipedal_walker:BipedalWalker",
    kwargs={"hardcore": True},
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id="CarRacing-v2",
    entry_point="robust_gymnasium.envs.robust_box2d.car_racing:CarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id="Blackjack-v1",
    entry_point="robust_gymnasium.envs.toy_text.blackjack:BlackjackEnv",
    kwargs={"sab": True, "natural": False},
)

register(
    id="FrozenLake-v1",
    entry_point="robust_gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FrozenLake8x8-v1",
    entry_point="robust_gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,  # optimum = 0.91
)

register(
    id="CliffWalking-v0",
    entry_point="robust_gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
)

register(
    id="Taxi-v3",
    entry_point="robust_gymnasium.envs.toy_text.taxi:TaxiEnv",
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=200,
)


# Tabular
# ----------------------------------------

register(
    id="tabular/Blackjack-v0",
    entry_point="robust_gymnasium.envs.tabular.blackjack:BlackJackJaxEnv",
    disable_env_checker=True,
)

register(
    id="tabular/CliffWalking-v0",
    entry_point="robust_gymnasium.envs.tabular.cliffwalking:CliffWalkingJaxEnv",
    disable_env_checker=True,
)


# Robust Mujoco
# ----------------------------------------

# manipulation

register(
    id="Reacher-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.reacher:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.reacher_v4:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.reacher_v5:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Pusher-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.pusher:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Pusher-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.pusher_v4:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Pusher-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.pusher_v5:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

# balance

register(
    id="InvertedPendulum-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_pendulum:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedPendulum-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedPendulum-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_pendulum_v5:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedDoublePendulum-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="InvertedDoublePendulum-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="InvertedDoublePendulum-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_double_pendulum_v5:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

# runners

register(
    id="HalfCheetah-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v3",
    entry_point="robust_gymnasium.envs.robust_mujoco.half_cheetah_v3:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.half_cheetah_v4:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.half_cheetah_v5:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Hopper-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.hopper:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v3",
    entry_point="robust_gymnasium.envs.robust_mujoco.hopper_v3:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.hopper_v4:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.hopper_v5:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Swimmer-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.swimmer:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v3",
    entry_point="robust_gymnasium.envs.robust_mujoco.swimmer_v3:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.swimmer_v4:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.swimmer_v5:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Walker2d-v2",
    max_episode_steps=1000,
    entry_point="robust_gymnasium.envs.robust_mujoco.walker2d:Walker2dEnv",
)

register(
    id="Walker2d-v3",
    max_episode_steps=1000,
    entry_point="robust_gymnasium.envs.robust_mujoco.walker2d_v3:Walker2dEnv",
)

register(
    id="Walker2d-v4",
    max_episode_steps=1000,
    entry_point="robust_gymnasium.envs.robust_mujoco.walker2d_v4:Walker2dEnv",
)

register(
    id="Walker2d-v5",
    max_episode_steps=1000,
    entry_point="robust_gymnasium.envs.robust_mujoco.walker2d_v5:Walker2dEnv",
)

register(
    id="Ant-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.ant:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v3",
    entry_point="robust_gymnasium.envs.robust_mujoco.ant_v3:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.ant_v4:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.ant_v5:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)



register(
    id="Humanoid-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoid:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v3",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoid_v3:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoid_v5:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v2",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoidstandup:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v4",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoidstandup_v4:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v5",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoidstandup_v5:HumanoidStandupEnv",
    max_episode_steps=1000,
)

# Robust-MA-MuJoCo
# ----------------------------------------
register(
    id="Ant-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.ant_v5s:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="HalfCheetah-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.half_cheetah_v5s:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="MA-HalfCheetah",
    entry_point="robust_gymnasium.envs.robust_ma_mujoco.mujoco_multi:MultiAgentMujocoEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Hopper-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.hopper_v5s:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Humanoid-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoid_v5s:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.humanoidstandup_v5s:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="InvertedDoublePendulum-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_double_pendulum_v5s:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="InvertedPendulum-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.inverted_pendulum_v5s:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="Pusher-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.pusher_v5s:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Reacher-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.reacher_v5s:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Swimmer-v5s",
    entry_point="robust_gymnasium.envs.robust_mujoco.swimmer_v5s:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Walker2d-v5s",
    max_episode_steps=1000,
    entry_point="robust_gymnasium.envs.robust_mujoco.walker2d_v5s:Walker2dEnv",
)

# Safety MuJoCo
# ----------------------------------------

register(
    id="SafetyAnt-v4",
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.ant_v4:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="SafetyHalfCheetah-v4",
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.half_cheetah_v4:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="SafetyHopper-v4",
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.hopper_v4:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="SafetyHumanoid-v4",
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="SafetyHumanoidStandup-v4",
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.humanoidstandup_v4:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="SafetyPusher-v4",
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.pusher_v4:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="SafetyReacher-v4",
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.reacher_v4:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="SafetyWalker2d-v4",
    max_episode_steps=1000,
    entry_point="robust_gymnasium.envs.robust_safety_mujoco.walker2d_v4:Walker2dEnv",
)

# --- For shimmy compatibility
def _raise_shimmy_error(*args: Any, **kwargs: Any):
    raise ImportError(
        'To use the gym compatibility environments, run `pip install "shimmy[gym-v21]"` or `pip install "shimmy[gym-v26]"`'
    )


# When installed, shimmy will re-register these environments with the correct entry_point
register(id="GymV21Environment-v0", entry_point=_raise_shimmy_error)
register(id="GymV26Environment-v0", entry_point=_raise_shimmy_error)


"""
register for gymnasium robotics start 
"""
__version__ = "0.0.1"


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        # Fetch
        register(
            id=f"FetchSlide{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_fetch.slide:MujocoPyFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchSlide{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_fetch.slide:MujocoFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_fetch.pick_and_place:MujocoPyFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_fetch.reach:MujocoPyFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_fetch.reach:MujocoFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_fetch.push:MujocoPyFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_fetch.push:MujocoFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        # Hand
        register(
            id=f"HandReach{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.reach:MujocoPyHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandReach{suffix}-v2",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.reach:MujocoHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateBlock{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateEgg{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v0",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v1",
            entry_point="robust_gymnasium.envs.robust_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        #####################
        # D4RL Environments #
        #####################

        # ----- AntMaze -----

        register(
            id=f"AntMaze_UMaze{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_UMaze{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_G{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_G{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_GR{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_GR{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Medium{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_G{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_GR{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_G{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_G{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_GR{suffix}-v4",
            entry_point="robust_gymnasium.envs.robust_maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        # ----- PointMaze -----

        register(
            id=f"PointMaze_UMaze{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open_Diverse_G{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open_Diverse_GR{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Medium{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Large{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_G{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="robust_gymnasium.envs.robust_maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        # robust non-stationary envs
        # from https://github.com/Roythuly/OMPO/tree/main/Non_stationary_env
        register(
            id='HopperRandom-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.random_hopper_v5:HopperRandomEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
        )

        register(
            id='HopperTransfer-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.transfer_hopper_v5:HopperTransferEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
        )

        register(
            id='Walker2dRandom-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.random_walker2d_v5:Walker2dRandomEnv',
            max_episode_steps=1000
        )

        register(
            id='Walker2dTransfer-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.transfer_walker2d_v5:Walker2dTransferEnv',
            max_episode_steps=1000
        )

        register(
            id='AntTransfer-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.transfer_ant_v5:AntTransferEnv',
            max_episode_steps=1000,
            reward_threshold=6000.0,
        )

        register(
            id='AntRandom-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.random_ant_v5:AntRandomEnv',
            max_episode_steps=1000,
            reward_threshold=6000.0,
        )

        register(
            id='HumanoidTransfer-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.transfer_humanoid_v5:HumanoidTransferEnv',
            max_episode_steps=1000
        )

        register(
            id='HumanoidRandom-v5',
            entry_point='robust_gymnasium.envs.robust_nonstationary_env.random_humanoid_v5:HumanoidRandomEnv',
            max_episode_steps=1000
        )

    for reward_type in ["sparse", "dense"]:
        suffix = "Sparse" if reward_type == "sparse" else ""
        version = "v1"
        kwargs = {
            "reward_type": reward_type,
        }

        register(
            id=f"AdroitHandDoor{suffix}-{version}",
            entry_point="robust_gymnasium.envs.robust_adroit_hand.adroit_door:AdroitHandDoorEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandHammer{suffix}-{version}",
            entry_point="robust_gymnasium.envs.robust_adroit_hand.adroit_hammer:AdroitHandHammerEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandPen{suffix}-{version}",
            entry_point="robust_gymnasium.envs.robust_adroit_hand.adroit_pen:AdroitHandPenEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandRelocate{suffix}-{version}",
            entry_point="robust_gymnasium.envs.robust_adroit_hand.adroit_relocate:AdroitHandRelocateEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

    register(
        id="FrankaKitchen-v1",
        entry_point="robust_gymnasium.envs.robust_franka_kitchen:KitchenEnv",
        max_episode_steps=280,
    )


register_robotics_envs()


try:
    import sys

    from farama_notifications import notifications

    if (
        "gymnasium_robotics" in notifications
        and __version__ in notifications["gymnasium_robotics"]
    ):
        print(notifications["gymnasium_robotics"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass

"""
register for gymnasium robotics end
"""
