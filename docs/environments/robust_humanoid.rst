.. Robust Gymnasium documentation master file, created by Robust RL Team
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Robust Humanoid Tasks
--------------------------------

.. list-table:: Robust Humanoid Tasks
   :widths: 30 20 20 20
   :header-rows: 1

   * - Tasks\Robust type
     - Robust State
     - Robust Action
     - Robust Reward
   * - Robust1hand-reach-v0
     - ✅
     - ✅
     - ✅
   * - Robust1hand-push-v0
     - ✅
     - ✅
     - ✅
   * - h1hand-truck-v0
     - ✅
     - ✅
     - ✅
   * - Robust1hand-slide-v0
     - ✅
     - ✅
     - ✅

**A Simple Example**

.. code-block:: python

    import argparse
    import pathlib
    import cv2
    import gymnasium as gym
    import numpy as np

    # Import robust_gymnasium modules
    from robust_gymnasium.envs.robust_humanoid.env import ROBOTS, TASKS
    from robust_gymnasium.configs.robust_setting import get_config

    # Parse robust gymnasium arguments
    robust_args = get_config().parse_args()

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
        parser.add_argument("--env", default="h1-walk-v0", help="e.g. h1-walk-v0")
        # Examples: h1-push-v0, h1-reach-v0, h1-slide-v0, h1-pole-v0, etc.
        parser.add_argument("--keyframe", default=None)
        parser.add_argument("--policy_path", default=None)
        parser.add_argument("--mean_path", default=None)
        parser.add_argument("--var_path", default=None)
        parser.add_argument("--policy_type", default=None)
        parser.add_argument("--small_obs", default="False")
        parser.add_argument("--obs_wrapper", default="False")
        parser.add_argument("--sensors", default="")
        parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
        args = parser.parse_args()

        # Prepare environment arguments
        kwargs = vars(args).copy()
        kwargs.pop("env")
        kwargs.pop("render_mode")
        if kwargs["keyframe"] is None:
            kwargs.pop("keyframe")
        print(f"arguments: {kwargs}")

        # Test offscreen rendering
        print(f"Test offscreen mode...")
        env = gym.make(args.env, render_mode="rgb_array", **kwargs)
        ob, _ = env.reset()
        if isinstance(ob, dict):
            print(f"ob_space = {env.observation_space}")
            print(f"ob = ")
            for k, v in ob.items():
                print(f"  {k}: {v.shape}")
        else:
            print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
        print(f"ac_space = {env.action_space.shape}")

        # Render and save an offscreen image
        img = env.render()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("test_env_img.png", rgb_img)

        # Test interactive rendering
        print(f"Test onscreen mode...")
        env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
        ob, _ = env.reset()
        if isinstance(ob, dict):
            print(f"ob_space = {env.observation_space}")
            print(f"ob = ")
            for k, v in ob.items():
                print(f"  {k}: {v.shape}")
                assert (
                    v.shape == env.observation_space.spaces[k].shape
                ), f"{v.shape} != {env.observation_space.spaces[k].shape}"
            assert ob.keys() == env.observation_space.spaces.keys()
        else:
            print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
            assert env.observation_space.shape == ob.shape
        print(f"ac_space = {env.action_space.shape}")

        # Run environment loop
        ret = 0
        while True:
            action = env.action_space.sample()
            robust_input = {
                "action": action,
                "robust_type": "action",
                "robust_config": robust_args,
            }
            ob, rew, terminated, truncated, info = env.step(robust_input)
            img = env.render()
            ret += rew

            if args.render_mode == "rgb_array":
                cv2.imshow("test_env", img[:, :, ::-1])
                cv2.waitKey(1)

            if terminated or truncated:
                ret = 0
                env.reset()
        env.close()


.. `Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

.. `Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__