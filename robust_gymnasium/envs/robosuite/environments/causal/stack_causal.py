'''
from: https://arxiv.org/pdf/2307.07907#page=26.58
'''
from collections import OrderedDict

import numpy as np

from robust_gymnasium.envs.robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robust_gymnasium.envs.robosuite.models.arenas import TableArena
from robust_gymnasium.envs.robosuite.models.objects import BoxObject
from robust_gymnasium.envs.robosuite.models.tasks import ManipulationTask
from robust_gymnasium.envs.robosuite.utils.mjcf_utils import CustomMaterial
from robust_gymnasium.envs.robosuite.utils.observables import Observable, sensor
from robust_gymnasium.envs.robosuite.utils.placement_samplers import UniformRandomSampler
from robust_gymnasium.envs.robosuite.utils.transform_utils import convert_quat


class StackCausal(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        spurious_type='diag'
    ):
        assert spurious_type in ['diag', 'vert'], "spurious_type must be either 'diag' or 'vert'"
        self.spurious_type = spurious_type

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_stack = self.staged_rewards(action)

        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self, action):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to the center of the cube
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = 0.25 * (1 - np.tanh(10.0 * dist))

        # grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        if grasping_cubeA:
            r_reach += 0.25

        def normalize_val(reward, min_val, max_val):
            # normalize reward to [0, 1]
            reward = np.clip(reward, min_val, max_val)
            return (reward - min_val) / (max_val - min_val)

        # lifting is successful when the cube is above the table top by a margin
        cubeA_lifted = cubeA_pos[2] > self.table_offset[2] + self.cubeB_height + self.cubeA_height
        r_lift = 0.5 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        horiz_dist = 10
        vert_dist = 10
        horiz_max = 0.7
        vert_max = 0.3
        start_vert_reward = 0.2
        hover = False
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            vert_dist = np.abs(cubeA_pos[2]- cubeB_pos[2])
            #r_lift += 0.25 * (1 - np.tanh(10.0*horiz_dist))
            r_lift += 0.25 * (1 - normalize_val(horiz_dist, 0, horiz_max))

            # use verticle distance when horizontal distance is small
            if horiz_dist < start_vert_reward: # TODO: 0.1 might be too big
                #r_lift += 0.25 * (1 - np.tanh(10.0*vert_dist))
                r_lift += 0.25 * (1 - normalize_val(vert_dist, 0, vert_max))
                if vert_dist < 0.05:
                    hover = True

        # # open gripper when two objects are close
        r_stack = 0
        #print('hover', hover)
        # if not grasping_cubeA and hover:
        #     r_stack += 1.0

        # # stacking is successful when the block is lifted and the gripper is not holding the object
        # cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        # if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
        #     r_stack += 0.5

        #print(r_reach, r_lift, dist, horiz_dist, vert_dist)
        return r_reach, r_lift, r_stack

    def staged_rewards_3d_dist(self, action):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to the center of the cube
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        if grasping_cubeA:
            r_reach += 0.25

        def normalize_val(reward, min_val, max_val):
            # normalize reward to [0, 1]
            reward = np.clip(reward, min_val, max_val)
            return (reward - min_val) / (max_val - min_val)
        
        # lifting is successful when the cube is above the table top by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 0.5 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        horiz_dist = 10
        vert_dist = 10
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            vert_dist = np.abs(cubeA_pos[2] - (cubeB_pos[2] + self.cubeB_height/2))
            r_lift += 0.25 * (1 - normalize_val(horiz_dist, 0, 0.8))
            r_lift += 0.25 * (1 - normalize_val(vert_dist, 0, 0.8))

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_stack = 0
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
            r_stack += 1

        print(r_reach, r_lift, r_stack, dist, horiz_dist, vert_dist)
        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.7, 0, 1.7],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA_height = 0.02
        self.cubeB_height = 0.002
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, self.cubeA_height],
            size_max=[0.02, 0.02, self.cubeA_height],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.04, 0.04, self.cubeB_height],
            size_max=[0.04, 0.04, self.cubeB_height],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]

        # Create placement initializer
        self.cube_x_range = [-0.3, 0.1]
        self.cube_y_range = [-0.2, 0.2]
        # self.cube_x_range = [-0.08, 0.08]
        # self.cube_y_range = [-0.08, 0.08]
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=self.cube_x_range,
                y_range=self.cube_y_range,
                rotation=(np.pi, np.pi+0.001),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        #names = self.sim.model.geom_names

        def get_region(x, y, mean=(0, 0)):
            if x <= mean[0] and y <= mean[1]:
                return 1
            elif x <= mean[0] and y >= mean[1]:
                return 2
            elif x >= mean[0] and y <= mean[1]:
                return 3
            elif x >= mean[0] and y >= mean[1]:
                return 4
            else:
                raise ValueError("Invalid position")

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            while True:
                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()
                for obj_pos, obj_quat, obj in object_placements.values():
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

                cubeA_pos_x = object_placements['cubeA'][0][0]
                cubeA_pos_y = object_placements['cubeA'][0][1]
                cubeB_pos_x = object_placements['cubeB'][0][0]
                cubeB_pos_y = object_placements['cubeB'][0][1]

                # ensure that two objects are not two close to each other
                mean = ((self.cube_x_range[0] + self.cube_x_range[1]) / 2, (self.cube_y_range[0] + self.cube_y_range[1]) / 2)
                boundary = 0.15
                if np.abs(cubeA_pos_x - mean[0]) < boundary or \
                    np.abs(cubeA_pos_y - mean[1]) < boundary or \
                    np.abs(cubeB_pos_x - mean[0]) < boundary or \
                    np.abs(cubeB_pos_y - mean[1]) < boundary:
                    continue
                
                A_region = get_region(cubeA_pos_x, cubeA_pos_y, mean=mean)
                B_region = get_region(cubeB_pos_x, cubeB_pos_y, mean=mean)
                if self.spurious_type == 'diag':   # two blocks are in cross positions
                    condition = (A_region == 1 and B_region == 4) or (A_region == 2 and B_region == 3)
                elif self.spurious_type == 'vert':  # two blocks are in the same line
                    condition = (A_region == 1 and B_region == 3) or (A_region == 2 and B_region == 4) 
                else:
                    raise ValueError('unknown spuriousness type')

                if condition:
                    break

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeA_body_id])

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeB_body_id])

            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeA(obs_cache):
                return (
                    obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeB_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat, gripper_to_cubeA, gripper_to_cubeB]
            #sensors = [gripper_to_cubeA, gripper_to_cubeB, cubeA_to_cubeB]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq,)

        return observables

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)
