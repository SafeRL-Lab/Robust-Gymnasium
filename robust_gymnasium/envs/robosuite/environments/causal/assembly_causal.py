import random
from collections import OrderedDict

import numpy as np

import robust_gymnasium.envs.robosuite.utils.transform_utils as T
from robust_gymnasium.envs.robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robust_gymnasium.envs.robosuite.models.arenas import PegsArena
from robust_gymnasium.envs.robosuite.models.objects import RoundNutObject, SquareNutObject
from robust_gymnasium.envs.robosuite.models.tasks import ManipulationTask
from robust_gymnasium.envs.robosuite.utils.observables import Observable, sensor
from robust_gymnasium.envs.robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class AssemblyCausal(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=2,
        nut_type="round",
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
        spurious_type='diag',
    ):  
        # spurious type
        assert spurious_type in ['diag', 'vert'], "spurious_type must be either 'diag' or 'full'"
        self.spurious_type = spurious_type

        # task settings
        self.single_object_mode = single_object_mode
        self.nut_to_id = {"square": 0, "round": 1}
        self.nut_id_to_sensors = {}  # Maps nut id to sensor names for that nut
        if nut_type is not None:
            assert nut_type in self.nut_to_id.keys(), "invalid @nut_type argument - choose one of {}".format(list(self.nut_to_id.keys()))
            self.nut_id = self.nut_to_id[nut_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.82))

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

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

          - a discrete reward of 1.0 per nut if it is placed around its correct peg

        Un-normalized components if using reward shaping, where the maximum is returned if not solved:

          - Reaching: in [0, 0.1], proportional to the distance between the gripper and the closest nut
          - Grasping: in {0, 0.35}, nonzero if the gripper is grasping a nut
          - Lifting: in {0, [0.35, 0.5]}, nonzero only if nut is grasped; proportional to lifting height
          - Hovering: in {0, [0.5, 0.7]}, nonzero only if nut is lifted; proportional to distance from nut to peg

        Note that a successfully completed task (nut around peg) will return 1.0 per nut irregardless of whether the
        environment is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 2.0 (or 1.0 if only a single nut is
        being used) as well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_on_pegs)

        # add in shaped rewards
        if self.reward_shaping:
            r_reach, r_grasp, r_lift, r_hover = self.staged_rewards()
            reward += max(r_reach, r_grasp, r_lift, r_hover)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= 2.0
        return reward

    def staged_rewards(self):
        """
        Calculates staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already on the correct pegs
        active_nuts = []
        for i, nut in enumerate(self.nuts):
            if self.objects_on_pegs[i]:
                continue
            active_nuts.append(nut)

        # reaching reward governed by distance to closest object
        r_reach = 0.0
        if active_nuts:
            # reaching reward via minimum distance to the handles of the objects
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_nut.important_sites["handle"],
                    target_type="site",
                    return_distance=True,
                )
                for active_nut in active_nuts
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[g for active_nut in active_nuts for g in active_nut.contact_geoms],
                )
            ) * grasp_mult
        )

        # lifting reward for picking up an object
        r_lift = 0.0
        table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
        if active_nuts and r_grasp > 0.0:
            z_target = table_pos[2] + 0.2
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_nut.name] for active_nut in active_nuts]][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        # hover reward for getting object above peg
        r_hover = 0.0
        if active_nuts:
            r_hovers = np.zeros(len(active_nuts))
            peg_body_ids = [self.peg1_body_id, self.peg2_body_id]
            for i, nut in enumerate(active_nuts):
                valid_obj = False
                peg_pos = None
                for nut_name, idn in self.nut_to_id.items():
                    if nut_name in nut.name.lower():
                        peg_pos = np.array(self.sim.data.body_xpos[peg_body_ids[idn]])[:2]
                        valid_obj = True
                        break
                if not valid_obj:
                    raise Exception("Got invalid object to reach: {}".format(nut.name))
                ob_xy = self.sim.data.body_xpos[self.obj_body_id[nut.name]][:2]
                dist = np.linalg.norm(peg_pos - ob_xy)
                r_hovers[i] = r_lift + (1 - np.tanh(10.0 * dist)) * (hover_mult - lift_mult)
            r_hover = np.max(r_hovers)

        return r_reach, r_grasp, r_lift, r_hover

    def on_peg(self, obj_pos, peg_id):
        if peg_id == 0:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
        else:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
        res = False
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < self.table_offset[2] + 0.05
        ):
            res = True
        return res

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = PegsArena(
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

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # Create default (SequentialCompositeSampler) sampler if it has not already been specified
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for nut_name, default_y_range in zip(nut_names, ([0.11, 0.225], [-0.225, -0.11])):
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{nut_name}Sampler",
                        x_range=[-0.115, -0.11],
                        y_range=default_y_range,
                        rotation=None,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                    )
                )
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(zip((SquareNutObject, RoundNutObject), nut_names)):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        self.table_body_id = self.sim.model.body_name2id("table")
        self.peg1_body_id = self.sim.model.body_name2id("peg1")
        self.peg2_body_id = self.sim.model.body_name2id("peg2")

        for nut in self.nuts:
            self.obj_body_id[nut.name] = self.sim.model.body_name2id(nut.root_body)
            self.obj_geom_id[nut.name] = [self.sim.model.geom_name2id(g) for g in nut.contact_geoms]

        # information of objects
        self.object_site_ids = [self.sim.model.site_name2id(nut.important_sites["handle"]) for nut in self.nuts]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros(len(self.nuts))

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

            # Reset nut sensor mappings
            self.nut_id_to_sensors = {}

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return (
                    T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"])))
                    if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                    else np.eye(4)
                )

            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            enableds = [True]
            actives = [False]

            # Define nut related sensors
            for i, nut in enumerate(self.nuts):
                # Create sensors for this nut
                using_nut = self.single_object_mode == 0 or self.nut_id == i
                nut_sensors, nut_sensor_names = self._create_nut_sensors(nut_name=nut.name, modality=modality)
                sensors += nut_sensors
                names += nut_sensor_names
                enableds += [using_nut] * 4
                actives += [using_nut] * 4
                self.nut_id_to_sensors[i] = nut_sensor_names

            if self.single_object_mode == 1:
                # This is randomly sampled object, so we need to include object id as observation
                @sensor(modality=modality)
                def nut_id(obs_cache):
                    return self.nut_id

                sensors.append(nut_id)
                names.append("nut_id")
                enableds.append(True)
                actives.append(True)

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )

        return observables

    def _create_nut_sensors(self, nut_name, modality="object"):
        """
        Helper function to create sensors for a given nut. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            nut_name (str): Name of nut to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given nut
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def nut_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[nut_name]])

        @sensor(modality=modality)
        def nut_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[nut_name]], to="xyzw")

        @sensor(modality=modality)
        def nut_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [name not in obs_cache for name in [f"{nut_name}_pos", f"{nut_name}_quat", "world_pose_in_gripper"]]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{nut_name}_pos"], obs_cache[f"{nut_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{nut_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def nut_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{nut_name}_to_{pf}eef_quat"] if f"{nut_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)
            )

        sensors = [nut_pos, nut_quat, nut_to_eef_pos, nut_to_eef_quat]
        names = [f"{nut_name}_pos", f"{nut_name}_quat", f"{nut_name}_to_{pf}eef_pos", f"{nut_name}_to_{pf}eef_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                print(obj)
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _check_success(self):
        """
        Check if all nuts have been successfully placed around their corresponding pegs.

        Returns:
            bool: True if all nuts are placed correctly
        """
        # remember objects that are on the correct pegs
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        for i, nut in enumerate(self.nuts):
            obj_str = nut.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_on_pegs[i] = int(self.on_peg(obj_pos, i) and r_reach < 0.6)

        if self.single_object_mode > 0:
            return np.sum(self.objects_on_pegs) > 0  # need one object on peg

        # returns True if all objects are on correct pegs
        return np.sum(self.objects_on_pegs) == len(self.nuts)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest nut.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest nut
        if vis_settings["grippers"]:
            # find closest object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=nut.important_sites["handle"],
                    target_type="site",
                    return_distance=True,
                )
                for nut in self.nuts
            ]
            closest_nut_id = np.argmin(dists)
            # Visualize the distance to this target
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.nuts[closest_nut_id].important_sites["handle"],
                target_type="site",
            )
