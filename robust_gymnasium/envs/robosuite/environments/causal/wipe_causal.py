import multiprocessing
from collections import OrderedDict

import numpy as np

from robust_gymnasium.envs.robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robust_gymnasium.envs.robosuite.models.arenas import WipeArena
from robust_gymnasium.envs.robosuite.models.tasks import ManipulationTask
from robust_gymnasium.envs.robosuite.models.objects import BoxObject
from robust_gymnasium.envs.robosuite.utils.observables import Observable, sensor


# Default Wipe environment configuration
DEFAULT_WIPE_CONFIG = {
    # settings for reward
    "arm_limit_collision_penalty": -10.0,     # penalty for reaching joint limit or arm collision (except the wiping tool) with the table
    "wipe_contact_reward": 0.01,              # reward for contacting something with the wiping tool
    "unit_wiped_reward": 50.0,                # reward per peg wiped
    "ee_accel_penalty": 0,                    # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.05,         # penalty for each step that the force is over the safety threshold
    "distance_multiplier": 5.0,               # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
    "distance_th_multiplier": 5.0,            # multiplier in the tanh function for the aforementioned reward
    # settings for table top
    "table_full_size": [0.5, 0.8, 0.05],      # Size of tabletop
    "table_offset": [0.15, 0, 0.9],           # Offset of table (z dimension defines max height of table)
    "table_friction": [0.03, 0.005, 0.0001],  # Friction parameters for the table
    "table_friction_std": 0,                  # Standard deviation to sample different friction parameters for the table each episode
    "table_height": 0.0,                      # Additional height of the table over the default location
    "table_height_std": 0.0,                  # Standard deviation to sample different heigths of the table each episode
    "line_width": 0.04,                       # Width of the line to wipe (diameter of the pegs)
    "two_clusters": False,                    # if the dirt to wipe is one continuous line or two
    "coverage_factor": 0.6,                   # how much of the table surface we cover
    "num_markers": 20,                        # How many particles of dirt to generate in the environment
    # settings for thresholds
    "contact_threshold": 1.0,                 # Minimum eef force to qualify as contact [N]
    "pressure_threshold": 0.5,                # force threshold (N) to overcome to get increased contact wiping reward
    "pressure_threshold_max": 60.0,           # maximum force allowed (N)
    # misc settings
    "print_results": False,                   # Whether to print results or not
    "get_info": False,                        # Whether to grab info after each env step if not
    "use_robot_obs": True,                    # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,                  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": True,               # Whether we allow for early terminations or not
    "use_condensed_obj_obs": False,           # Whether to use condensed object observation representation (only applicable if obj obs is active)
}


class WipeCausal(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="WipingGripper",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
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
        task_config=None,
        renderer="mujoco",
        renderer_config=None,
        spurious_type='none',
    ):
        # Assert that the gripper type is None
        assert (
            gripper_types == "WipingGripper"
        ), "Tried to specify gripper other than WipingGripper in Wipe environment!"

        # Get config
        self.task_config = task_config if task_config is not None else DEFAULT_WIPE_CONFIG

        # Set spurious type
        assert spurious_type in ['ll', 'lr']
        self.spurious_type = spurious_type

        # Set task-specific parameters

        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_limit_collision_penalty = self.task_config["arm_limit_collision_penalty"]
        self.wipe_contact_reward = self.task_config["wipe_contact_reward"]
        self.unit_wiped_reward = self.task_config["unit_wiped_reward"]
        self.ee_accel_penalty = self.task_config["ee_accel_penalty"]
        self.excess_force_penalty_mul = self.task_config["excess_force_penalty_mul"]
        self.distance_multiplier = self.task_config["distance_multiplier"]
        self.distance_th_multiplier = self.task_config["distance_th_multiplier"]
        # Final reward computation
        # So that is better to finish that to stay touching the table for 100 steps
        # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well
        self.task_complete_reward = self.unit_wiped_reward * (self.wipe_contact_reward + 0.5)
        # Verify that the distance multiplier is not greater than the task complete reward
        assert (
            self.task_complete_reward > self.distance_multiplier
        ), "Distance multiplier cannot be greater than task complete reward!"

        # settings for table top
        self.table_full_size = self.task_config["table_full_size"]
        self.table_height = self.task_config["table_height"]
        self.table_height_std = self.task_config["table_height_std"]
        delta_height = min(0, np.random.normal(self.table_height, self.table_height_std))  # sample variation in height
        self.table_offset = np.array(self.task_config["table_offset"]) + np.array((0, 0, delta_height))
        self.table_friction = self.task_config["table_friction"]
        self.table_friction_std = self.task_config["table_friction_std"]
        self.line_width = self.task_config["line_width"]
        self.two_clusters = self.task_config["two_clusters"]
        self.coverage_factor = self.task_config["coverage_factor"]
        self.num_markers = self.task_config["num_markers"]

        # settings for thresholds
        self.contact_threshold = self.task_config["contact_threshold"]
        self.pressure_threshold = self.task_config["pressure_threshold"]
        self.pressure_threshold_max = self.task_config["pressure_threshold_max"]

        # misc settings
        self.print_results = self.task_config["print_results"]
        self.get_info = self.task_config["get_info"]
        self.use_robot_obs = self.task_config["use_robot_obs"]
        self.use_contact_obs = self.task_config["use_contact_obs"]
        self.early_terminations = self.task_config["early_terminations"]
        self.use_condensed_obj_obs = self.task_config["use_condensed_obj_obs"]

        # Scale reward if desired (see reward method for details)
        self.reward_normalization_factor = horizon / (
            self.num_markers * self.unit_wiped_reward + horizon * (self.wipe_contact_reward + self.task_complete_reward)
        )

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # set other wipe-specific attributes
        self.wiped_markers = []
        self.collisions = 0
        self.f_excess = 0
        self.metadata = []
        self.spec = "spec"

        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

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

            - a discrete reward of self.unit_wiped_reward is provided per single dirt (peg) wiped during this step
            - a discrete reward of self.task_complete_reward is provided if all dirt is wiped

        Note that if the arm is either colliding or near its joint limit, a reward of 0 will be automatically given

        Un-normalized summed components if using reward shaping (individual components can be set to 0:

            - Reaching: in [0, self.distance_multiplier], proportional to distance between wiper and centroid of dirt
              and zero if the table has been fully wiped clean of all the dirt
            - Table Contact: in {0, self.wipe_contact_reward}, non-zero if wiper is in contact with table
            - Wiping: in {0, self.unit_wiped_reward}, non-zero for each dirt (peg) wiped during this step
            - Cleaned: in {0, self.task_complete_reward}, non-zero if no dirt remains on the table
            - Collision / Joint Limit Penalty: in {self.arm_limit_collision_penalty, 0}, nonzero if robot arm
              is colliding with an object
              - Note that if this value is nonzero, no other reward components can be added
            - Large Force Penalty: in [-inf, 0], scaled by wiper force and directly proportional to
              self.excess_force_penalty_mul if the current force exceeds self.pressure_threshold_max
            - Large Acceleration Penalty: in [-inf, 0], scaled by estimated wiper acceleration and directly
              proportional to self.ee_accel_penalty

        Note that the final per-step reward is normalized given the theoretical best episode return and then scaled:
        reward_scale * (horizon /
        (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        total_force_ee = np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3]))

        # Neg Reward from collisions of the arm with the table
        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        else:
            # If the arm is not colliding or in joint limits, we check if we are wiping
            # (we don't want to reward wiping if there are unsafe situations)
            active_markers = []

            # Current 3D location of the corners of the wiping tool in world frame
            c_geoms = self.robots[0].gripper.important_geoms["corners"]
            corner1_id = self.sim.model.geom_name2id(c_geoms[0])
            corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
            corner2_id = self.sim.model.geom_name2id(c_geoms[1])
            corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
            corner3_id = self.sim.model.geom_name2id(c_geoms[2])
            corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
            corner4_id = self.sim.model.geom_name2id(c_geoms[3])
            corner4_pos = np.array(self.sim.data.geom_xpos[corner4_id])

            # Unit vectors on my plane
            v1 = corner1_pos - corner2_pos
            v1 /= np.linalg.norm(v1)
            v2 = corner4_pos - corner2_pos
            v2 /= np.linalg.norm(v2)

            # Corners of the tool in the coordinate frame of the plane
            t1 = np.array([np.dot(corner1_pos - corner2_pos, v1), np.dot(corner1_pos - corner2_pos, v2)])
            t2 = np.array([np.dot(corner2_pos - corner2_pos, v1), np.dot(corner2_pos - corner2_pos, v2)])
            t3 = np.array([np.dot(corner3_pos - corner2_pos, v1), np.dot(corner3_pos - corner2_pos, v2)])
            t4 = np.array([np.dot(corner4_pos - corner2_pos, v1), np.dot(corner4_pos - corner2_pos, v2)])

            pp = [t1, t2, t4, t3]

            # Normal of the plane defined by v1 and v2
            n = np.cross(v1, v2)
            n /= np.linalg.norm(n)

            def isLeft(P0, P1, P2):
                return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

            def PointInRectangle(X, Y, Z, W, P):
                return isLeft(X, Y, P) < 0 and isLeft(Y, Z, P) < 0 and isLeft(Z, W, P) < 0 and isLeft(W, X, P) < 0

            # Only go into this computation if there are contact points
            if self.sim.data.ncon != 0:
                # Check each marker that is still active
                for marker in self.model.mujoco_arena.markers:
                    # Current marker 3D location in world frame
                    marker_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])

                    # We use the second tool corner as point on the plane and define the vector connecting
                    # the marker position to that point
                    v = marker_pos - corner2_pos

                    # Shortest distance between the center of the marker and the plane
                    dist = np.dot(v, n)

                    # Projection of the center of the marker onto the plane
                    projected_point = np.array(marker_pos) - dist * n

                    # Positive distances means the center of the marker is over the plane
                    # The plane is aligned with the bottom of the wiper and pointing up, so the marker would be over it
                    if dist > 0.0:
                        # Distance smaller than this threshold means we are close to the plane on the upper part
                        if dist < 0.02:
                            # Write touching points and projected point in coordinates of the plane
                            pp_2 = np.array(
                                [np.dot(projected_point - corner2_pos, v1), np.dot(projected_point - corner2_pos, v2)]
                            )
                            # Check if marker is within the tool center:
                            if PointInRectangle(pp[0], pp[1], pp[2], pp[3], pp_2):
                                active_markers.append(marker)

            # Obtain the list of currently active (wiped) markers that where not wiped before
            # These are the markers we are wiping at this step
            lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
            new_active_markers = np.array(active_markers)[lall]

            # Loop through all new markers we are wiping at this step
            for new_active_marker in new_active_markers:
                # Grab relevant marker id info
                new_active_marker_geom_id = self.sim.model.geom_name2id(new_active_marker.visual_geoms[0])
                # Make this marker transparent since we wiped it (alpha = 0)
                self.sim.model.geom_rgba[new_active_marker_geom_id][3] = 0
                # Add this marker the wiped list
                self.wiped_markers.append(new_active_marker)
                # Add reward if we're using the dense reward
                if self.reward_shaping:
                    reward += self.unit_wiped_reward

            # Additional reward components if using dense rewards
            if self.reward_shaping:
                # If we haven't wiped all the markers yet, add a smooth reward for getting closer
                # to the centroid of the dirt to wipe
                if len(self.wiped_markers) < self.num_markers:
                    _, _, mean_pos_to_things_to_wipe = self._get_wipe_information()
                    mean_distance_to_things_to_wipe = np.linalg.norm(mean_pos_to_things_to_wipe)
                    reward += self.distance_multiplier * (1 - np.tanh(self.distance_th_multiplier * mean_distance_to_things_to_wipe))

                # Reward for keeping contact
                if self.sim.data.ncon != 0 and self._has_gripper_contact:
                    reward += self.wipe_contact_reward

                # Penalty for excessive force with the end-effector
                if total_force_ee > self.pressure_threshold_max:
                    reward -= self.excess_force_penalty_mul * total_force_ee
                    self.f_excess += 1

                # Reward for pressing into table
                # TODO: Need to include this computation somehow in the scaled reward computation
                elif total_force_ee > self.pressure_threshold and self.sim.data.ncon > 1:
                    reward += self.wipe_contact_reward + 0.01 * total_force_ee
                    if self.sim.data.ncon > 50:
                        reward += 10.0 * self.wipe_contact_reward

                # Penalize large accelerations
                reward -= self.ee_accel_penalty * np.mean(abs(self.robots[0].recent_ee_acc.current))

            # Final reward if all wiped
            if len(self.wiped_markers) == self.num_markers:
                reward += self.task_complete_reward

        # Printing results
        if self.print_results:
            string_to_print = (
                "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}"
                "wiped markers: {ws:>3} collisions: {sc:>3} f-excess: {fe:>3}".format(
                    pid=id(multiprocessing.current_process()),
                    ts=self.timestep,
                    rw=reward,
                    ws=len(self.wiped_markers),
                    sc=self.collisions,
                    fe=self.f_excess,
                )
            )
            print(string_to_print)

        # If we're scaling our reward, we normalize the per-step rewards given the theoretical best episode return
        # This is equivalent to scaling the reward by:
        #   reward_scale * (horizon /
        #       (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))
        if self.reward_scale:
            reward *= self.reward_scale * self.reward_normalization_factor
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        # initialize objects of interest
        self.cube = BoxObject(
            name="cube",
            size_min=[0.02, 0.02, 0.02], 
            size_max=[0.02, 0.02, 0.02], 
            rgba=[1, 0, 0, 1],
            material=None,
            density=10000,
        )

        mujoco_arena = WipeArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std,
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            two_clusters=self.two_clusters,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.7, 0, 1.7],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )
        
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        sensors = []
        names = []

        # Add binary contact observation
        if self.use_contact_obs:
            @sensor(modality=f"{pf}proprio")
            def gripper_contact(obs_cache):
                return self._has_gripper_contact

            sensors.append(gripper_contact)
            names.append(f"{pf}contact")

        # object information in the observation
        if self.use_object_obs:
            if self.use_condensed_obj_obs:
                # use implicit representation of wiping objects
                @sensor(modality=modality)
                def wipe_radius(obs_cache):
                    wipe_rad, wipe_cent, _ = self._get_wipe_information()
                    obs_cache["wipe_centroid"] = wipe_cent
                    return wipe_rad

                @sensor(modality=modality)
                def wipe_centroid(obs_cache):
                    return obs_cache["wipe_centroid"] if "wipe_centroid" in obs_cache else np.zeros(3)

                @sensor(modality=modality)
                def proportion_wiped(obs_cache):
                    return len(self.wiped_markers) / self.num_markers

                sensors += [proportion_wiped, wipe_radius, wipe_centroid]
                names += ["proportion_wiped", "wipe_radius", "wipe_centroid"]

                if self.use_robot_obs:
                    # also use ego-centric obs
                    @sensor(modality=modality)
                    def gripper_to_wipe_centroid(obs_cache):
                        return (
                            obs_cache["wipe_centroid"] - obs_cache[f"{pf}eef_pos"]
                            if "wipe_centroid" in obs_cache and f"{pf}eef_pos" in obs_cache
                            else np.zeros(3)
                        )

                    sensors.append(gripper_to_wipe_centroid)
                    names.append("gripper_to_wipe_centroid")
            else:
                # use explicit representation of wiping objects
                for i, marker in enumerate(self.model.mujoco_arena.markers):
                    marker_sensors, marker_sensor_names = self._create_marker_sensors(i, marker, modality)
                    sensors += marker_sensors
                    names += marker_sensor_names
            
            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            sensors += [cube_pos]
            names += [cube_pos.__name__]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_marker_sensors(self, i, marker, modality="object"):
        """
        Helper function to create sensors for a given marker. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            i (int): ID number corresponding to the marker
            marker (MujocoObject): Marker to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given marker
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def marker_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])

        @sensor(modality=modality)
        def marker_wiped(obs_cache):
            return [0, 1][marker in self.wiped_markers]

        #sensors = [marker_pos, marker_wiped]
        #names = [f"marker{i}_pos", f"marker{i}_wiped"]

        sensors = [marker_wiped]
        names = [f"marker{i}_wiped"]

        if self.use_robot_obs:
            # also use ego-centric obs
            @sensor(modality=modality)
            def gripper_to_marker(obs_cache):
                return (
                    obs_cache[f"marker{i}_pos"] - obs_cache[f"{pf}eef_pos"]
                    if f"marker{i}_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors.append(gripper_to_marker)
            names.append(f"gripper_to_marker{i}")

        return sensors, names

    def _reset_internal(self):
        super()._reset_internal()

        # random select position of cube
        cube_type = np.random.choice(['left', 'right'])
        if self.spurious_type == 'lr':   # dirt left object right
            if cube_type == 'left':
                dirt_type = 'right'
            else:
                dirt_type = 'left'
        elif self.spurious_type == 'll': # dirt left object left
            if cube_type == 'left':
                dirt_type = 'left'
            else:
                dirt_type = 'right'
        else:
            raise ValueError('Invalid spurious type')

        # inherited class should reset positions of objects (only if we're not using a deterministic reset)
        if not self.deterministic_reset:
            self.model.mujoco_arena.reset_arena_causal(self.sim, dirt_type)

        # set cube position
        if cube_type == 'right':
            obj_pos = np.array([-0.05, 0.25, 0.93])
        elif cube_type == 'left':
            obj_pos = np.array([-0.05, -0.25, 0.93])
        else:
            raise ValueError('Invalid cube type')

        obj_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos('cube_joint0', np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Reset all internal vars for this wipe task
        self.timestep = 0
        self.wiped_markers = []
        self.collisions = 0
        self.f_excess = 0

        # ee resets - bias at initial state
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

    def _check_success(self):
        """
        Checks if Task succeeds (all dirt wiped).

        Returns:
            bool: True if completed task
        """
        return True if len(self.wiped_markers) == self.num_markers else False

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision
            - Task completion (wiping succeeded)
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            if self.print_results:
                print(40 * "-" + " COLLIDED " + 40 * "-")
            terminated = True

        # Prematurely terminate if task is success
        if self._check_success():
            if self.print_results:
                print(40 * "+" + " FINISHED WIPING " + 40 * "+")
            terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self.robots[0].check_q_limits():
            if self.print_results:
                print(40 * "-" + " JOINT LIMIT " + 40 * "-")
            terminated = True

        return terminated

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        # Update force bias
        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque

        if self.get_info:
            info["add_vals"] = ["nwipedmarkers", "colls", "percent_viapoints_", "f_excess"]
            info["nwipedmarkers"] = len(self.wiped_markers)
            info["colls"] = self.collisions
            info["percent_viapoints_"] = len(self.wiped_markers) / self.num_markers
            info["f_excess"] = self.f_excess

        # allow episode to finish early if allowed
        if self.early_terminations:
            done = done or self._check_terminated()

        return reward, done, info

    def _get_wipe_information(self):
        """Returns set of wiping information"""
        mean_pos_to_things_to_wipe = np.zeros(3)
        wipe_centroid = np.zeros(3)
        marker_positions = []
        num_non_wiped_markers = 0
        if len(self.wiped_markers) < self.num_markers:
            for marker in self.model.mujoco_arena.markers:
                if marker not in self.wiped_markers:
                    marker_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])
                    wipe_centroid += marker_pos
                    marker_positions.append(marker_pos)
                    num_non_wiped_markers += 1
            wipe_centroid /= max(1, num_non_wiped_markers)
            mean_pos_to_things_to_wipe = wipe_centroid - self._eef_xpos
        # Radius of circle from centroid capturing all remaining wiping markers
        max_radius = 0
        if num_non_wiped_markers > 0:
            max_radius = np.max(np.linalg.norm(np.array(marker_positions) - wipe_centroid, axis=1))
        # Return all values
        return max_radius, wipe_centroid, mean_pos_to_things_to_wipe

    @property
    def _has_gripper_contact(self):
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold
