import math
from dataclasses import MISSING

from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg, DistantLightCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg as BaseCommandsCfg

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains import TerrainGeneratorCfg

from leggedrobotslab.tasks.locomotion import mdp


##
# Scene Definition
##


ROUGH_TERRAINS_CFG_v1 = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(proportion=0.1, amplitude_range=(0.02, 0.1), num_waves=10, border_width=0.25),
    },
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # height sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )



##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # gait_command = mdp.UniformGaitCommandCfgQuad(
    #     resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
    #     debug_vis=False,  # No debug visualization needed
    #     ranges=mdp.UniformGaitCommandCfgQuad.Ranges(
    #         frequencies=(1.5, 2.5),  # Gait frequency range [Hz]
    #         durations=(0.35, 0.35),  # Contact duration range [0-1]
    #         offsets2=(0.5, 0.5),  # Phase offset2 range [0-1]
    #         offsets3=(0.5, 0.5),  # Phase offset3 range [0-1]
    #         offsets4=(0.0, 0.0),  # Phase offset4 range [0-1]
    #     ),
    # )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        # Commmands (3)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) 

        # base_lin_vel (3)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoise(n_min=-0.1, n_max=0.1))
        # base_ang_vel w (3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(n_min=-0.2, n_max=0.2)) 
        # projected_gravity (3)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoise(n_min=-0.05, n_max=0.05),
        )
        # Joint positions (12)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoise(n_min=-0.01, n_max=0.01)) 
       
        # Joint velocities (12)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=UniformNoise(n_min=-1.5, n_max=1.5)) 
        # last action (12)
        #['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        actions = ObsTerm(func=mdp.last_action) 

        # gaits
        # gait_phase = ObsTerm(func=mdp.get_gait_phase)
        # gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # height_scan
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=UniformNoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            # 带噪声的观测
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        # observation terms (order preserved)
        # Commmands (3)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 角速度 w (3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(n_min=-0.2, n_max=0.2))
        # 重力 (3)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoise(n_min=-0.05, n_max=0.05),
        )
        # Joint positions (12)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoise(n_min=-0.01, n_max=0.01))
        # Joint velocities (12)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=UniformNoise(n_min=-1.5, n_max=1.5))
        # last action (3)
        actions = ObsTerm(func=mdp.last_action)

        # gaits
        # gait_phase = ObsTerm(func=mdp.get_gait_phase)
        # gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # 线速度 (3)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoise(n_min=-0.1, n_max=0.1))
        # 高度 (1)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=UniformNoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            # 带噪声的观测
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True


    # @configclass
    # class before_step_CriticCfg(ObsGroup):
    #     """Observations for critic group."""
    #     # observation terms (order preserved)
    #     # Commmands (3)
    #     velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    #     # 角速度 w (3)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(n_min=-0.2, n_max=0.2))
    #     # 重力 (3)
    #     projected_gravity = ObsTerm(
    #         func=mdp.projected_gravity,
    #         noise=UniformNoise(n_min=-0.05, n_max=0.05),
    #     )
    #     # Joint positions (12)
    #     joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoise(n_min=-0.01, n_max=0.01))
    #     # Joint velocities (12)
    #     joint_vel = ObsTerm(func=mdp.joint_vel, noise=UniformNoise(n_min=-1.5, n_max=1.5))
    #     # last action (3)
    #     actions = ObsTerm(func=mdp.last_action)
    #     # 线速度 (3)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoise(n_min=-0.1, n_max=0.1))
    #     # 高度 (1)
    #     height_scan = ObsTerm(
    #         func=mdp.height_scan,
    #         params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    #         noise=UniformNoise(n_min=-0.1, n_max=0.1),
    #         clip=(-1.0, 1.0),
    #     )

    #     def __post_init__(self):
    #         # 带噪声的观测
    #         self.enable_corruption = False
    #         self.concatenate_terms = False
    #         self.history_length = 1
    #         self.flatten_history_dim = False
            

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    # before_step_Critic: before_step_CriticCfg = before_step_CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup 
    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 4.5),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (16, 24),
            "damping_distribution_params": (0.4, 0.6),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # randomize_motor_strength_scale = EventTerm(
    #     function=mdp.randomize_motor_strength_scale,
    #     mode="startup",
    #     params={
    #         "strength_scale": (0.8, 1.2)",
    #         "operation": "scale",
    #         "distribution": "uniform",},
    # )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),   
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (-0.5, 0.5),
                # "roll": (-0.5, 0.5),
                # "pitch": (-0.5, 0.5),
                # "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5), # 0.8 1.2
            "velocity_range": (0.0, 0.0),
        },
    )

    # randomize_joint_parameters = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "friction_distribution_params": (0.75, 1.25),
    #         "armature_distribution_params": (0.75, 1.25),
    #         "lower_limit_distribution_params": (0.75, 1.25),
    #         "upper_limit_distribution_params": (0.75, 1.25),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --------------- rewards ---------------
    rew_track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # rew_feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.01,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.7,
    #     },
    # )

    # --------------- penalties ---------------
    pen_lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    pen_ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    pen_joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    pen_joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    pen_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    # pen_joint_vel_limits = RewTerm(
    #     func=mdp.joint_vel_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    # )
    pen_joint_power = RewTerm(func=mdp.joint_powers_l1, weight=-2.0e-5)
    # pen_joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    # )

    pen_action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    pen_flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    pen_base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.35,
        },
    )
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )
    # pen_stand_still_when_zero_command = RewTerm(
    #     func=mdp.stand_still_when_zero_command,
    #     weight=-0.5,
    #     params={"command_name": "base_velocity"},
    # )

    pen_feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=-0.01,
        params={
            "asset_feet_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_feet_height": -0.2,
        },
    )
    # # Gait reward
    # pen_gait_reward = RewTerm(
    #     func=mdp.GaitRewardQuad,
    #     weight=1.0,
    #     params={
    #         "tracking_contacts_shaped_force": -1.0,
    #         "tracking_contacts_shaped_vel": -1.0,
    #         "gait_force_sigma": 25.0,
    #         "gait_vel_sigma": 0.25,
    #         "kappa_gait_probs": 0.05,
    #         "command_name": "gait_command",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##
@configclass
class Go1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
