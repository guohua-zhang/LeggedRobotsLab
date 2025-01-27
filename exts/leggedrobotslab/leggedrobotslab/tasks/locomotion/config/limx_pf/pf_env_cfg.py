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
from omni.isaac.lab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg as BaseCommandsCfg

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains import TerrainGeneratorCfg

from leggedrobotslab.tasks.locomotion import mdp


##################
# Scene Definition
##################


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
class PFSceneCfg(InteractiveSceneCfg):
    """Configuration for the test scene"""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG_v1,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # bipedal robot
    robot: ArticulationCfg = MISSING

    # height sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # height_scanner = None

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.5, 2.5),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
        ),
    )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(operation="add", n_min=-0.2, n_max=0.2))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(operation="add", n_min=-0.05, n_max=0.05))

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoise(operation="add", n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=UniformNoise(operation="add", n_min=-1.5, n_max=1.5))

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # velocity command
        vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # Policy observation
        # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(operation="add", n_min=-0.2, n_max=0.2))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(operation="add", n_min=-0.05, n_max=0.05))

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoise(operation="add", n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=UniformNoise(operation="add", n_min=-1.5, n_max=1.5))

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # velocity command
        vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # Privileged observation
        # height measurement
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
            },
        )

        # robot_mass = ObsTerm(func=mdp.robot_mass)
        # robot_inertia = ObsTerm(func=mdp.robot_inertia)
        # robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        # robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        # robot_pos = ObsTerm(func=mdp.robot_pos)
        # robot_vel = ObsTerm(func=mdp.robot_vel)
        # robot_material_propertirs = ObsTerm(func=mdp.robot_material_properties)
        # robot_base_pose = ObsTerm(func=mdp.robot_base_pose)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2), # 0.4 1.2
            "dynamic_friction_range": (0.5, 0.8), # 0.7 0.9
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (32, 48),
            "damping_distribution_params": (2.0, 3.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.075, 0.075), (-0.05, 0.06), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi / 6, math.pi / 6)}},
    )
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque_stochastic,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
    #         "force_range": {
    #             "x": (-500.0, 500.0),
    #             "y": (-500.0, 500.0),
    #             "z": (-0.0, 0.0),
    #         },  # force = mass * dv / dt
    #         "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
    #         "probability": 0.002,  # Expect step = 1 / probability
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""
    # rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_no_fly = RewTerm(
        func=mdp.no_fly,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
            "threshold": 5.0,
        },
    )
    # This reward can be used, but it is conflicting with gait_reward. So we can use a better to make robot's feet takeoff and touch down instead.
    # rew_feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=2.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.65,
    #     },
    # )
    # pen_unbalance_feet_air_time = RewTerm(
    #     func=mdp.unbalance_feet_air_time,
    #     weight=-25.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
    #     },
    # )
    # pen_unbalance_feet_height = RewTerm(
    #     func=mdp.unbalance_feet_height,
    #     weight=-5.0,
    #     params={
    #          "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
    #     },
    # )
    # pen_feet_regulation = RewTerm(
    #     func=mdp.feet_regulation,
    #     weight=-0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "desired_body_height": 0.65,
    #     }
    # )
    # pen_feet_regulation = RewTerm(
    #     func=mdp.feet_regulation,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
    #         "desired_body_height": 0.65,
    #     }
    # )
    pen_feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=-0.5,
        params={
            "asset_feet_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
            "target_feet_height": -0.35,
        },
    )

    # penalizations
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # pen_joint_deviation_abad = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["abad_.*"])},
    # )
    pen_joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*", "knee_.*"])},
    )
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 1.0,
        },
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0) # -1.0
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-05)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-2.0e-05)
    pen_base_height = RewTerm(
        func=mdp.base_height_l2,
        params={
            "target_height": 0.65,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        weight=-1.0,
    )
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-05)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # pen_no_contact = RewTerm(
    #     func=mdp.no_contact,
    #     weight=-5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
    #     },
    # )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        }
    )
    # Gait reward
    pen_gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=2.0,
        params={
            "tracking_contacts_shaped_force": -1.0,
            "tracking_contacts_shaped_vel": -1.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )
    # pen_stand_still = RewTerm(func=mdp.stand_still, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


########################
# Environment definition
########################


@configclass
class PFEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: PFSceneCfg = PFSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization"""
            # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.seed = 66 # 42
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True # disable contact processing all the rigid bodies in the simulation; you can define needed contact processing in the scencfg such as contact_force
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
