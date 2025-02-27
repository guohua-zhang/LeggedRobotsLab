from omni.isaac.lab.utils import configclass

from leggedrobotslab.assets.config.unitree import UNITREE_GO1_CFG
from leggedrobotslab.tasks.locomotion.config.unitree_go1.go1_env_cfg import Go1EnvCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.go1_env_cfg_vision import Go1VisionEnvCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.go1_env_cfg_handstand import Go1HandStandEnvCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.go1_env_cfg_gait import Go1GaitEnvCfg

############################
# UnitreeGO1 Blind Rough Environment v0
############################


@configclass
class UnitreeGo1BlindRoughEnvCfg_v0(Go1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


@configclass
class UnitreeGo1BlindRoughEnvCfg_PLAY_v0(UnitreeGo1BlindRoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None


############################
# UnitreeGO1 Blind Rough Environment v1
############################


@configclass
class UnitreeGo1BlindRoughEnvCfg_v1(Go1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


@configclass
class UnitreeGo1BlindRoughEnvCfg_PLAY_v1(UnitreeGo1BlindRoughEnvCfg_v1):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None


############################
# UnitreeGO1 Vision Rough Environment v0
############################


@configclass
class UnitreeGo1VisionRoughEnvCfg_v0(Go1VisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"


@configclass
class UnitreeGo1VisionRoughEnvCfg_PLAY_v0(UnitreeGo1VisionRoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None

        # self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.commands.base_velocity.heading_command = False
        # self.commands.base_velocity.rel_heading_envs = 0.0


############################
# UnitreeGO1 HandStand Rough Environment v0
############################


@configclass
class UnitreeGo1HandStandRoughEnvCfg_v0(Go1HandStandEnvCfg):

    foot_link_name = ".*_foot"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # HandStand
        handstand_type = "back"  # which leg on air, can be "front", "back", "left", "right"
        if handstand_type == "front":
            air_foot_name = "F.*_foot"
            self.rewards.pen_handstand_orientation_l2.weight = -1.0
            self.rewards.pen_handstand_orientation_l2.params["target_gravity"] = [-1.0, 0.0, 0.0]
            self.rewards.pen_handstand_feet_height_exp.params["target_height"] = 0.5
        elif handstand_type == "back":
            air_foot_name = "R.*_foot"
            self.rewards.pen_handstand_orientation_l2.weight = -1.0
            self.rewards.pen_handstand_orientation_l2.params["target_gravity"] = [1.0, 0.0, 0.0]
            self.rewards.pen_handstand_feet_height_exp.params["target_height"] = 0.5
        elif handstand_type == "left":
            air_foot_name = ".*L_foot"
            self.rewards.pen_handstand_orientation_l2.weight = 0
            self.rewards.pen_handstand_orientation_l2.params["target_gravity"] = [0.0, -1.0, 0.0]
            self.rewards.pen_handstand_feet_height_exp.params["target_height"] = 0.3
        elif handstand_type == "right":
            air_foot_name = ".*R_foot"
            self.rewards.pen_handstand_orientation_l2.weight = 0
            self.rewards.pen_handstand_orientation_l2.params["target_gravity"] = [0.0, 1.0, 0.0]
            self.rewards.pen_handstand_feet_height_exp.params["target_height"] = 0.3
        self.rewards.pen_handstand_feet_height_exp.weight = 10
        self.rewards.pen_handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_name]
        self.rewards.pen_handstand_feet_on_air.weight = 1.0
        self.rewards.pen_handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.pen_handstand_feet_air_time.weight = 1.0
        self.rewards.pen_handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_name]

        self.terminations.base_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]


@configclass
class UnitreeGo1HandStandRoughEnvCfg_PLAY_v0(UnitreeGo1HandStandRoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0


############################
# UnitreeGO1 Blind Rough Gait Environment v0
############################


@configclass
class UnitreeGo1BlindRoughGaitEnvCfg_v0(Go1GaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


@configclass
class UnitreeGo1BlindRoughGaitEnvCfg_PLAY_v0(UnitreeGo1BlindRoughGaitEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None