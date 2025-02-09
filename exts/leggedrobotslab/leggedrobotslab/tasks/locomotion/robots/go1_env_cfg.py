from omni.isaac.lab.utils import configclass

from leggedrobotslab.assets.config.unitree import UNITREE_GO1_CFG
from leggedrobotslab.tasks.locomotion.config.unitree_go1.go1_env_cfg import Go1EnvCfg


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