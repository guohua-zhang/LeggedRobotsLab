from omni.isaac.lab.utils import configclass

from leggedrobotslab.assets.config.unitree import G1_MINIMAL_CFG
from leggedrobotslab.tasks.locomotion.config.unitree_g1.g1_env_cfg import G1EnvCfg

############################
# UnitreeGO1 Blind Rough Environment v0
############################


@configclass
class UnitreeG1BlindRoughEnvCfg_v0(G1EnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class UnitreeG1BlindRoughEnvCfg_PLAY_v0(UnitreeG1BlindRoughEnvCfg_v0):
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
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.commands.base_velocity.heading_command = False
        # self.commands.base_velocity.rel_heading_envs = 0.0