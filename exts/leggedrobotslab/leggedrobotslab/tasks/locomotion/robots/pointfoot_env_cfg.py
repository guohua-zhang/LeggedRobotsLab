import math

from omni.isaac.lab.utils import configclass

from leggedrobotslab.assets.config.limx import POINTFOOT_CFG
from leggedrobotslab.tasks.locomotion.config.limx_pf.pf_env_cfg import PFEnvCfg


#############################
# Pointfoot Blind Rough Environment
#############################


@configclass
class PFBlindRoughEnvCfg_v0(PFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0918,
            "hip_R_Joint": 0.0918,
            "knee_L_Joint": -0.057,
            "knee_R_Joint": -0.057,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class PFBlindRoughEnvCfg_PLAY_v0(PFBlindRoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


#############################
# Pointfoot Blind Rough Environment v1
#############################


@configclass
class PFBlindRoughEnvCfg_v1(PFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0918,
            "hip_R_Joint": 0.0918,
            "knee_L_Joint": -0.057,
            "knee_R_Joint": -0.057,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class PFBlindRoughEnvCfg_PLAY_v1(PFBlindRoughEnvCfg_v1):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None