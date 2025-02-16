import gymnasium as gym

# Limx Dynamic Pointfoot
from leggedrobotslab.tasks.locomotion.config.limx_pf.agents.rsl_rl_ppo_cfg import PointFootPPORunnerCfg
from leggedrobotslab.tasks.locomotion.config.limx_pf.agents.rsl_rl_ppo_mlp_cfg import PFRoughPPORunnerEncCfg
from . import pointfoot_env_cfg

# Unitree GO1
from leggedrobotslab.tasks.locomotion.config.unitree_go1.agents.rsl_rl_ppo_cfg import GO1PPORunnerCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.agents.rsl_rl_ppo_mlp_cfg import GO1RoughPPORunnerEncCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.agents.rsl_rl_ppo_cfg_vision import Go1VisionRoughPPORunnerCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.agents.rsl_rl_ppo_cfg_handstand import GO1HandStandPPORunnerCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.agents.rsl_rl_ppo_cfg_gait import GO1GaitPPORunnerCfg
from . import go1_env_cfg

##
# Create PPO runners for RSL-RL
##

# Unitree GO1
unitreego1_blind_rough_runner_cfg_v0 = GO1PPORunnerCfg()
unitreego1_blind_rough_runner_cfg_v0.experiment_name = "unitreego1_blind_rough_v0"
unitreego1_blind_rough_runner_cfg_v0.run_name = "v1_rsl_rl_trot"
unitreego1_blind_rough_runner_cfg_v0.max_iterations = 3001

unitreego1_blind_rough_runner_cfg_v1 = GO1RoughPPORunnerEncCfg()
unitreego1_blind_rough_runner_cfg_v1.experiment_name = "unitreego1_blind_rough_v1"
unitreego1_blind_rough_runner_cfg_v1.run_name = "v0"
unitreego1_blind_rough_runner_cfg_v1.max_iterations = 5001

unitreego1_vision_rough_runner_cfg_v0 = Go1VisionRoughPPORunnerCfg()
unitreego1_vision_rough_runner_cfg_v0.experiment_name = "unitreego1_vision_rough_v0"
unitreego1_vision_rough_runner_cfg_v0.run_name = "v0"
unitreego1_vision_rough_runner_cfg_v0.max_iterations = 5001

unitreego1_handstand_rough_runner_cfg_v0 = GO1HandStandPPORunnerCfg()
unitreego1_handstand_rough_runner_cfg_v0.experiment_name = "unitreego1_handstand_rough_v0"
unitreego1_handstand_rough_runner_cfg_v0.run_name = "v0"
unitreego1_handstand_rough_runner_cfg_v0.max_iterations = 3001

unitreego1_blind_rough_gait_runner_cfg_v0 = GO1GaitPPORunnerCfg()
unitreego1_blind_rough_gait_runner_cfg_v0.experiment_name = "unitreego1_blind_rough_gait_v0"
unitreego1_blind_rough_gait_runner_cfg_v0.run_name = "v1"
unitreego1_blind_rough_gait_runner_cfg_v0.max_iterations = 3001

# Limx Dynamic Pointfoot
pf_blind_rough_runner_cfg_v0 = PointFootPPORunnerCfg()
pf_blind_rough_runner_cfg_v0.experiment_name = "pf_blind_rough_v0"
pf_blind_rough_runner_cfg_v0.run_name = "v0"
pf_blind_rough_runner_cfg_v0.max_iterations = 5001

pf_blind_rough_runner_cfg_v1 = PFRoughPPORunnerEncCfg()
pf_blind_rough_runner_cfg_v1.experiment_name = "pf_blind_rough_v1"
pf_blind_rough_runner_cfg_v1.run_name = "v0"
pf_blind_rough_runner_cfg_v1.max_iterations = 5001


##
# Register Gym environments
##


#############################
# UnitreeGO1 Blind Rough Environment v0
#############################

gym.register(
    id="Isaac-UnitreeGO1-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1BlindRoughEnvCfg_v0,
        "rsl_rl_cfg_entry_point": unitreego1_blind_rough_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-UnitreeGO1-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1BlindRoughEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": unitreego1_blind_rough_runner_cfg_v0,
    },
)


#############################
# UnitreeGO1 Blind Rough Environment zgh v1
#############################

gym.register(
    id="Isaac-UnitreeGO1-Blind-Rough-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1BlindRoughEnvCfg_v1,
        "rsl_rl_cfg_entry_point": unitreego1_blind_rough_runner_cfg_v1,
    },
)

gym.register(
    id="Isaac-UnitreeGO1-Blind-Rough-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1BlindRoughEnvCfg_PLAY_v1,
        "rsl_rl_cfg_entry_point": unitreego1_blind_rough_runner_cfg_v1,
    },
)


#############################
# UnitreeGO1 Vision Rough Environment v0
#############################

gym.register(
    id="Isaac-UnitreeGO1-Vision-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1VisionRoughEnvCfg_v0,
        "rsl_rl_cfg_entry_point": unitreego1_vision_rough_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-UnitreeGO1-Vision-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1VisionRoughEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": unitreego1_vision_rough_runner_cfg_v0,
    },
)


#############################
# UnitreeGO1 HandStand Rough Environment v0
#############################

gym.register(
    id="Isaac-UnitreeGO1-HandStand-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1HandStandRoughEnvCfg_v0,
        "rsl_rl_cfg_entry_point": unitreego1_handstand_rough_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-UnitreeGO1-HandStand-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1HandStandRoughEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": unitreego1_handstand_rough_runner_cfg_v0,
    },
)


#############################
# UnitreeGO1 Blind Rough Gait Environment v0
#############################

gym.register(
    id="Isaac-UnitreeGO1-Blind-Rough-Gait-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1BlindRoughGaitEnvCfg_v0,
        "rsl_rl_cfg_entry_point": unitreego1_blind_rough_gait_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-UnitreeGO1-Blind-Rough-Gait-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.UnitreeGo1BlindRoughGaitEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": unitreego1_blind_rough_gait_runner_cfg_v0,
    },
)


#############################
# PF Blind Rough Environment v0
#############################

gym.register(
    id="Isaac-PF-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg_v0,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-PF-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg_v0,
    },
)


#############################
# PF Blind Rough Environment zgh v1
#############################

gym.register(
    id="Isaac-PF-Blind-Rough-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg_v1,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg_v1,
    },
)

gym.register(
    id="Isaac-PF-Blind-Rough-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg_PLAY_v1,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg_v1,
    },
)