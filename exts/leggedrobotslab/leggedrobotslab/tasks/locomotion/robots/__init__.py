import gymnasium as gym

# Limx Dynamic Pointfoot
from leggedrobotslab.tasks.locomotion.config.limx_pf.agents.rsl_rl_ppo_cfg import PointFootPPORunnerCfg
from leggedrobotslab.tasks.locomotion.config.limx_pf.agents.rsl_rl_ppo_mlp_cfg import PFRoughPPORunnerEncCfg
from . import pointfoot_env_cfg

# Unitree GO1
from leggedrobotslab.tasks.locomotion.config.unitree_go1.agents.rsl_rl_ppo_cfg import GO1PPORunnerCfg
from leggedrobotslab.tasks.locomotion.config.unitree_go1.agents.rsl_rl_ppo_mlp_cfg import GO1RoughPPORunnerEncCfg
from . import go1_env_cfg

##
# Create PPO runners for RSL-RL
##

# Unitree GO1
unitreego1_blind_rough_runner_cfg_v0 = GO1PPORunnerCfg()
unitreego1_blind_rough_runner_cfg_v0.experiment_name = "unitreego1_blind_rough_v0"
unitreego1_blind_rough_runner_cfg_v0.run_name = "v1_rsl_rl_trot"
unitreego1_blind_rough_runner_cfg_v0.max_iterations = 3001

unitreego1_blind_rough_runner_cfg_v1 = PFRoughPPORunnerEncCfg()
unitreego1_blind_rough_runner_cfg_v1.experiment_name = "unitreego1_blind_rough_v1"
unitreego1_blind_rough_runner_cfg_v1.run_name = "v0"
unitreego1_blind_rough_runner_cfg_v1.max_iterations = 5001

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