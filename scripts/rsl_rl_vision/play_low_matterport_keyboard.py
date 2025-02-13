# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import gymnasium as gym
import os
import math
import torch
import numpy as np

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="h1_rough", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# args_cli.experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit'

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.leggedloco.config import *
from omni.isaac.lab.devices.keyboard import Se2Keyboard
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

from utils import quat2eulers


def define_markers() -> VisualizationMarkers:
    """Define path markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pathMarkers",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli, play=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    # import pdb; pdb.set_trace()
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    

    keyboard_interface = Se2Keyboard(v_x_sensitivity = 0.05, v_y_sensitivity = 0.05, omega_z_sensitivity=0.05)
    keyboard_interface.reset()
    
    # reset environment
    obs, _ = env.get_observations()
    vel_command_keyboard = torch.zeros(3, device = obs.device)
    num_count = 0

    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    cam_eye = (robot_pos_w[0]+0.0, robot_pos_w[1]+2.0, 1.0)
    cam_target = (robot_pos_w[0], robot_pos_w[1], 1.0)
    # set the camera view
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)
    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    cam_eye = (robot_pos_w[0]-0.0, robot_pos_w[1]-1.5, 1.2)
    robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
    cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
    cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])

    # set the camera view
    # env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            commands_key = torch.tensor(keyboard_interface.advance(), device=obs.device)
            vel_command_keyboard += commands_key
            if num_count % 10 == 0:
                print("vel_command_keyboard: ", vel_command_keyboard)
            # import pdb; pdb.set_trace()
            # assert torch.allclose(env.unwrapped.command_manager._terms['base_velocity'].vel_command_b, torch.tensor([0., 0., 0.], device = obs.device))
            # env.command_manager._terms['base_velocity'].vel_command_b[0,:] = commands
            # obs[:,9:12] = commands_key

            obs[:,6:9] = torch.tensor([vel_command_keyboard[0], vel_command_keyboard[1], vel_command_keyboard[2]], device = obs.device)
            env.update_command(torch.tensor([vel_command_keyboard[0], vel_command_keyboard[1], vel_command_keyboard[2]], device = obs.device))
            # agent stepping
            actions = policy(obs)

            # robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            # robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
            # print("robot quat: ", robot_quat_w)
            # print("robot pos: ", robot_pos_w)

            # roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
            # cam_eye = (robot_pos_w[0] - 2.0 * math.cos(-yaw), robot_pos_w[1] - 2.0 * math.sin(yaw), robot_pos_w[2] + 1.0)
            # cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
            # set the camera view
            # env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)
            # env stepping
            obs, _, _, infos = env.step(actions)
    

    # close the simulator
    env.close()




if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
