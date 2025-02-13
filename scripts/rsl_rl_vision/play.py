# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import subprocess

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import math
import torch
import imageio

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab.utils import update_class_from_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
)

from omni.isaac.leggedloco.config import *
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

from utils import quat2eulers


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    # env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
    #     args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    # )
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    # update agent config with the one from the loaded run
    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    assert os.path.exists(log_agent_cfg_file_path), f"Agent config file not found: {log_agent_cfg_file_path}"
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    cam_eye = (robot_pos_w[0]+2.5, robot_pos_w[1]+2.5, 20)
    cam_target = (robot_pos_w[0], robot_pos_w[1], 0.0)
    # set the camera view
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")
    export_policy_as_jit(ppo_runner.alg.actor_critic, None, path=export_model_dir, filename="policy.jit")
    # reset environment
    obs, _ = env.get_observations()
    if args_cli.video:
        base_env = env.unwrapped
        init_frame = base_env.render()
        frames = [init_frame]
    
        robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        cam_eye = (robot_pos_w[0]+2.5, robot_pos_w[1]+2.5, 5)
        cam_target = (robot_pos_w[0], robot_pos_w[1], 0.0)
        # set the camera view
        env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, infos = env.step(actions)
            # import pdb; pdb.set_trace()

            if args_cli.video and len(frames) < args_cli.video_length:
                base_env = env.unwrapped
                frame = base_env.render()
                frames.append(frame)

                robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
                roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
                cam_eye = (robot_pos_w[0] + 3.0, robot_pos_w[1] + 3.0, robot_pos_w[2] + 3.0)
                cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
                # set the camera view
                env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

            if args_cli.video and len(frames) == args_cli.video_length:
                break
    
    writer = imageio.get_writer(os.path.join(log_dir, f"{args_cli.load_run}.mp4"), fps=50)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
