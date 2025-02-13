# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


def get_proprio_obs_dim(env: ManagerBasedRLEnv) -> int:
    """Returns the dimension of the proprioceptive observations."""
    return env.unwrapped.observation_manager.compute_group("proprio").shape[1]


class RslRlVecEnvHistoryWrapper(RslRlVecEnvWrapper):
    """Wraps around Isaac Lab environment for RSL-RL to add history buffer to the proprioception observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv, history_length: int = 1):
        """Initializes the wrapper."""
        super().__init__(env)

        self.history_length = history_length
        self.proprio_obs_dim = get_proprio_obs_dim(env)
        self.proprio_obs_buf = torch.zeros(self.num_envs, self.history_length, self.proprio_obs_dim,
                                                    dtype=torch.float, device=self.unwrapped.device)
        
        self.clip_actions = 20.0

    """
    Properties
    """
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        self.proprio_obs_buf = torch.cat([proprio_obs.unsqueeze(1)] * self.history_length, dim=1)
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        obs_dict["policy"] = curr_obs

        return curr_obs, {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip the actions (for testing only)
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        # print("============== Height Map ==============")
        # print(obs_dict["test_height_map"])
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # update obsservation history buffer & reset the history buffer for done environments
        self.proprio_obs_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None], 
            torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1),
            torch.cat([
                self.proprio_obs_buf[:, 1:],
                proprio_obs.unsqueeze(1)
            ], dim=1)
        )
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        extras["observations"]["policy"] = curr_obs

        # return the step information
        return curr_obs, rew, dones, extras

    def update_command(self, command: torch.Tensor) -> None:
        """Updates the command for the environment."""
        self.proprio_obs_buf[:, -1, 6:9] = command

    def close(self):  # noqa: D102
        return self.env.close()