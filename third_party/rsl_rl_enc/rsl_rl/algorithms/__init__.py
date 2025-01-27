#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .ppo_mlp import PPO_MLP

__all__ = ["PPO", "PPO_MLP"]
