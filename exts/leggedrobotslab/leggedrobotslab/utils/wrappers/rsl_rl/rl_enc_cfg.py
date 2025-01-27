# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlOnPolicyRunnerEncCfg(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    runner_type: str = "OnPolicyRunner"


@configclass
class RslRlPpoAlgorithmEncCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO_Mlp"
    """The algorithm class name. Default is PPO_Enc."""


@configclass
class EncModuleCfg:
    """Configuration for the proprio and privileged networks."""

    latent_dim: int = MISSING

    privileged_encoder_hidden_dims: list[int] = MISSING

    proprio_encoder_hidden_dims: list[int] = MISSING

    activation: str = MISSING

    output_normalize: float = MISSING

    orthogonal_init: bool = MISSING
