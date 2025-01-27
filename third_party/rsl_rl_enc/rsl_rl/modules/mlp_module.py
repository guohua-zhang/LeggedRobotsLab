from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from .actor_critic import get_activation


class MlpModule(nn.Module):
    def __init__(
        self,
        privileged_input_dim,
        proprio_input_dim,
        latent_dim,
        privileged_encoder_hidden_dims=[256, 256],
        proprio_encoder_hidden_dims=[256, 256],
        activation="elu",
        output_normalize=0,
        orthogonal_init=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()
        self.privileged_input_dim = privileged_input_dim
        self.proprio_input_dim = proprio_input_dim
        self.output_normalize = output_normalize
        self.latent_dim = latent_dim
        self.activation = activation
        self.orthogonal_init = orthogonal_init

        self.privileged_mlp = EncoderModel(self.privileged_input_dim, self.latent_dim, 512, 1, activation)
        print(f"Privileged Generator MLP: {self.privileged_mlp}")

        self.proprio_mlp = EncoderModel(self.proprio_input_dim, self.latent_dim, 512, 1, activation)
        print(f"Propriocetive Generator MLP: {self.proprio_mlp}")

    def privileged_mlp_forward(self, input):
        latent = self.privileged_mlp(input)
        if self.output_normalize != 0:
            return torch.nn.functional.normalize(latent, p=2, dim=-1) * self.output_normalize
        else:
            return latent

    def proprio_mlp_forward(self, input):
        latent = self.proprio_mlp(input)
        if self.output_normalize != 0:
            return torch.nn.functional.normalize(latent, p=2, dim=-1) * self.output_normalize
        else:
            return latent

    def privileged_mlp_forward_with_norm(self, input):
        latent = self.privileged_mlp(input)
        if self.output_normalize != 0:
            return (
                torch.nn.functional.normalize(latent, p=2, dim=-1) * self.output_normalize,
                latent.norm(dim=-1),
            )
        else:
            return latent, latent.norm(dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layer(x) + x


class EncoderModel(nn.Module):
    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dim=256,
        num_blocks=1,
        activation="elu",
    ):
        super().__init__()

        self.activation = get_activation(activation)

        self.input_layer = nn.Linear(num_input_dim, hidden_dim)
        self.tanh = nn.Tanh()

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, self.activation) for _ in range(num_blocks)])

        self.output_layer = nn.Linear(hidden_dim, num_output_dim)

    def forward(self, x):
        x = self.input_layer(x)

        x = self.tanh(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)
