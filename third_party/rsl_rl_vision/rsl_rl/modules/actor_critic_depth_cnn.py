from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone, DepthBackbone
from rsl_rl.modules.actor_critic_recurrent import Memory
from rsl_rl.utils import unpad_trajectories


class ActorDepthCNN(nn.Module):
    def __init__(self, 
                 num_obs_proprio, 
                 obs_depth_shape, 
                 num_actions,
                 activation,
                 hidden_dims=[256, 256, 128], 
        ):
        super().__init__()

        self.prop_mlp = nn.Sequential(
            nn.Linear(num_obs_proprio, hidden_dims[0]),
            activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation,
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            activation,
        )
        self.depth_backbone = DepthOnlyFCBackbone(
            output_dim=hidden_dims[2],
            hidden_dim=hidden_dims[1],
            activation=activation,
            num_frames=1,
        )
        # base_backbone = DepthOnlyFCBackbone(
        #     output_dim=hidden_dims[2],
        #     hidden_dim=hidden_dims[1],
        #     activation=activation,
        #     num_frames=1,
        # )
        # self.depth_backbone = DepthBackbone(base_backbone, hidden_dims[2], hidden_dims[2])

        self.action_head = nn.Linear(2 * hidden_dims[2], num_actions)

        self.num_obs_proprio = num_obs_proprio
        self.obs_depth_shape = obs_depth_shape
    
    def forward(self, x):
        prop_input = x[..., :self.num_obs_proprio]
        prop_latent = self.prop_mlp(prop_input)

        depth_input = x[..., self.num_obs_proprio:]
        ori_shape = depth_input.shape
        depth_input = depth_input.reshape(-1, *self.obs_depth_shape)
        depth_latent = self.depth_backbone(depth_input)

        actions = self.action_head(torch.cat((prop_latent, depth_latent), dim=-1))
        return actions
    
    def encode(self, observations):
        original_shape = observations.shape
        
        if observations.dim() == 3:
            observations = observations.reshape(-1, original_shape[-1])

        prop_input = observations[..., :self.num_obs_proprio]
        prop_latent = self.prop_mlp(prop_input)

        depth_input = observations[..., self.num_obs_proprio:]
        depth_input = depth_input.reshape(-1, *self.obs_depth_shape)
        depth_latent = self.depth_backbone(depth_input)

        if len(original_shape) == 3:
            return torch.cat((prop_latent, depth_latent), dim=-1).reshape(*original_shape[:-1], -1)
        
        return torch.cat((prop_latent, depth_latent), dim=-1)
    
    def reset(self, dones=None):
        self.depth_backbone.reset(dones)


class ActorCriticDepthCNN(nn.Module):
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_actor_obs_prop=48,
        obs_depth_shape=(15, 15),
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDepth.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()
        activation = get_activation(activation)
        mlp_input_dim_c = num_critic_obs

        # Policy Function
        self.actor = ActorDepthCNN(num_actor_obs_prop, obs_depth_shape, num_actions, activation, actor_hidden_dims)
        
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP+CNN: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def act_hidden(self, hidden_states):
        mean = self.actor.action_head(hidden_states)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()
    
    def act_hidden_inference(self, hidden_states):
        actions_mean = self.actor.action_head(hidden_states)
        return actions_mean
    
    def evaluate_hidden(self, hidden_states):
        return self.critic.value_head(hidden_states)
    
    def get_hidden_states(self):
        return self.actor.depth_backbone.hidden_states, self.actor.depth_backbone.hidden_states
    


class ActorCriticDepthCNNRecurrent(ActorCriticDepthCNN):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_actor_obs_prop=48,
        num_critic_obs_prop=48,
        obs_depth_shape=(15, 15),
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_input_size=256,
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDepthCNNRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_actor_obs_prop=num_actor_obs_prop,
            num_critic_obs_prop=num_critic_obs_prop,
            obs_depth_shape=obs_depth_shape,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.memory_a = Memory(rnn_input_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(rnn_input_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        observations = self.actor.encode(observations)
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act_hidden(input_a.squeeze(0))

    def act_inference(self, observations):
        observations = self.actor.encode(observations)
        input_a = self.memory_a(observations)
        return super().act_hidden_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        critic_observations = self.critic.encode(critic_observations)
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate_hidden(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
    