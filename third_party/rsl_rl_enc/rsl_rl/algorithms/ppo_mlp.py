#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, MlpModule
from rsl_rl.storage import RolloutStorageMlp


class PPO_MLP(PPO):
    actor_critic: ActorCritic
    mlp: MlpModule

    def __init__(
        self,
        actor_critic,
        mlp_module,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):
        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
        )

        self.mlp = mlp_module

        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.parameters()},
                {"params": self.mlp.privileged_mlp.parameters()},
                {"params": self.mlp.proprio_mlp.parameters()},
            ],
            lr=learning_rate,
        )
        self.transition = RolloutStorageMlp.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageMlp(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()
        self.mlp.test()

    def train_mode(self):
        self.actor_critic.train()
        self.mlp.train()

    def act(self, obs, critic_obs):
        privileged_latent = self.mlp.privileged_mlp_forward(critic_obs)
        proprio_latent = self.mlp.proprio_mlp_forward(obs)

        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(torch.cat((obs, proprio_latent), dim=1)).detach()
        self.transition.values = self.actor_critic.evaluate(torch.cat((critic_obs, privileged_latent), dim=1)).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        mean_reconstruction_loss = 0
        mean_latent_norm = 0
        mini_batch_count = 0
        mlp_mini_batch_count = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            self.proprio_latent_batch = self.mlp.proprio_mlp_forward(obs_batch)
            self.privileged_latent_batch = self.mlp.privileged_mlp_forward(critic_obs_batch)

            self.actor_critic.act(torch.cat((obs_batch, self.proprio_latent_batch), dim=-1))
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                torch.cat((critic_obs_batch, self.privileged_latent_batch), dim=-1)
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            # if mini_batch_count == 0:
            #     print(ratio.max(), ratio.min())
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()
            mini_batch_count += 1

        generator = self.storage.mlp_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            mlp_obs_batch,
            mlp_critic_obs_batch,
        ) in generator:
            proprio_latent_batch = self.mlp.proprio_mlp_forward(mlp_obs_batch)
            privileged_latent_batch, latent_norm = self.mlp.privileged_mlp_forward_with_norm(mlp_critic_obs_batch)
            reconstruction_loss = (privileged_latent_batch.detach() - proprio_latent_batch).pow(2).mean()

            loss = reconstruction_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.mlp.proprio_mlp.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_reconstruction_loss += reconstruction_loss.item()
            mean_latent_norm += latent_norm.mean().item()
            mlp_mini_batch_count += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        mean_reconstruction_loss /= mlp_mini_batch_count
        mean_latent_norm /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_kl, mean_reconstruction_loss, mean_latent_norm
