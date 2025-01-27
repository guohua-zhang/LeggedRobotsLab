from omni.isaac.lab.utils import configclass

from leggedrobotslab.utils.wrappers.rsl_rl import (
    EncModuleCfg,
    RslRlOnPolicyRunnerEncCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmEncCfg,
)


@configclass
class GO1FlatPPORunnerEncCfg(RslRlOnPolicyRunnerEncCfg):
    runner_type = "OnPolicyRunnerMlp"
    num_steps_per_env = 24
    max_iterations = 3001
    save_interval = 200
    experiment_name = "go1_flat"
    run_name = ""
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmEncCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    mlp_cfg = EncModuleCfg(
        latent_dim=64,
        privileged_encoder_hidden_dims=[512, 256],
        proprio_encoder_hidden_dims=[512, 256],
        activation="elu",
        output_normalize=1,
        orthogonal_init=True,
    )


class GO1RoughPPORunnerEncCfg(GO1FlatPPORunnerEncCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go1_rough"
        self.run_name = ""
        self.runner_type = "OnPolicyRunnerMlp"
