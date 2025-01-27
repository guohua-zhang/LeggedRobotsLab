import copy
import os
import torch


def export_enc_encoder_as_onnx(
    obs_dim,
    encoder: object,
    path: str,
    normalizer: object | None = None,
    latent_normalize: float = 0,
    filename="encoder.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxEncEncoderExporter(obs_dim, encoder, normalizer, latent_normalize, verbose)
    policy_exporter.export(path, filename)


class _OnnxEncEncoderExporter(torch.nn.Module):
    def __init__(self, obs_dim, encoder, normalizer=None, latent_normalize=0, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.obs_dim = obs_dim
        self.encoder = copy.deepcopy(encoder)
        self.latent_normalize = latent_normalize
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs):
        obs = self.normalizer(obs.view(-1, self.obs_dim)).view(1, -1)
        latent = self.encoder(obs)
        if self.latent_normalize != 0:
            return torch.nn.functional.normalize(latent, p=2, dim=-1) * self.latent_normalize
        else:
            return latent

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.obs_dim)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["latent"],
            dynamic_axes={},
        )
