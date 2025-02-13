import torch
import torch.nn as nn


class DepthBackbone(nn.Module):
    def __init__(self, base_backbone, rnn_hidden_dim, output_dim) -> None:
        super().__init__()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.rnn = nn.GRU(
            input_size=base_backbone.output_dim, 
            hidden_size=rnn_hidden_dim, 
            num_layers=1,
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, output_dim),
            last_activation
        )
        self.hidden_states = None

    def forward(self, depth_input, ori_shape, masks=None, hidden_states=None):
        depth_latent = self.base_backbone(depth_input)
        # depth_latent = self.base_backbone(depth_image)
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            depth_latent = depth_latent.view(*ori_shape[:2], -1)
            out, _ = self.rnn(depth_latent, hidden_states)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(depth_latent.unsqueeze(0), self.hidden_states)
            out = out.squeeze(0)
        out = self.output_mlp(out.squeeze(1))
        
        return out

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

    def reset(self, dones):
        self.hidden_states[..., dones, :] = 0.0

    
class DepthOnlyFCBackbone(nn.Module):
    def __init__(self, output_dim, hidden_dim, activation, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        self.output_dim = output_dim
        self.image_compression = nn.Sequential(
            # [1, 24, 32]
            nn.Conv2d(in_channels=self.num_frames, out_channels=16, kernel_size=5),
            # [16, 20, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [16, 10, 14]
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # [32, 8, 12]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 4, 6]
            activation,
            nn.Flatten(),
            
            nn.Linear(32 * 4 * 6, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation
        )

    def forward(self, images: torch.Tensor):
        latent = self.image_compression(images.unsqueeze(1))

        return latent