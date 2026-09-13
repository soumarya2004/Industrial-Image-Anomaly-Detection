from __future__ import annotations
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 256,
        base_channels: int = 32,
        num_downsample_blocks: int = 4,
    ):
        super().__init__()
        if image_size % (2 ** num_downsample_blocks) != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"2**num_downsample_blocks (2**{num_downsample_blocks} = "
                f"{2 ** num_downsample_blocks})"
            )
        self.image_size=image_size
        self.num_downsample_blocks=num_downsample_blocks
        self.spatial_after_encode=image_size//(2**num_downsample_blocks)

        #encoder
        encoder_layers=[]
        in_channels=3
        out_channels=base_channels
        for _ in range(num_downsample_blocks):
            encoder_layers+=[
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels=out_channels
            out_channels*=2
        self.encoder_conv=nn.Sequential(*encoder_layers)
        self.final_encoder_channels=in_channels
        flattened_dim=in_channels*self.spatial_after_encode*self.spatial_after_encode
        self.to_latent=nn.Linear(flattened_dim, latent_dim)

        #decoder
        self.from_latent=nn.Linear(latent_dim, flattened_dim)
        decoder_layers=[]
        in_channels=self.final_encoder_channels
        for i in range(num_downsample_blocks):
            out_channels=in_channels//2 if i<num_downsample_blocks-1 else base_channels
            decoder_layers+=[
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels=out_channels
        self.decoder_conv=nn.Sequential(*decoder_layers)
        self.output_conv=nn.Conv2d(in_channels, 3, kernel_size=1)
        self.output_activation=nn.Sigmoid()
        self.latent_dim=latent_dim

    def encode(self, x: torch.Tensor)->torch.Tensor:
        features=self.encoder_conv(x)
        flattened=features.flatten(start_dim=1)
        return self.to_latent(flattened)

    def decode(self, z: torch.Tensor)->torch.Tensor:
        x=self.from_latent(z)
        x=x.view(
            -1, self.final_encoder_channels, self.spatial_after_encode, self.spatial_after_encode
        )
        x=self.decoder_conv(x)
        x=self.output_conv(x)
        return self.output_activation(x)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        z=self.encode(x)
        return self.decode(z)


def reconstruction_error_per_image(
    original: torch.Tensor, reconstructed: torch.Tensor
)->torch.Tensor:
    
    per_pixel_squared_error = (original - reconstructed) ** 2
    return per_pixel_squared_error.mean(dim=[1, 2, 3])
