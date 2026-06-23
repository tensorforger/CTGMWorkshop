import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(
    channels: int, max_groups: int = 32, eps: float = 1e-6
) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=eps)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        timesteps = timesteps.float()

        freqs = torch.exp(
            -torch.log(torch.tensor(float(self.max_period), device=timesteps.device))
            * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype)
            / half
        )
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ConditioningEncoder(nn.Module):
    def __init__(self, time_dim: int = 128, cond_dim: int = 256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_vec = self.time_proj(self.time_embed(timestep))
        return time_vec


class ConditionedResidualBlock(nn.Module):
    """
    SDXL-style residual block:
      GN -> SiLU -> Conv
      + condition (scale/shift)
      GN -> SiLU -> Dropout -> Conv
      + skip connection
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = make_group_norm(input_channels)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * output_channels),
        )

        self.norm2 = make_group_norm(output_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )

        if input_channels != output_channels:
            self.skip = nn.Conv2d(
                input_channels, output_channels, kernel_size=1, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class DownStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        cond_dim: int = 256,
        dropout: float = 0.0,
        num_blocks: int = 1,
        downsample_first: bool = False,
    ):
        super().__init__()
        self.downsample_first = downsample_first

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = input_channels if i == 0 else output_channels
            self.blocks.append(
                ConditionedResidualBlock(
                    input_channels=in_ch,
                    output_channels=output_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )

        self.downsample = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):

        if self.downsample_first:
            x = self.downsample(x)

        for block in self.blocks:
            x = block(x, cond)
        skip = x

        if not self.downsample_first:
            x = self.downsample(x)

        return x, skip


class UpStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        cond_dim: int = 256,
        dropout: float = 0.0,
        num_blocks: int = 1,
    ):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = (input_channels + skip_channels) if i == 0 else output_channels
            self.blocks.append(
                ConditionedResidualBlock(
                    input_channels=in_ch,
                    output_channels=output_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        for block in self.blocks:
            x = block(x, cond)

        return x


class FilmCond2D(nn.Module):
    def __init__(self, base_channels: int = 256, cond_channels: int = 256):
        super().__init__()

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(cond_channels, base_channels * 2, kernel_size=1),
        )

    def forward(self, x, cond):
        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)

        x = x * (1 + scale) + shift

        return x


class FlowRefractorModel(nn.Module):
    def __init__(
        self,
        sample_channels: int = 32,
        conditioning_channels: int = 96,
        base_channels: int = 384,
        time_dim: int = 512,
        cond_dim: int = 1024,
        dropout: float = 0.01,
    ):
        super().__init__()

        self.conditioning = ConditioningEncoder(
            time_dim=time_dim,
            cond_dim=cond_dim,
        )

        self.in_conv = nn.Conv2d(
            sample_channels + conditioning_channels,
            base_channels,
            kernel_size=1,
            padding=0,
        )

        self.down_stages = nn.ModuleList(
            [
                DownStage(
                    input_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=2,
                ),
                DownStage(
                    input_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=2,
                ),
            ]
        )

        self.mid_stages = nn.ModuleList(
            [
                ConditionedResidualBlock(
                    input_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
                for i in range(2)
            ]
        )

        self.up_stages = nn.ModuleList(
            [
                UpStage(
                    input_channels=base_channels,
                    skip_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=2,
                ),
                UpStage(
                    input_channels=base_channels,
                    skip_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=2,
                ),
            ]
        )

        self.out_conv = nn.Conv2d(
            base_channels, sample_channels, kernel_size=1, padding=0
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        frame_A: torch.Tensor,
        frame_B: torch.Tensor,
        edited_frame_A: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.conditioning(timestep)

        x = torch.cat([sample, frame_A, frame_B, edited_frame_A], dim=1)

        x = self.in_conv(x)

        skips = []
        for down in self.down_stages:
            x, skip = down(x, cond)
            skips.append(skip)

        for mid in self.mid_stages:
            x = mid(x, cond)

        for up in self.up_stages:
            x = up(x, skips.pop(), cond)

        x = self.out_conv(x)
        return x
