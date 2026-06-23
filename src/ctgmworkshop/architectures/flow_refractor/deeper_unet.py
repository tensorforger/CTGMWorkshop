"""
Run 5: Deeper UNet with 3 down/up stages, thinner bottleneck.

As suggested in program.md, more stages = better feature hierarchy.
Bottleneck at 1/8 resolution allows capturing global structure.
Fewer mid blocks to stay under param budget.

Channels: 128 -> 256 -> 512 -> 512 -> 256 -> 128
Still concatenates all inputs (same as baseline strategy).
"""

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
    def __init__(self, time_dim: int = 256, cond_dim: int = 512):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.proj(self.time_embed(timestep))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0):
        super().__init__()
        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * out_ch))
        self.norm2 = make_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, cond):
        residual = self.skip(x)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        s, t = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1 + s[:, :, None, None]) + t[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + residual


class SelfAttn(nn.Module):
    def __init__(self, ch, heads=4, head_dim=64):
        super().__init__()
        self.norm = make_group_norm(ch)
        self.heads = heads
        self.head_dim = head_dim
        d = heads * head_dim
        self.to_qkv = nn.Conv2d(ch, d * 3, 1)
        self.to_out = nn.Conv2d(d, ch, 1)

    def forward(self, x):
        b, c, H, W = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h).view(b, 3, self.heads, self.head_dim, H * W)
        q, k, v = [qkv[:, i].permute(0, 1, 3, 2) for i in range(3)]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.heads * self.head_dim, H, W)
        return x + self.to_out(out)


class Stage(nn.Module):
    def __init__(
        self, in_ch, out_ch, cond_dim, num_blocks, dropout=0.0, use_attn=False, heads=4
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(in_ch if i == 0 else out_ch, out_ch, cond_dim, dropout)
                for i in range(num_blocks)
            ]
        )
        self.attns = nn.ModuleList(
            [
                SelfAttn(out_ch, heads) if use_attn else nn.Identity()
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, cond):
        for res, attn in zip(self.blocks, self.attns):
            x = attn(res(x, cond))
        return x


class DownStage(nn.Module):
    def __init__(
        self, in_ch, out_ch, cond_dim, num_blocks, dropout=0.0, use_attn=False, heads=4
    ):
        super().__init__()
        self.stage = Stage(
            in_ch, out_ch, cond_dim, num_blocks, dropout, use_attn, heads
        )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, cond):
        x = self.stage(x, cond)
        return self.downsample(x), x


class UpStage(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        cond_dim,
        num_blocks,
        dropout=0.0,
        use_attn=False,
        heads=4,
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.stage = Stage(
            in_ch + skip_ch, out_ch, cond_dim, num_blocks, dropout, use_attn, heads
        )

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.stage(x, cond)


class FlowRefractorModel(nn.Module):
    """
    3-stage UNet with deeper hierarchy but thinner bottleneck.
    All inputs concatenated at start (same as baseline strategy).
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 128,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        num_heads: int = 4,
    ):
        super().__init__()
        self.conditioning = ConditioningEncoder(time_dim=time_dim, cond_dim=cond_dim)

        ch1 = base_channels  # 128
        ch2 = base_channels * 2  # 256
        ch3 = base_channels * 3  # 384

        self.in_conv = nn.Conv2d(sample_channels * 4, ch1, 1)

        # 3 down stages
        self.down1 = DownStage(
            ch1,
            ch1,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            heads=num_heads,
        )
        self.down2 = DownStage(
            ch1,
            ch2,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )
        self.down3 = DownStage(
            ch2,
            ch3,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )

        # Mid (bottleneck at 1/8 resolution)
        self.mid = Stage(
            ch3,
            ch3,
            cond_dim,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )

        # 3 up stages
        self.up3 = UpStage(
            ch3,
            ch3,
            ch2,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )
        self.up2 = UpStage(
            ch2,
            ch2,
            ch1,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )
        self.up1 = UpStage(
            ch1,
            ch1,
            ch1,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            heads=num_heads,
        )

        self.out_conv = nn.Conv2d(ch1, sample_channels, 1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        frame_A: torch.Tensor,
        frame_B: torch.Tensor,
        edited_frame_A: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.conditioning(timestep)
        x = self.in_conv(torch.cat([sample, frame_A, frame_B, edited_frame_A], dim=1))

        x, s1 = self.down1(x, cond)
        x, s2 = self.down2(x, cond)
        x, s3 = self.down3(x, cond)

        x = self.mid(x, cond)

        x = self.up3(x, s3, cond)
        x = self.up2(x, s2, cond)
        x = self.up1(x, s1, cond)

        return self.out_conv(x)
