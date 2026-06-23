"""
Run 2: Edit-delta FiLM conditioning.

Key idea: The edit is fully captured by (edited_frame_A - frame_A) in feature space.
Encode this delta into a spatial conditioning map and inject it via FiLM.

The main UNet processes (noisy_sample + frame_B).
A small "edit encoder" encodes both frame_A and edited_frame_A,
computes per-scale difference features, and injects them as
additional FiLM scale/shift into each stage of the main UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── helpers ───────────────────────────────────────────────────────────────────


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


class TimeCondEncoder(nn.Module):
    def __init__(self, time_dim: int = 256, cond_dim: int = 512):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.proj(self.time_embed(timestep))


# ─── edit encoder ─────────────────────────────────────────────────────────────


class EditEncoder(nn.Module):
    """
    Encodes (frame_A, edited_frame_A) and produces per-scale spatial delta features.
    Output: list of feature maps representing "what changed" at each resolution.
    """

    def __init__(self, sample_channels: int = 32, channels_list=(64, 128, 256)):
        super().__init__()
        # Shared encoder for both A and edited_A
        self.in_conv = nn.Conv2d(sample_channels, channels_list[0], 3, padding=1)

        self.stages = nn.ModuleList()
        ch_in = channels_list[0]
        for ch_out in channels_list[1:]:
            self.stages.append(
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(ch_in, ch_out, 3, padding=1),
                    nn.GroupNorm(min(32, ch_out), ch_out),
                    nn.SiLU(),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                )
            )
            ch_in = ch_out

        # Delta projection: (frame_A_feat - edited_A_feat) -> spatial FiLM params
        self.delta_projs = nn.ModuleList()
        self.delta_projs.append(
            nn.Conv2d(channels_list[0] * 2, channels_list[0] * 2, 1)
        )
        for ch in channels_list[1:]:
            self.delta_projs.append(nn.Conv2d(ch * 2, ch * 2, 1))

    def _encode(self, x):
        feats = []
        x = self.in_conv(x)
        feats.append(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats

    def forward(self, frame_A, edited_frame_A):
        feats_A = self._encode(frame_A)
        feats_eA = self._encode(edited_frame_A)
        delta_feats = []
        for fa, fea, proj in zip(feats_A, feats_eA, self.delta_projs):
            # Concatenate instead of subtract to let model learn the relationship
            delta = proj(torch.cat([fa, fea], dim=1))
            delta_feats.append(delta)
        return delta_feats


# ─── spatial FiLM-conditioned residual block ───────────────────────────────────


class SpatialFiLMBlock(nn.Module):
    """
    Residual block conditioned by:
    1. Global time conditioning (FiLM, as in SDXL)
    2. Spatial delta features from edit encoder (channel-wise scale+shift)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        spatial_cond_ch: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Global FiLM from time
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * out_ch))

        # Spatial FiLM from edit delta (if provided)
        if spatial_cond_ch > 0:
            self.spatial_proj = nn.Conv2d(spatial_cond_ch, 2 * out_ch, 1)
        else:
            self.spatial_proj = None

        self.norm2 = make_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        spatial_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = self.skip(x)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Global FiLM from time
        t_scale, t_shift = self.time_proj(time_cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + t_scale[:, :, None, None]) + t_shift[:, :, None, None]

        # Spatial FiLM from edit delta
        if self.spatial_proj is not None and spatial_cond is not None:
            if spatial_cond.shape[-2:] != h.shape[-2:]:
                spatial_cond = F.interpolate(
                    spatial_cond,
                    size=h.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            s_scale, s_shift = self.spatial_proj(spatial_cond).chunk(2, dim=1)
            h = h * (1 + s_scale) + s_shift

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + residual


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.norm = make_group_norm(channels)
        self.num_heads = num_heads
        self.head_dim = head_dim
        d = num_heads * head_dim
        self.to_q = nn.Conv2d(channels, d, 1)
        self.to_k = nn.Conv2d(channels, d, 1)
        self.to_v = nn.Conv2d(channels, d, 1)
        self.to_out = nn.Conv2d(d, channels, 1)

    def forward(self, x):
        b, c, H, W = x.shape
        h = self.norm(x)
        q = (
            self.to_q(h)
            .view(b, self.num_heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        k = (
            self.to_k(h)
            .view(b, self.num_heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        v = (
            self.to_v(h)
            .view(b, self.num_heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.num_heads * self.head_dim, H, W)
        return x + self.to_out(out)


# ─── UNet stages ───────────────────────────────────────────────────────────────


class DownStage(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        cond_dim,
        spatial_cond_ch,
        num_blocks=2,
        dropout=0.0,
        use_attn=False,
        num_heads=4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                SpatialFiLMBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    cond_dim,
                    spatial_cond_ch,
                    dropout,
                )
            )
        self.attns = nn.ModuleList(
            [
                AttentionBlock(out_ch, num_heads) if use_attn else nn.Identity()
                for _ in range(num_blocks)
            ]
        )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, time_cond, spatial_cond=None):
        for blk, attn in zip(self.blocks, self.attns):
            x = blk(x, time_cond, spatial_cond)
            x = attn(x)
        skip = x
        return self.downsample(x), skip


class UpStage(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        cond_dim,
        spatial_cond_ch,
        num_blocks=2,
        dropout=0.0,
        use_attn=False,
        num_heads=4,
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                SpatialFiLMBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    cond_dim,
                    spatial_cond_ch,
                    dropout,
                )
            )
        self.attns = nn.ModuleList(
            [
                AttentionBlock(out_ch, num_heads) if use_attn else nn.Identity()
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, skip, time_cond, spatial_cond=None):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for blk, attn in zip(self.blocks, self.attns):
            x = blk(x, time_cond, spatial_cond)
            x = attn(x)
        return x


# ─── main model ────────────────────────────────────────────────────────────────


class FlowRefractorModel(nn.Module):
    """
    Edit-delta FiLM UNet:
    - A small edit encoder produces spatial "delta" features (frame_A vs edited_frame_A)
    - Main UNet processes (noisy_sample + frame_B) and is spatially conditioned by delta features
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 224,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        num_heads: int = 4,
    ):
        super().__init__()

        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)

        ch1 = base_channels
        ch2 = base_channels * 2

        # Edit encoder: produces delta feats at [ch1, ch1, ch2] resolutions
        edit_channels_list = (ch1 // 2, ch1, ch2)
        self.edit_encoder = EditEncoder(
            sample_channels, channels_list=edit_channels_list
        )

        # delta feature channels at each scale
        delta_ch_full = edit_channels_list[0] * 2  # spatial FiLM at full res
        delta_ch_half = edit_channels_list[1] * 2  # spatial FiLM at /2 res
        delta_ch_qtr = edit_channels_list[2] * 2  # spatial FiLM at /4 res

        # Main UNet input: (noisy_sample + frame_B)
        self.in_conv = nn.Conv2d(sample_channels * 2, ch1, 1)

        self.down1 = DownStage(
            ch1,
            ch1,
            cond_dim,
            delta_ch_full,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            num_heads=num_heads,
        )
        self.down2 = DownStage(
            ch1,
            ch2,
            cond_dim,
            delta_ch_half,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            num_heads=num_heads,
        )

        self.mid_blocks = nn.ModuleList(
            [
                SpatialFiLMBlock(ch2, ch2, cond_dim, delta_ch_qtr, dropout)
                for _ in range(4)
            ]
        )
        self.mid_attns = nn.ModuleList(
            [AttentionBlock(ch2, num_heads) for _ in range(4)]
        )

        self.up1 = UpStage(
            ch2,
            ch2,
            ch1,
            cond_dim,
            delta_ch_half,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            num_heads=num_heads,
        )
        self.up2 = UpStage(
            ch1,
            ch1,
            ch1,
            cond_dim,
            delta_ch_full,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            num_heads=num_heads,
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
        time_cond = self.conditioning(timestep)

        # Get edit delta features at each scale
        delta_feats = self.edit_encoder(frame_A, edited_frame_A)
        delta_full, delta_half, delta_qtr = delta_feats

        # Main UNet
        x = self.in_conv(torch.cat([sample, frame_B], dim=1))

        x, skip1 = self.down1(x, time_cond, delta_full)
        x, skip2 = self.down2(x, time_cond, delta_half)

        for blk, attn in zip(self.mid_blocks, self.mid_attns):
            x = blk(x, time_cond, delta_qtr)
            x = attn(x)

        x = self.up1(x, skip2, time_cond, delta_half)
        x = self.up2(x, skip1, time_cond, delta_full)

        return self.out_conv(x)
