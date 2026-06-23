"""
Same as multihead_flow_warp_assist_batched8_deeperflow.py but more base channels
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

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


def conv_norm_act(
    in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        make_group_norm(out_ch),
        nn.SiLU(),
    )


def upsample_flow_heads(flow: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Upsample flow with shape [B, Hh, 2, H, W] and rescale offsets.
    """
    if flow.shape[-2:] == target_hw:
        return flow
    b, hh, _, src_h, src_w = flow.shape
    tgt_h, tgt_w = target_hw
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h
    flat = flow.reshape(b * hh, 2, src_h, src_w)
    flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
    flat = flat.clone()
    flat[:, 0] = flat[:, 0] * scale_x
    flat[:, 1] = flat[:, 1] * scale_y
    return flat.reshape(b, hh, 2, tgt_h, tgt_w)


def batched_grid_warp_head_chunks(
    features: torch.Tensor, flows: torch.Tensor
) -> torch.Tensor:
    """
    Batched grid_sample over heads.

    Args:
        features: [B, Hh, C, H, W]
        flows:    [B, Hh, 2, H, W]

    Returns:
        warped:   [B, Hh, C, H, W]
    """
    b, hh, c, h, w = features.shape
    assert flows.shape == (b, hh, 2, h, w), (flows.shape, (b, hh, 2, h, w))

    flat_x = features.reshape(b * hh, c, h, w)
    flat_flow = flows.reshape(b * hh, 2, h, w)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=features.device, dtype=features.dtype),
        torch.arange(w, device=features.device, dtype=features.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)[None].expand(b * hh, -1, -1, -1)
    sampling_grid = base_grid + flat_flow.permute(0, 2, 3, 1)

    if w > 1:
        sampling_grid[..., 0] = (sampling_grid[..., 0] / (w - 1)) * 2 - 1
    else:
        sampling_grid[..., 0] = 0
    if h > 1:
        sampling_grid[..., 1] = (sampling_grid[..., 1] / (h - 1)) * 2 - 1
    else:
        sampling_grid[..., 1] = 0

    warped = F.grid_sample(
        flat_x,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.reshape(b, hh, c, h, w)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = make_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * out_ch))
            if cond_dim is not None
            else None
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.skip(x)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h)
        if self.cond_proj is not None and cond is not None:
            scale, shift = self.cond_proj(cond).chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + residual


class MultiHeadWarpFiLMBlock(nn.Module):
    """
    Attention-like latent mixer:
    - project conditioning tensor to warp space
    - split warp space into heads
    - warp heads in one batched grid_sample
    - score each head from warped + unwarped chunks
    - softmax fuse
    - project to FiLM scale/shift and modulate x
    """

    def __init__(self, out_ch: int, spatial_ch: int, num_flow_heads: int):
        super().__init__()
        assert out_ch % num_flow_heads == 0, (
            f"out_ch={out_ch} must be divisible by num_flow_heads={num_flow_heads}"
        )
        self.out_ch = out_ch
        self.num_flow_heads = num_flow_heads
        self.head_ch = out_ch // num_flow_heads

        self.warp_proj = nn.Conv2d(spatial_ch, out_ch, 1)
        self.score_net = nn.Sequential(
            nn.Conv2d(2 * self.head_ch, self.head_ch, 1),
            nn.SiLU(),
            nn.Conv2d(self.head_ch, 1, 1),
        )
        self.to_film = nn.Conv2d(out_ch, 2 * out_ch, 1)

    def forward(
        self, x: torch.Tensor, spatial: torch.Tensor, flow_heads: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.out_ch, f"Expected x channels={self.out_ch}, got {c}"

        if spatial.shape[-2:] != (h, w):
            spatial = F.interpolate(
                spatial, size=(h, w), mode="bilinear", align_corners=False
            )

        if flow_heads.shape[-2:] != (h, w):
            flow_heads = upsample_flow_heads(flow_heads, (h, w))

        warp = self.warp_proj(spatial)  # [B, C, H, W]
        warp = warp.view(b, self.num_flow_heads, self.head_ch, h, w)

        warped = batched_grid_warp_head_chunks(
            warp, flow_heads
        )  # [B, Hh, head_ch, H, W]

        score_in = torch.cat([warped, warp], dim=2).reshape(
            b * self.num_flow_heads, 2 * self.head_ch, h, w
        )
        scores = self.score_net(score_in).reshape(b, self.num_flow_heads, 1, h, w)
        weights = torch.softmax(scores, dim=1)

        fused = (weights * warped).permute(0, 2, 1, 3, 4).reshape(b, self.out_ch, h, w)
        scale, shift = self.to_film(fused).chunk(2, dim=1)
        return x * (1 + scale) + shift


class PyramidEncoder(nn.Module):
    """Three-scale encoder returning [full, half, quarter] features."""

    def __init__(
        self,
        in_ch: int,
        channels: Sequence[int],
        num_blocks: Sequence[int] = (2, 2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(channels) == len(num_blocks)
        self.stem = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList(
            [nn.AvgPool2d(2) for _ in range(len(channels) - 1)]
        )

        for i, ch in enumerate(channels):
            blocks = nn.ModuleList()
            in_stage_ch = channels[i - 1] if i > 0 else channels[0]
            for j in range(num_blocks[i]):
                blocks.append(
                    ResBlock(
                        in_stage_ch if j == 0 else ch,
                        ch,
                        cond_dim=None,
                        dropout=dropout,
                    )
                )
            self.stages.append(blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        x = self.stem(x)
        for i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, None)
            feats.append(x)
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x)
        return feats


class DownStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        spatial_ch: int,
        num_blocks: int,
        num_flow_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.flow_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(
                ResBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )
            self.flow_blocks.append(
                MultiHeadWarpFiLMBlock(
                    out_ch=out_ch, spatial_ch=spatial_ch, num_flow_heads=num_flow_heads
                )
            )
        self.downsample = nn.AvgPool2d(2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        spatial: torch.Tensor,
        flow_heads: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for res, flow in zip(self.res_blocks, self.flow_blocks):
            x = res(x, cond)
            x = flow(x, spatial, flow_heads)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        cond_dim: int,
        spatial_ch: int,
        num_blocks: int,
        num_flow_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.flow_blocks = nn.ModuleList()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        for i in range(num_blocks):
            self.res_blocks.append(
                ResBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )
            self.flow_blocks.append(
                MultiHeadWarpFiLMBlock(
                    out_ch=out_ch, spatial_ch=spatial_ch, num_flow_heads=num_flow_heads
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
        spatial: torch.Tensor,
        flow_heads: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for res, flow in zip(self.res_blocks, self.flow_blocks):
            x = res(x, cond)
            x = flow(x, spatial, flow_heads)
        return x


class FlowDownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    in_ch if i == 0 else out_ch, out_ch, cond_dim=None, dropout=dropout
                )
                for i in range(num_blocks)
            ]
        )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, None)
        skip = x
        x = self.downsample(x)
        return x, skip


class FlowUpStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        num_blocks: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    cond_dim=None,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, None)
        return x


class OpticalFlowUNet(nn.Module):
    """
    Dedicated flow UNet. Returns [full, half, quarter] flow hypotheses,
    each shaped [B, num_heads, 2, H, W].

    Compared with the previous version, this backbone adds one more internal
    down/up stage below the quarter-resolution bottleneck. The output flow
    pyramid stays the same three-scale interface used by the main UNet.
    """

    def __init__(
        self,
        in_ch: int,
        base_channels: int,
        num_flow_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_flow_heads = num_flow_heads
        ch0 = base_channels
        ch1 = base_channels * 2
        ch2 = base_channels * 2

        self.in_conv = nn.Conv2d(in_ch, ch0, 3, padding=1)

        # full -> half -> quarter -> eighth
        self.down1 = FlowDownStage(ch0, ch0, num_blocks=1, dropout=dropout)
        self.down2 = FlowDownStage(ch0, ch1, num_blocks=2, dropout=dropout)
        self.down3 = FlowDownStage(ch1, ch2, num_blocks=2, dropout=dropout)

        self.mid_blocks = nn.ModuleList(
            [ResBlock(ch2, ch2, cond_dim=None, dropout=dropout) for _ in range(2)]
        )

        # eighth -> quarter -> half -> full
        self.up3 = FlowUpStage(ch2, ch2, ch1, num_blocks=2, dropout=dropout)
        self.up2 = FlowUpStage(ch1, ch1, ch0, num_blocks=2, dropout=dropout)
        self.up1 = FlowUpStage(ch0, ch0, ch0, num_blocks=1, dropout=dropout)

        # Flow heads are attached at the three output resolutions only.
        self.flow_quarter = nn.Conv2d(ch1, 2 * num_flow_heads, 1)
        self.flow_half = nn.Conv2d(ch0, 2 * num_flow_heads, 1)
        self.flow_full = nn.Conv2d(ch0, 2 * num_flow_heads, 1)
        for layer in (self.flow_quarter, self.flow_half, self.flow_full):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, frame_A: torch.Tensor, frame_B: torch.Tensor
    ) -> List[torch.Tensor]:
        # B -> A flow convention
        x = self.in_conv(torch.cat([frame_B, frame_A], dim=1))
        x, skip0 = self.down1(x)
        x, skip1 = self.down2(x)
        x, skip2 = self.down3(x)

        for block in self.mid_blocks:
            x = block(x, None)

        x = self.up3(x, skip2)
        flow_quarter = self.flow_quarter(x).view(
            x.shape[0], self.num_flow_heads, 2, x.shape[-2], x.shape[-1]
        )

        x = self.up2(x, skip1)
        flow_half = self.flow_half(x).view(
            x.shape[0], self.num_flow_heads, 2, x.shape[-2], x.shape[-1]
        )

        x = self.up1(x, skip0)
        flow_full = self.flow_full(x).view(
            x.shape[0], self.num_flow_heads, 2, x.shape[-2], x.shape[-1]
        )

        return [flow_full, flow_half, flow_quarter]


class MultiHeadWarpAssistUNetBatched8(nn.Module):
    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 288,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        pyramid_channels: Sequence[int] | None = None,
        num_flow_heads: int = 8,
    ):
        super().__init__()
        self.sample_channels = sample_channels
        self.base_channels = base_channels
        self.num_flow_heads = num_flow_heads

        if pyramid_channels is None:
            pyramid_channels = (base_channels // 2, base_channels, base_channels * 2)
        pyramid_channels = tuple(int(ch) for ch in pyramid_channels)
        assert len(pyramid_channels) == 3, "Expected three pyramid scales."

        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)

        self.flow_unet = OpticalFlowUNet(
            in_ch=sample_channels * 2,
            base_channels=base_channels,
            num_flow_heads=num_flow_heads,
            dropout=dropout,
        )

        self.edit_encoder = PyramidEncoder(
            in_ch=sample_channels * 2,
            channels=pyramid_channels,
            num_blocks=(2, 2, 2),
            dropout=dropout,
        )

        ch0 = base_channels
        ch1 = base_channels * 2

        self.in_conv = nn.Conv2d(sample_channels * 2, ch0, 1)

        self.down1 = DownStage(
            in_ch=ch0,
            out_ch=ch0,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[0],
            num_blocks=2,
            num_flow_heads=num_flow_heads,
            dropout=dropout,
        )
        self.down2 = DownStage(
            in_ch=ch0,
            out_ch=ch1,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[1],
            num_blocks=4,
            num_flow_heads=num_flow_heads,
            dropout=dropout,
        )

        self.mid_blocks = nn.ModuleList(
            [ResBlock(ch1, ch1, cond_dim=cond_dim, dropout=dropout) for _ in range(4)]
        )
        self.mid_flow = nn.ModuleList(
            [
                MultiHeadWarpFiLMBlock(
                    out_ch=ch1,
                    spatial_ch=pyramid_channels[2],
                    num_flow_heads=num_flow_heads,
                )
                for _ in range(4)
            ]
        )

        self.up1 = UpStage(
            in_ch=ch1,
            skip_ch=ch1,
            out_ch=ch0,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[1],
            num_blocks=4,
            num_flow_heads=num_flow_heads,
            dropout=dropout,
        )
        self.up2 = UpStage(
            in_ch=ch0,
            skip_ch=ch0,
            out_ch=ch0,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[0],
            num_blocks=2,
            num_flow_heads=num_flow_heads,
            dropout=dropout,
        )

        self.out_conv = nn.Conv2d(ch0, sample_channels, 1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        frame_A: torch.Tensor,
        frame_B: torch.Tensor,
        edited_frame_A: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.conditioning(timestep)

        flow_pyr = self.flow_unet(frame_A, frame_B)
        edit_pyr = self.edit_encoder(torch.cat([frame_A, edited_frame_A], dim=1))

        x = self.in_conv(torch.cat([sample, frame_B], dim=1))

        x, skip0 = self.down1(x, cond, edit_pyr[0], flow_pyr[0])
        x, skip1 = self.down2(x, cond, edit_pyr[1], flow_pyr[1])

        for res, flow in zip(self.mid_blocks, self.mid_flow):
            x = res(x, cond)
            x = flow(x, edit_pyr[2], flow_pyr[2])

        x = self.up1(x, skip1, cond, edit_pyr[1], flow_pyr[1])
        x = self.up2(x, skip0, cond, edit_pyr[0], flow_pyr[0])

        return self.out_conv(x)


FlowRefractorModel = MultiHeadWarpAssistUNetBatched8
