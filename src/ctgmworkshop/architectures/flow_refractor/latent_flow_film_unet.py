"""Latent Flow-FiLM UNet experiment.

Clarified design:
- A Siamese multiscale encoder extracts frame_A and frame_B features.
- Those features are concatenated at the start of each UNet stage.
- edited_frame_A is only concatenated with the noisy sample at the UNet input.
- The main UNet has no general attention blocks.
- After every residual block, a latent flow block operates only on x:
    * predict multiple flow hypotheses from x
    * project x into warping latents
    * split latents into per-head chunks
    * warp each chunk separately
    * score each chunk using warped + unwarped latents
    * softmax fuse the chunks
    * project back to 2C and apply FiLM to x

The intent is for this block to behave like a self-attention analogue, but with
explicit multiscale latent flow routing.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(channels: int, max_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
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


def conv_norm_act(in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        make_group_norm(out_ch),
        nn.SiLU(),
    )


def _round_up_multiple(value: int, multiple: int) -> int:
    if value % multiple == 0:
        return value
    return value + (multiple - value % multiple)


def _reshape_flow_heads(flow: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert [B, 2 * Hh, H, W] to [B, Hh, 2, H, W]."""
    b, c, h, w = flow.shape
    assert c == 2 * num_heads, f"Expected {2 * num_heads} channels, got {c}"
    return flow.view(b, num_heads, 2, h, w)


def warp_features(features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp features using pixel-space flow offsets.

    flow[:, 0] = dx, flow[:, 1] = dy
    """
    b, c, h, w = features.shape
    if flow.shape[-2:] != (h, w):
        src_h, src_w = flow.shape[-2:]
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow = flow.clone()
        flow[:, 0] = flow[:, 0] * (w / src_w)
        flow[:, 1] = flow[:, 1] * (h / src_h)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=features.device, dtype=features.dtype),
        torch.arange(w, device=features.device, dtype=features.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)[None].expand(b, -1, -1, -1)
    sampling_grid = base_grid + flow.permute(0, 2, 3, 1)

    if w > 1:
        sampling_grid[..., 0] = (sampling_grid[..., 0] / (w - 1)) * 2 - 1
    else:
        sampling_grid[..., 0] = 0
    if h > 1:
        sampling_grid[..., 1] = (sampling_grid[..., 1] / (h - 1)) * 2 - 1
    else:
        sampling_grid[..., 1] = 0

    return F.grid_sample(
        features,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = make_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        self.cond_proj = None
        if cond_dim is not None:
            self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * out_ch))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
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


class SiamesePyramidEncoder(nn.Module):
    """Three-scale encoder returning [full, half, quarter] feature maps."""

    def __init__(
        self,
        in_ch: int,
        channels: Sequence[int],
        num_blocks: Sequence[int] = (2, 2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(channels) == len(num_blocks)
        self.channels = tuple(int(ch) for ch in channels)
        self.stem = nn.Conv2d(in_ch, self.channels[0], 3, padding=1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList([nn.AvgPool2d(2) for _ in range(len(self.channels) - 1)])

        for i, ch in enumerate(self.channels):
            blocks = nn.ModuleList()
            in_stage_ch = self.channels[i - 1] if i > 0 else self.channels[0]
            for j in range(num_blocks[i]):
                blocks.append(ResBlock(in_stage_ch if j == 0 else ch, ch, cond_dim=None, dropout=dropout))
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


class LatentFlowFiLMBlock(nn.Module):
    """Self-contained latent operator that behaves like flow-guided attention.

    Takes a single tensor x and returns the same x after latent flow routing
    and FiLM modulation.
    """

    def __init__(self, feat_ch: int, num_flow_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_flow_heads = num_flow_heads
        self.feat_ch = feat_ch

        warp_ch = _round_up_multiple(max(feat_ch // 2, num_flow_heads * 16), num_flow_heads)
        self.warp_ch = warp_ch
        self.chunk_ch = warp_ch // num_flow_heads

        flow_hidden = max(32, feat_ch // 4)
        score_hidden = max(32, self.chunk_ch)

        self.flow_net = nn.Sequential(
            conv_norm_act(feat_ch, flow_hidden),
            conv_norm_act(flow_hidden, flow_hidden),
            nn.Conv2d(flow_hidden, 2 * num_flow_heads, 1),
        )

        self.to_warp = nn.Conv2d(feat_ch, warp_ch, 1)
        self.to_score_ctx = nn.Conv2d(feat_ch, self.chunk_ch, 1)
        self.score_net = nn.Sequential(
            conv_norm_act(self.chunk_ch * 3, score_hidden),
            nn.Conv2d(score_hidden, 1, 1),
        )
        self.to_film = nn.Sequential(
            nn.Conv2d(self.chunk_ch, 2 * feat_ch, 1),
            nn.SiLU(),
            nn.Conv2d(2 * feat_ch, 2 * feat_ch, 1),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flow = _reshape_flow_heads(self.flow_net(x), self.num_flow_heads)
        warp_latent = self.to_warp(x)
        chunks = list(torch.chunk(warp_latent, self.num_flow_heads, dim=1))
        score_ctx = self.to_score_ctx(x)

        warped_chunks = []
        score_logits = []
        for i in range(self.num_flow_heads):
            warped_i = warp_features(chunks[i], flow[:, i])
            warped_chunks.append(warped_i)
            score_logits.append(self.score_net(torch.cat([warped_i, chunks[i], score_ctx], dim=1)))

        scores = torch.stack(score_logits, dim=1)  # [B, Hh, 1, H, W]
        weights = torch.softmax(scores, dim=1)
        fused = torch.stack(warped_chunks, dim=1)  # [B, Hh, C, H, W]
        fused = (weights * fused).sum(dim=1)

        film = self.to_film(fused)
        scale, shift = film.chunk(2, dim=1)
        x = x * (1 + scale) + shift
        x = self.dropout(x)
        return x


class StageBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int, num_flow_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.res = ResBlock(ch, ch, cond_dim=cond_dim, dropout=dropout)
        self.flow = LatentFlowFiLMBlock(ch, num_flow_heads=num_flow_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.res(x, cond)
        x = self.flow(x)
        return x


class DownStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        frame_ctx_ch: int,
        num_blocks: int,
        dropout: float = 0.0,
        num_flow_heads: int = 4,
    ):
        super().__init__()
        self.entry = nn.Conv2d(in_ch + frame_ctx_ch, out_ch, 1)
        self.blocks = nn.ModuleList(
            [StageBlock(out_ch, cond_dim, num_flow_heads=num_flow_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, frame_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if frame_ctx.shape[-2:] != x.shape[-2:]:
            frame_ctx = F.interpolate(frame_ctx, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.entry(torch.cat([x, frame_ctx], dim=1))
        for block in self.blocks:
            x = block(x, cond)
        skip = x
        x = self.downsample(x)
        return x, skip


class MidStage(nn.Module):
    def __init__(
        self,
        ch: int,
        cond_dim: int,
        frame_ctx_ch: int,
        num_blocks: int,
        dropout: float = 0.0,
        num_flow_heads: int = 4,
    ):
        super().__init__()
        self.entry = nn.Conv2d(ch + frame_ctx_ch, ch, 1)
        self.blocks = nn.ModuleList(
            [StageBlock(ch, cond_dim, num_flow_heads=num_flow_heads, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, frame_ctx: torch.Tensor) -> torch.Tensor:
        if frame_ctx.shape[-2:] != x.shape[-2:]:
            frame_ctx = F.interpolate(frame_ctx, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.entry(torch.cat([x, frame_ctx], dim=1))
        for block in self.blocks:
            x = block(x, cond)
        return x


class UpStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        cond_dim: int,
        frame_ctx_ch: int,
        num_blocks: int,
        dropout: float = 0.0,
        num_flow_heads: int = 4,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.entry = nn.Conv2d(in_ch + skip_ch + frame_ctx_ch, out_ch, 1)
        self.blocks = nn.ModuleList(
            [StageBlock(out_ch, cond_dim, num_flow_heads=num_flow_heads, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
        frame_ctx: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if frame_ctx.shape[-2:] != x.shape[-2:]:
            frame_ctx = F.interpolate(frame_ctx, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.entry(torch.cat([x, skip, frame_ctx], dim=1))
        for block in self.blocks:
            x = block(x, cond)
        return x


class MergeFlowDenoiseUNet(nn.Module):
    """Latent flow block integrated into the main UNet.

    Inputs:
      - sample: noisy latent / sample to denoise
      - timestep: diffusion timestep
      - frame_A
      - frame_B
      - edited_frame_A
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 192,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        num_flow_heads: int = 4,
        pyramid_channels: Sequence[int] | None = None,
    ):
        super().__init__()
        self.sample_channels = sample_channels
        self.base_channels = base_channels
        self.num_flow_heads = num_flow_heads

        if pyramid_channels is None:
            pyramid_channels = (base_channels // 2, base_channels, base_channels * 2)
        pyramid_channels = tuple(int(ch) for ch in pyramid_channels)
        assert len(pyramid_channels) == 3, "Expected 3 pyramid scales: full, half, quarter."

        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)

        self.frame_encoder = SiamesePyramidEncoder(
            in_ch=sample_channels,
            channels=pyramid_channels,
            num_blocks=(2, 2, 2),
            dropout=dropout,
        )

        frame_ctx_ch = [2 * ch for ch in pyramid_channels]

        ch0 = base_channels
        ch1 = base_channels * 2

        # edited_frame_A is only fed at the UNet input together with the noisy sample.
        self.in_conv = nn.Conv2d(sample_channels * 2, ch0, 1)

        # Baseline-like stage block counts: 2 -> 4 -> 8 -> 4 -> 2.
        self.down1 = DownStage(
            in_ch=ch0,
            out_ch=ch0,
            cond_dim=cond_dim,
            frame_ctx_ch=frame_ctx_ch[0],
            num_blocks=2,
            dropout=dropout,
            num_flow_heads=num_flow_heads,
        )
        self.down2 = DownStage(
            in_ch=ch0,
            out_ch=ch1,
            cond_dim=cond_dim,
            frame_ctx_ch=frame_ctx_ch[1],
            num_blocks=4,
            dropout=dropout,
            num_flow_heads=num_flow_heads,
        )

        self.mid_stage = MidStage(
            ch=ch1,
            cond_dim=cond_dim,
            frame_ctx_ch=frame_ctx_ch[2],
            num_blocks=8,
            dropout=dropout,
            num_flow_heads=num_flow_heads,
        )

        self.up1 = UpStage(
            in_ch=ch1,
            skip_ch=ch1,
            out_ch=ch0,
            cond_dim=cond_dim,
            frame_ctx_ch=frame_ctx_ch[1],
            num_blocks=4,
            dropout=dropout,
            num_flow_heads=num_flow_heads,
        )
        self.up2 = UpStage(
            in_ch=ch0,
            skip_ch=ch0,
            out_ch=ch0,
            cond_dim=cond_dim,
            frame_ctx_ch=frame_ctx_ch[0],
            num_blocks=2,
            dropout=dropout,
            num_flow_heads=num_flow_heads,
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

        frame_A_pyr = self.frame_encoder(frame_A)
        frame_B_pyr = self.frame_encoder(frame_B)
        frame_ctx_pyr = [torch.cat([a, b], dim=1) for a, b in zip(frame_A_pyr, frame_B_pyr)]

        x = self.in_conv(torch.cat([sample, edited_frame_A], dim=1))

        x, skip0 = self.down1(x, cond, frame_ctx_pyr[0])
        x, skip1 = self.down2(x, cond, frame_ctx_pyr[1])

        x = self.mid_stage(x, cond, frame_ctx_pyr[2])

        x = self.up1(x, skip1, cond, frame_ctx_pyr[1])
        x = self.up2(x, skip0, cond, frame_ctx_pyr[0])

        return self.out_conv(x)


# Backwards-compatible alias.
FlowRefractorModel = MergeFlowDenoiseUNet


if __name__ == "__main__":
    torch.set_num_threads(1)
    model = MergeFlowDenoiseUNet(sample_channels=4, base_channels=8, num_flow_heads=4)
    b, c, h, w = 1, 4, 16, 16
    sample = torch.randn(b, c, h, w)
    timestep = torch.randint(0, 1000, (b,))
    frame_A = torch.randn(b, c, h, w)
    frame_B = torch.randn(b, c, h, w)
    edited_frame_A = torch.randn(b, c, h, w)

    with torch.no_grad():
        out = model(sample, timestep, frame_A, frame_B, edited_frame_A)

    print("output:", tuple(out.shape))
