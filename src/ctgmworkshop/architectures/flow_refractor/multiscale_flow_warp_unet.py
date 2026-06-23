"""
Multiscale Warp-Assist UNet with pyramid feature extraction and pyramid flow.

Design goals from the experiment notes:
1. A multiscale feature extractor, built from the same residual blocks used by the UNet encoder.
2. A multiscale flow estimator that consumes feature pairs at each scale and returns a flow field per scale.
3. A multiscale edit encoder, structurally matched to the feature extractor.
4. Warp edit features at each scale using the corresponding flow estimate.
5. Inject the warped edit features into the UNet via spatial FiLM conditioning.

This keeps the inductive bias from the best warp-based runs, but makes the
correspondence and edit paths explicitly pyramidal.

Expected inputs:
- sample: noisy latent / sample tensor, shape [B, sample_channels, H, W]
- timestep: diffusion timestep tensor, shape [B]
- frame_A: source/reference frame A, shape [B, sample_channels, H, W]
- frame_B: target/content frame B, shape [B, sample_channels, H, W]
- edited_frame_A: edited reference frame A', same shape as frame_A

The model predicts a denoised sample with the same channel count as sample.
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


def downsample_2x(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def upsample_flow(flow: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Upsample a flow field and rescale offsets to the new pixel grid."""
    if flow.shape[-2:] == target_hw:
        return flow
    src_h, src_w = flow.shape[-2:]
    tgt_h, tgt_w = target_hw
    scale_y = tgt_h / src_h
    scale_x = tgt_w / src_w
    flow = F.interpolate(flow, size=target_hw, mode="bilinear", align_corners=False)
    flow = flow.clone()
    flow[:, 0] = flow[:, 0] * scale_x
    flow[:, 1] = flow[:, 1] * scale_y
    return flow


def warp_features(features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp `features` using `flow` in pixel coordinates.

    Convention:
    - flow is a dense offset field in the same resolution as `features`
    - flow[:, 0] is x displacement
    - flow[:, 1] is y displacement

    The warp samples from coordinates (x + dx, y + dy), which is suitable when
    the flow encodes "where to sample from" in the source/reference frame.
    """
    b, c, h, w = features.shape
    if flow.shape[-2:] != (h, w):
        flow = upsample_flow(flow, (h, w))

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
    """
    Residual block with optional timestep FiLM conditioning.
    If cond_dim is None, this acts as a plain UNet-style residual block.
    """

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int | None = None, dropout: float = 0.0
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

        self.cond_proj = None
        if cond_dim is not None:
            self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * out_ch))

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


class SpatialFiLMResBlock(nn.Module):
    """Residual block plus spatial FiLM from a conditioning feature map."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        spatial_ch: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim=cond_dim, dropout=dropout)
        self.spatial_proj = nn.Conv2d(spatial_ch, 2 * out_ch, 1)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, spatial: torch.Tensor | None
    ) -> torch.Tensor:
        x = self.res(x, cond)
        if spatial is None:
            return x
        if spatial.shape[-2:] != x.shape[-2:]:
            spatial = F.interpolate(
                spatial, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        scale, shift = self.spatial_proj(spatial).chunk(2, dim=1)
        return x * (1 + scale) + shift


class SelfAttn(nn.Module):
    def __init__(self, ch: int, heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.norm = make_group_norm(ch)
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_qkv = nn.Conv2d(ch, inner * 3, 1)
        self.to_out = nn.Conv2d(inner, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = self.norm(x)
        qkv = self.to_qkv(h_in).view(b, 3, self.heads, self.head_dim, h * w)
        q = qkv[:, 0].permute(0, 1, 3, 2)  # [B, heads, tokens, dim]
        k = qkv[:, 1].permute(0, 1, 3, 2)
        v = qkv[:, 2].permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.heads * self.head_dim, h, w)
        return x + self.to_out(out)


class PyramidEncoder(nn.Module):
    """
    Multiscale encoder that returns a feature pyramid.

    The same residual block topology is used for:
    - content feature extraction for flow estimation
    - edit feature extraction
    - could also be used as a UNet encoder backbone
    """

    def __init__(
        self,
        in_ch: int,
        channels: Sequence[int],
        num_blocks: Sequence[int] = (2, 2, 2),
        dropout: float = 0.0,
        use_attn: Sequence[bool] | None = None,
        num_heads: int = 4,
        head_dim: int = 64,
    ):
        super().__init__()
        assert len(channels) == len(num_blocks), "channels and num_blocks must match"
        if use_attn is None:
            use_attn = tuple(False for _ in channels)
        assert len(use_attn) == len(channels), "use_attn must match channels"

        self.stem = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        self.stages = nn.ModuleList()
        self.attn_stages = nn.ModuleList()

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
            self.attn_stages.append(
                nn.ModuleList(
                    [SelfAttn(ch, heads=num_heads, head_dim=head_dim) for _ in range(1)]
                )
                if use_attn[i]
                else nn.ModuleList([nn.Identity()])
            )

        self.use_attn = list(use_attn)
        self.downsamples = nn.ModuleList(
            [nn.AvgPool2d(2) for _ in range(len(channels) - 1)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        x = self.stem(x)

        for i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, None)
            if self.use_attn[i]:
                x = self.attn_stages[i][0](x)
            feats.append(x)
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x)

        return feats


class FlowHead(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            conv_norm_act(in_ch, hidden_ch),
            conv_norm_act(hidden_ch, hidden_ch),
            conv_norm_act(hidden_ch, hidden_ch),
            nn.Conv2d(hidden_ch, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleFlowEstimator(nn.Module):
    """
    Coarse-to-fine multiscale flow estimator.

    Inputs:
        feat_A_pyr, feat_B_pyr: lists of features from fine -> coarse
    Outputs:
        flow_pyr: list of flows in the same order, each at its own scale
    """

    def __init__(self, feat_channels: Sequence[int], hidden_ch: int = 128):
        super().__init__()
        self.feat_channels = list(feat_channels)
        self.flow_heads = nn.ModuleList()
        for i, ch in enumerate(self.feat_channels):
            # Coarse-to-fine refinement: every level except the coarsest receives
            # the upsampled flow from the next coarser scale.
            in_ch = ch * 2 + (2 if i < len(self.feat_channels) - 1 else 0)
            self.flow_heads.append(FlowHead(in_ch, hidden_ch=hidden_ch))

    def forward(
        self, feat_A_pyr: Sequence[torch.Tensor], feat_B_pyr: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(feat_A_pyr) == len(self.feat_channels)
        assert len(feat_B_pyr) == len(self.feat_channels)

        flows: List[torch.Tensor] = [None] * len(self.feat_channels)  # type: ignore[assignment]
        prev_flow = None

        # coarse -> fine refinement
        for idx in reversed(range(len(self.feat_channels))):
            feat_A = feat_A_pyr[idx]
            feat_B = feat_B_pyr[idx]
            parts = [feat_B, feat_A]

            if prev_flow is not None:
                prev_flow = upsample_flow(prev_flow, feat_A.shape[-2:])
                parts.append(prev_flow)

            flow_delta = self.flow_heads[idx](torch.cat(parts, dim=1))
            flow = flow_delta if prev_flow is None else prev_flow + flow_delta
            flows[idx] = flow
            prev_flow = flow

        return flows


class MultiScaleEditEncoder(PyramidEncoder):
    """Multiscale edit encoder, structurally matched to the content encoder."""

    pass


class DownStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        spatial_ch: int,
        num_blocks: int,
        dropout: float = 0.0,
        use_attn: bool = False,
        num_heads: int = 4,
        head_dim: int = 64,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.append(
                SpatialFiLMResBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    cond_dim=cond_dim,
                    spatial_ch=spatial_ch,
                    dropout=dropout,
                )
            )
            self.attns.append(
                SelfAttn(out_ch, heads=num_heads, head_dim=head_dim)
                if use_attn
                else nn.Identity()
            )

        self.downsample = nn.AvgPool2d(2)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, spatial: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk, attn in zip(self.blocks, self.attns):
            x = blk(x, cond, spatial)
            x = attn(x)
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
        dropout: float = 0.0,
        use_attn: bool = False,
        num_heads: int = 4,
        head_dim: int = 64,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        for i in range(num_blocks):
            self.blocks.append(
                SpatialFiLMResBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    cond_dim=cond_dim,
                    spatial_ch=spatial_ch,
                    dropout=dropout,
                )
            )
            self.attns.append(
                SelfAttn(out_ch, heads=num_heads, head_dim=head_dim)
                if use_attn
                else nn.Identity()
            )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
        spatial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for blk, attn in zip(self.blocks, self.attns):
            x = blk(x, cond, spatial)
            x = attn(x)
        return x


class MultiScaleWarpAssistUNet(nn.Module):
    """
    Multiscale Warp-Assist UNet.

    Pipeline:
    1. Encode frame_A and frame_B with a pyramid content encoder.
    2. Estimate flow at each scale using pairwise concatenation of matching pyramids.
    3. Encode (frame_A, edited_frame_A) with a matching pyramid edit encoder.
    4. Warp each edit feature map with the corresponding scale flow.
    5. Inject warped edit features through spatial FiLM at the matching UNet stage.
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 192,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        num_heads: int = 4,
        head_dim: int = 64,
        pyramid_channels: Sequence[int] | None = None,
        pyramid_blocks: Sequence[int] = (2, 2, 2),
        feat_hidden_ch: int = 128,
    ):
        super().__init__()
        self.sample_channels = sample_channels
        self.base_channels = base_channels

        if pyramid_channels is None:
            pyramid_channels = (base_channels // 2, base_channels, base_channels * 2)
        pyramid_channels = tuple(int(ch) for ch in pyramid_channels)
        assert len(pyramid_channels) == 3, (
            "This implementation expects 3 pyramid scales: full, half, quarter."
        )

        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)

        # Shared topology, separate weights:
        # - content encoder for flow estimation
        # - edit encoder for warped edit features
        self.content_encoder = PyramidEncoder(
            in_ch=sample_channels,
            channels=pyramid_channels,
            num_blocks=pyramid_blocks,
            dropout=dropout,
            use_attn=(False, False, False),
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.edit_encoder = MultiScaleEditEncoder(
            in_ch=sample_channels * 2,
            channels=pyramid_channels,
            num_blocks=pyramid_blocks,
            dropout=dropout,
            use_attn=(False, False, False),
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.flow_estimator = MultiScaleFlowEstimator(
            feat_channels=pyramid_channels,
            hidden_ch=feat_hidden_ch,
        )

        # Main UNet
        ch0 = base_channels
        ch1 = base_channels * 2

        self.in_conv = nn.Conv2d(sample_channels * 2, ch0, 1)

        # Stage-specific warped edit features:
        # - full resolution -> down1 / up2
        # - half resolution -> down2 / up1
        # - quarter resolution -> bottleneck
        self.down1 = DownStage(
            in_ch=ch0,
            out_ch=ch0,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[0],
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.down2 = DownStage(
            in_ch=ch0,
            out_ch=ch1,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[1],
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.mid_blocks = nn.ModuleList(
            [
                SpatialFiLMResBlock(
                    ch1,
                    ch1,
                    cond_dim=cond_dim,
                    spatial_ch=pyramid_channels[2],
                    dropout=dropout,
                )
                for _ in range(4)
            ]
        )
        self.mid_attns = nn.ModuleList(
            [SelfAttn(ch1, heads=num_heads, head_dim=head_dim) for _ in range(4)]
        )

        self.up1 = UpStage(
            in_ch=ch1,
            skip_ch=ch1,
            out_ch=ch0,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[1],
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.up2 = UpStage(
            in_ch=ch0,
            skip_ch=ch0,
            out_ch=ch0,
            cond_dim=cond_dim,
            spatial_ch=pyramid_channels[0],
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            num_heads=num_heads,
            head_dim=head_dim,
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

        # Multiscale content pyramids
        feat_A_pyr = self.content_encoder(frame_A)
        feat_B_pyr = self.content_encoder(frame_B)

        # Multiscale flow pyramid
        flow_pyr = self.flow_estimator(feat_A_pyr, feat_B_pyr)

        # Multiscale edit pyramid
        edit_pyr = self.edit_encoder(torch.cat([frame_A, edited_frame_A], dim=1))

        # Warp edit features at each scale with the corresponding flow
        warped_edit_pyr = [
            warp_features(edit_feat, flow)
            for edit_feat, flow in zip(edit_pyr, flow_pyr)
        ]

        # Main UNet input
        x = self.in_conv(torch.cat([sample, frame_B], dim=1))

        # Encoder
        x, skip0 = self.down1(x, cond, warped_edit_pyr[0])
        x, skip1 = self.down2(x, cond, warped_edit_pyr[1])

        # Bottleneck
        for blk, attn in zip(self.mid_blocks, self.mid_attns):
            x = blk(x, cond, warped_edit_pyr[2])
            x = attn(x)

        # Decoder
        x = self.up1(x, skip1, cond, warped_edit_pyr[1])
        x = self.up2(x, skip0, cond, warped_edit_pyr[0])

        return self.out_conv(x), flow_pyr


# Backwards-compatible alias for training scripts that already import the old name.
FlowRefractorModel = MultiScaleWarpAssistUNet


if __name__ == "__main__":
    # Smoke test
    model = MultiScaleWarpAssistUNet(sample_channels=32, base_channels=192)
    b, c, h, w = 2, 32, 64, 64
    sample = torch.randn(b, c, h, w)
    timestep = torch.randint(0, 1000, (b,))
    frame_A = torch.randn(b, c, h, w)
    frame_B = torch.randn(b, c, h, w)
    edited_frame_A = torch.randn(b, c, h, w)

    with torch.no_grad():
        out = model(sample, timestep, frame_A, frame_B, edited_frame_A)
    print("output:", tuple(out.shape))
