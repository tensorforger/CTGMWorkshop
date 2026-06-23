"""
Run 6: Warp-Assist UNet with explicit flow estimation.

Core idea:
1. Estimate a warp field from frame_B to frame_A coordinates (using a lightweight flow head)
2. Warp edit features from frame_A space to frame_B space
3. Use warped edit features as spatial conditioning for the denoising UNet

This is a principled approach: the video frames have smooth motion, so
the warp between A and B is simple. Once we warp the edit features,
the model just needs to apply them at each position.

Architecture:
- Feature extractor: shallow CNN to extract features from frame_A and frame_B
- Flow estimator: lightweight module that estimates flow from B->A
- Warp: use grid_sample to warp frame_A edit features to frame_B space
- Main UNet: processes (noisy_sample + frame_B) with warped edit features as spatial FiLM
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


class TimeCondEncoder(nn.Module):
    def __init__(self, time_dim: int = 256, cond_dim: int = 512):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.proj(self.time_embed(timestep))


def conv_norm_act(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        make_group_norm(out_ch),
        nn.SiLU(),
    )


class FlowEstimator(nn.Module):
    """
    Lightweight flow estimator. Takes concatenated features of (frame_B, frame_A)
    and estimates a 2D flow field (B->A direction, for warping edit features to B).
    """

    def __init__(self, feat_ch: int, hidden_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_norm_act(feat_ch * 2, hidden_ch),
            conv_norm_act(hidden_ch, hidden_ch),
            conv_norm_act(hidden_ch, hidden_ch),
            nn.Conv2d(hidden_ch, 2, 1),  # 2-channel flow field
        )
        # Init to near-zero flow
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat_B, feat_A):
        return self.net(torch.cat([feat_B, feat_A], dim=1))


def warp_features(features, flow):
    """
    Warp `features` (in frame_A space) to frame_B space using `flow` (B->A offsets).
    flow: [B, 2, H, W] pixel offsets in A coordinates to sample from
    """
    B, C, H, W = features.shape
    # Build sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=features.device),
        torch.arange(W, dtype=torch.float32, device=features.device),
        indexing="ij",
    )
    grid = (
        torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    )  # [B, H, W, 2]

    # flow is [B, 2, H, W]: dx, dy offsets
    flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    grid = grid + flow_permuted

    # Normalize to [-1, 1]
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

    return F.grid_sample(
        features, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


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


class SpatialCondResBlock(nn.Module):
    """ResBlock with additional spatial FiLM from warped edit features."""

    def __init__(self, in_ch, out_ch, cond_dim, spatial_ch, dropout=0.0):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim, dropout)
        self.spatial_proj = nn.Conv2d(spatial_ch, 2 * out_ch, 1)

    def forward(self, x, cond, spatial):
        x = self.res(x, cond)
        if spatial is not None:
            if spatial.shape[-2:] != x.shape[-2:]:
                spatial = F.interpolate(
                    spatial, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            s, t = self.spatial_proj(spatial).chunk(2, dim=1)
            norm = (
                make_group_norm(x.shape[1]).to(x.device) if False else x
            )  # skip extra norm
            x = x * (1 + s) + t
        return x


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


class WarpDownStage(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        cond_dim,
        spatial_ch,
        num_blocks,
        dropout=0.0,
        use_attn=False,
        heads=4,
        head_dim=64,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                SpatialCondResBlock(
                    in_ch if i == 0 else out_ch, out_ch, cond_dim, spatial_ch, dropout
                )
            )
            self.attns.append(
                SelfAttn(out_ch, heads, head_dim) if use_attn else nn.Identity()
            )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, cond, spatial=None):
        for blk, attn in zip(self.blocks, self.attns):
            x = blk(x, cond, spatial)
            x = attn(x)
        return self.downsample(x), x


class WarpUpStage(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        cond_dim,
        spatial_ch,
        num_blocks,
        dropout=0.0,
        use_attn=False,
        heads=4,
        head_dim=64,
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                SpatialCondResBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    cond_dim,
                    spatial_ch,
                    dropout,
                )
            )
            self.attns.append(
                SelfAttn(out_ch, heads, head_dim) if use_attn else nn.Identity()
            )

    def forward(self, x, skip, cond, spatial=None):
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


class FlowRefractorModel(nn.Module):
    """
    Warp-Assist UNet:
    1. Extract shallow features from frame_A and frame_B
    2. Estimate flow field B->A
    3. Warp "edit features" (from edited_frame_A - frame_A) to frame_B space
    4. Main UNet: denoises (noisy_sample + frame_B) using warped edit features as spatial FiLM
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 192,
        feat_channels: int = 48,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        num_heads: int = 4,
    ):
        super().__init__()
        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)

        # Shallow feature extractor for flow estimation
        self.feat_extractor = nn.Sequential(
            conv_norm_act(sample_channels, feat_channels),
            conv_norm_act(feat_channels, feat_channels),
        )

        # Flow estimator: takes feat_B and feat_A, produces B->A flow
        self.flow_estimator = FlowEstimator(feat_ch=feat_channels, hidden_ch=64)

        # Edit feature encoder: encodes (frame_A, edited_frame_A) -> spatial edit features
        edit_out_ch = base_channels // 2
        self.edit_feat_enc = nn.Sequential(
            conv_norm_act(sample_channels * 2, edit_out_ch),
            conv_norm_act(edit_out_ch, edit_out_ch),
            conv_norm_act(edit_out_ch, edit_out_ch),
        )
        spatial_ch = edit_out_ch

        ch1 = base_channels
        ch2 = base_channels * 2

        # Main UNet
        self.in_conv = nn.Conv2d(sample_channels * 2, ch1, 1)  # noisy + frame_B

        self.down1 = WarpDownStage(
            ch1,
            ch1,
            cond_dim,
            spatial_ch,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            heads=num_heads,
        )
        self.down2 = WarpDownStage(
            ch1,
            ch2,
            cond_dim,
            spatial_ch,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )

        self.mid_blocks = nn.ModuleList(
            [
                SpatialCondResBlock(ch2, ch2, cond_dim, spatial_ch, dropout)
                for _ in range(4)
            ]
        )
        self.mid_attns = nn.ModuleList([SelfAttn(ch2, num_heads) for _ in range(4)])

        self.up1 = WarpUpStage(
            ch2,
            ch2,
            ch1,
            cond_dim,
            spatial_ch,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )
        self.up2 = WarpUpStage(
            ch1,
            ch1,
            ch1,
            cond_dim,
            spatial_ch,
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

        # Extract features for flow estimation
        feat_A = self.feat_extractor(frame_A)
        feat_B = self.feat_extractor(frame_B)

        # Estimate B->A flow
        flow = self.flow_estimator(feat_B, feat_A)

        # Encode edit features in frame_A space
        edit_feat_A = self.edit_feat_enc(torch.cat([frame_A, edited_frame_A], dim=1))

        # Warp edit features to frame_B space
        edit_feat_B = warp_features(edit_feat_A, flow)

        # Main UNet: (noisy + frame_B) conditioned by warped edit features
        x = self.in_conv(torch.cat([sample, frame_B], dim=1))

        x, s1 = self.down1(x, cond, edit_feat_B)
        x, s2 = self.down2(x, cond, edit_feat_B)

        for blk, attn in zip(self.mid_blocks, self.mid_attns):
            x = blk(x, cond, edit_feat_B)
            x = attn(x)

        x = self.up1(x, s2, cond, edit_feat_B)
        x = self.up2(x, s1, cond, edit_feat_B)

        return self.out_conv(x)
