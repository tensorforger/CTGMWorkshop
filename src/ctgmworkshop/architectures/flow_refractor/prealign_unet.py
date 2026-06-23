"""
Run 8: Input pre-alignment UNet.

Core insight: The main challenge is that frame_A and frame_B are spatially misaligned.
If we align them first, the UNet's job becomes much simpler: just apply the edit.

Pipeline:
1. Estimate flow from frame_A to frame_B coordinate system (A->B)
2. Warp frame_A and edited_frame_A to frame_B's coordinate system
3. Main UNet sees: (noisy_sample, frame_B, warped_A, warped_edited_A)
   -> All in the same spatial coordinate system!

After alignment, the edit delta (warped_edited_A - frame_B) is directly
spatially aligned with frame_B, making the task almost trivial for the UNet.

This is simpler than run 6 (warp_assist_unet) because:
- We warp inputs, not internal features (simpler)
- Main UNet is standard (no special spatial FiLM needed)
- The alignment network can be supervised implicitly
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

    def forward(self, t):
        return self.proj(self.time_embed(t))


def conv_norm_silu(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p),
        make_group_norm(out_ch),
        nn.SiLU(),
    )


class AlignmentNetwork(nn.Module):
    """
    Estimates flow from frame_A to frame_B coordinate system.
    Takes (frame_A, frame_B) and produces a 2D flow field.
    Flow is used to warp frame_A features to frame_B space.
    """

    def __init__(self, sample_channels: int, feat_ch: int = 64):
        super().__init__()
        in_ch = sample_channels * 2  # A + B concatenated

        self.encoder = nn.Sequential(
            conv_norm_silu(in_ch, feat_ch),
            conv_norm_silu(feat_ch, feat_ch),
            nn.AvgPool2d(2),
            conv_norm_silu(feat_ch, feat_ch * 2),
            conv_norm_silu(feat_ch * 2, feat_ch * 2),
            nn.AvgPool2d(2),
            conv_norm_silu(feat_ch * 2, feat_ch * 2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_norm_silu(feat_ch * 2, feat_ch),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_norm_silu(feat_ch, feat_ch),
            nn.Conv2d(feat_ch, 2, 1),  # 2-channel flow
        )
        # Initialize to zero flow
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, frame_A, frame_B):
        x = self.encoder(torch.cat([frame_A, frame_B], dim=1))
        flow = self.decoder(x)
        return flow


def warp(src, flow):
    """
    Warp `src` using `flow` (A->B direction, pixel offsets in src's space).
    flow: [B, 2, H, W]
    """
    B, C, H, W = src.shape
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=src.device),
        torch.arange(W, dtype=torch.float32, device=src.device),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid + flow.permute(0, 2, 3, 1)
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1
    return F.grid_sample(
        src, grid, mode="bilinear", padding_mode="border", align_corners=True
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
        r = self.skip(x)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        s, t = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1 + s[:, :, None, None]) + t[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + r


class SelfAttn(nn.Module):
    def __init__(self, ch, heads=4, hd=64):
        super().__init__()
        self.norm = make_group_norm(ch)
        self.h, self.hd = heads, hd
        d = heads * hd
        self.to_qkv = nn.Conv2d(ch, d * 3, 1)
        self.to_out = nn.Conv2d(d, ch, 1)

    def forward(self, x):
        b, c, H, W = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h).view(b, 3, self.h, self.hd, H * W)
        q, k, v = [qkv[:, i].permute(0, 1, 3, 2) for i in range(3)]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.h * self.hd, H, W)
        return x + self.to_out(out)


class DownStage(nn.Module):
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
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, cond):
        for res, attn in zip(self.blocks, self.attns):
            x = attn(res(x, cond))
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
        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch, out_ch, cond_dim, dropout
                )
                for i in range(num_blocks)
            ]
        )
        self.attns = nn.ModuleList(
            [
                SelfAttn(out_ch, heads) if use_attn else nn.Identity()
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for res, attn in zip(self.blocks, self.attns):
            x = attn(res(x, cond))
        return x


class FlowRefractorModel(nn.Module):
    """
    Pre-alignment UNet:
    1. Estimate A->B flow using lightweight alignment network
    2. Warp frame_A and edited_frame_A to frame_B's coordinate system
    3. Standard UNet on (noisy_sample, frame_B, warped_A, warped_edited_A)
    All inputs are now spatially aligned -> simpler task for UNet.
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 256,
        align_feat_ch: int = 64,
        time_dim: int = 512,
        cond_dim: int = 1024,
        dropout: float = 0.01,
        num_heads: int = 4,
    ):
        super().__init__()
        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)
        self.alignment = AlignmentNetwork(sample_channels, feat_ch=align_feat_ch)

        ch1 = base_channels
        ch2 = base_channels * 2

        # All 4 frames are now aligned -> can concatenate safely
        self.in_conv = nn.Conv2d(sample_channels * 4, ch1, 1)

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
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )

        self.mid = nn.ModuleList(
            [
                ResBlock(ch2, ch2, cond_dim, dropout)
                if i % 2 == 0
                else SelfAttn(ch2, num_heads)
                for i in range(8)
            ]
        )

        self.up1 = UpStage(
            ch2,
            ch2,
            ch1,
            cond_dim,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )
        self.up2 = UpStage(
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

        # Estimate A->B alignment flow
        flow = self.alignment(frame_A, frame_B)

        # Warp frame_A and edited_frame_A to frame_B coordinate system
        warped_A = warp(frame_A, flow)
        warped_edited_A = warp(edited_frame_A, flow)

        # Now all frames are spatially aligned
        x = self.in_conv(torch.cat([sample, frame_B, warped_A, warped_edited_A], dim=1))

        x, s1 = self.down1(x, cond)
        x, s2 = self.down2(x, cond)

        res_out = None
        for i, layer in enumerate(self.mid):
            if i % 2 == 0:
                x = layer(x, cond)
            else:
                x = layer(x)

        x = self.up1(x, s2, cond)
        x = self.up2(x, s1, cond)

        return self.out_conv(x)
