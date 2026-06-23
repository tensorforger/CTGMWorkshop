"""
Run 1: Dual-stream UNet with cross-attention.

Edit stream: processes (frame_A, edited_frame_A) -> "what edit was applied"
Content stream: processes (noisy_sample, frame_B) -> "what to edit"
Cross-attention lets content stream query the edit stream.

This architecture directly models the relationship between the edit pair (A -> edited_A)
and applies it to generate edited_B from B.
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


# ─── time embedding ────────────────────────────────────────────────────────────


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
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.time_proj(self.time_embed(timestep))


# ─── residual block ────────────────────────────────────────────────────────────


class ConditionedResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.dropout(F.silu(h))
        h = self.conv2(h)
        return h + residual


# ─── self-attention ────────────────────────────────────────────────────────────


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.norm = make_group_norm(channels)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.to_q = nn.Conv2d(channels, self.attn_dim, 1)
        self.to_k = nn.Conv2d(channels, self.attn_dim, 1)
        self.to_v = nn.Conv2d(channels, self.attn_dim, 1)
        self.to_out = nn.Conv2d(self.attn_dim, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        out = out.permute(0, 1, 3, 2).reshape(b, self.attn_dim, H, W)
        return x + self.to_out(out)


# ─── cross-attention (content queries edit) ────────────────────────────────────


class CrossAttentionBlock(nn.Module):
    """Content stream queries the edit stream."""

    def __init__(
        self,
        content_channels: int,
        edit_channels: int,
        num_heads: int = 4,
        head_dim: int = 64,
    ):
        super().__init__()
        self.norm_q = make_group_norm(content_channels)
        self.norm_kv = make_group_norm(edit_channels)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.to_q = nn.Conv2d(content_channels, self.attn_dim, 1)
        self.to_k = nn.Conv2d(edit_channels, self.attn_dim, 1)
        self.to_v = nn.Conv2d(edit_channels, self.attn_dim, 1)
        self.to_out = nn.Conv2d(self.attn_dim, content_channels, 1)

    def forward(self, content: torch.Tensor, edit: torch.Tensor) -> torch.Tensor:
        b, c, H, W = content.shape
        q_h = self.norm_q(content)
        kv_h = self.norm_kv(edit)
        # Resize edit features to match content if needed
        if kv_h.shape[-2:] != content.shape[-2:]:
            kv_h = F.interpolate(
                kv_h, size=content.shape[-2:], mode="bilinear", align_corners=False
            )
        q = (
            self.to_q(q_h)
            .view(b, self.num_heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        k = (
            self.to_k(kv_h)
            .view(b, self.num_heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        v = (
            self.to_v(kv_h)
            .view(b, self.num_heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.attn_dim, H, W)
        return content + self.to_out(out)


# ─── combined block for edit stream (res + self-attn) ─────────────────────────


class EditBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        dropout: float = 0.0,
        use_attn: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.res = ConditionedResidualBlock(in_ch, out_ch, cond_dim, dropout)
        self.attn = SelfAttentionBlock(out_ch, num_heads) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.res(x, cond)
        x = self.attn(x)
        return x


# ─── combined block for content stream (res + self-attn + cross-attn) ─────────


class ContentBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        edit_ch: int,
        cond_dim: int,
        dropout: float = 0.0,
        use_attn: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.res = ConditionedResidualBlock(in_ch, out_ch, cond_dim, dropout)
        self.self_attn = (
            SelfAttentionBlock(out_ch, num_heads) if use_attn else nn.Identity()
        )
        self.cross_attn = (
            CrossAttentionBlock(out_ch, edit_ch, num_heads) if use_attn else None
        )

    def forward(
        self, x: torch.Tensor, edit_feat: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.res(x, cond)
        x = self.self_attn(x)
        if self.cross_attn is not None:
            x = self.cross_attn(x, edit_feat)
        return x


# ─── down/up stages ────────────────────────────────────────────────────────────


class EditDownStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        num_blocks: int,
        dropout: float = 0.0,
        use_attn: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                EditBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    cond_dim,
                    dropout,
                    use_attn,
                    num_heads,
                )
            )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, cond):
        for blk in self.blocks:
            x = blk(x, cond)
        skip = x
        return self.downsample(x), skip


class ContentDownStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        edit_ch: int,
        cond_dim: int,
        num_blocks: int,
        dropout: float = 0.0,
        use_attn: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ContentBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    edit_ch,
                    cond_dim,
                    dropout,
                    use_attn,
                    num_heads,
                )
            )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, edit_feats, cond):
        for blk in self.blocks:
            x = blk(x, edit_feats, cond)
        skip = x
        return self.downsample(x), skip


class ContentUpStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        edit_ch: int,
        cond_dim: int,
        num_blocks: int,
        dropout: float = 0.0,
        use_attn: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ContentBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    edit_ch,
                    cond_dim,
                    dropout,
                    use_attn,
                    num_heads,
                )
            )

    def forward(self, x, skip, edit_feats, cond):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for blk in self.blocks:
            x = blk(x, edit_feats, cond)
        return x


# ─── main model ────────────────────────────────────────────────────────────────


class FlowRefractorModel(nn.Module):
    """
    Dual-stream UNet:
    - Edit stream: encodes (frame_A, edited_frame_A) to extract the edit transformation
    - Content stream: denoises (noisy_sample, frame_B), querying edit stream via cross-attention
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 128,
        cond_dim: int = 512,
        time_dim: int = 256,
        dropout: float = 0.01,
        num_heads: int = 4,
    ):
        super().__init__()

        self.conditioning = ConditioningEncoder(time_dim=time_dim, cond_dim=cond_dim)

        # Edit stream input: (frame_A, edited_frame_A) concatenated
        self.edit_in = nn.Conv2d(sample_channels * 2, base_channels, 1)

        # Content stream input: (noisy_sample, frame_B) concatenated
        self.content_in = nn.Conv2d(sample_channels * 2, base_channels, 1)

        ch1 = base_channels
        ch2 = base_channels * 2

        # Edit stream encoder (no cross-attention, just self-attn to understand edit)
        self.edit_down1 = EditDownStage(
            ch1,
            ch1,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            num_heads=num_heads,
        )
        self.edit_down2 = EditDownStage(
            ch1,
            ch2,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=True,
            num_heads=num_heads,
        )
        self.edit_mid = nn.ModuleList(
            [
                EditBlock(
                    ch2, ch2, cond_dim, dropout, use_attn=True, num_heads=num_heads
                )
                for _ in range(2)
            ]
        )

        # Content stream encoder (cross-attends to edit stream)
        self.content_down1 = ContentDownStage(
            ch1,
            ch1,
            ch1,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            num_heads=num_heads,
        )
        self.content_down2 = ContentDownStage(
            ch1,
            ch2,
            ch2,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=True,
            num_heads=num_heads,
        )
        self.content_mid = nn.ModuleList(
            [
                ContentBlock(
                    ch2, ch2, ch2, cond_dim, dropout, use_attn=True, num_heads=num_heads
                )
                for _ in range(2)
            ]
        )

        self.content_up1 = ContentUpStage(
            ch2,
            ch2,
            ch1,
            ch2,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=True,
            num_heads=num_heads,
        )
        self.content_up2 = ContentUpStage(
            ch1,
            ch1,
            ch1,
            ch1,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            num_heads=num_heads,
        )

        self.out_conv = nn.Conv2d(ch1, sample_channels, 1)

    def _run_edit_stream(self, frame_A, edited_frame_A, cond):
        """Encode edit stream and return features at each scale."""
        x = self.edit_in(torch.cat([frame_A, edited_frame_A], dim=1))
        x, skip1 = self.edit_down1(x, cond)
        x, skip2 = self.edit_down2(x, cond)
        for blk in self.edit_mid:
            x = blk(x, cond)
        # Return features at three scales: full, /2, /4 (bottleneck)
        return skip1, skip2, x

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        frame_A: torch.Tensor,
        frame_B: torch.Tensor,
        edited_frame_A: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.conditioning(timestep)

        # Edit stream
        edit_feat1, edit_feat2, edit_feat_mid = self._run_edit_stream(
            frame_A, edited_frame_A, cond
        )

        # Content stream
        x = self.content_in(torch.cat([sample, frame_B], dim=1))

        x, skip1 = self.content_down1(x, edit_feat1, cond)
        x, skip2 = self.content_down2(x, edit_feat2, cond)

        for blk in self.content_mid:
            x = blk(x, edit_feat_mid, cond)

        x = self.content_up1(x, skip2, edit_feat2, cond)
        x = self.content_up2(x, skip1, edit_feat1, cond)

        return self.out_conv(x)
