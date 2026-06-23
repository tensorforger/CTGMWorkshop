"""
Run 7: Compact dual-stream with explicit edit difference + cross-attention.

Combines insights:
1. Dual-stream: edit stream (frame_A, edited_frame_A) + content stream (noisy, frame_B)
2. Instead of feeding raw frames to edit stream, feed the DIFFERENCE (edited_A - A)
   alongside frame_A. The difference makes the "what changed" explicit.
3. Cross-attention from content to edit for spatial correspondence.
4. Compact: fewer channels to fit more efficiently.

Key difference from run 1 (dual_stream_unet):
- Edit input is (frame_A, edited_frame_A - frame_A) not (frame_A, edited_frame_A)
  This makes the difference signal more explicit
- More compact design
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


class CrossAttn(nn.Module):
    """Content queries edit stream."""

    def __init__(self, q_ch, kv_ch, heads=4, hd=64):
        super().__init__()
        self.norm_q = make_group_norm(q_ch)
        self.norm_kv = make_group_norm(kv_ch)
        self.h, self.hd = heads, hd
        d = heads * hd
        self.to_q = nn.Conv2d(q_ch, d, 1)
        self.to_k = nn.Conv2d(kv_ch, d, 1)
        self.to_v = nn.Conv2d(kv_ch, d, 1)
        self.to_out = nn.Conv2d(d, q_ch, 1)

    def forward(self, content, edit):
        b, c, H, W = content.shape
        if edit.shape[-2:] != content.shape[-2:]:
            edit = F.interpolate(
                edit, size=content.shape[-2:], mode="bilinear", align_corners=False
            )
        q = (
            self.to_q(self.norm_q(content))
            .view(b, self.h, self.hd, H * W)
            .permute(0, 1, 3, 2)
        )
        k = (
            self.to_k(self.norm_kv(edit))
            .view(b, self.h, self.hd, H * W)
            .permute(0, 1, 3, 2)
        )
        v = (
            self.to_v(self.norm_kv(edit))
            .view(b, self.h, self.hd, H * W)
            .permute(0, 1, 3, 2)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.h * self.hd, H, W)
        return content + self.to_out(out)


class EditBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0, use_attn=False, heads=4):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim, dropout)
        self.attn = SelfAttn(out_ch, heads) if use_attn else nn.Identity()

    def forward(self, x, cond):
        return self.attn(self.res(x, cond))


class ContentBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, edit_ch, cond_dim, dropout=0.0, use_attn=False, heads=4
    ):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim, dropout)
        self.self_attn = SelfAttn(out_ch, heads) if use_attn else nn.Identity()
        self.cross_attn = CrossAttn(out_ch, edit_ch, heads) if use_attn else None

    def forward(self, x, edit, cond):
        x = self.res(x, cond)
        x = self.self_attn(x)
        if self.cross_attn is not None:
            x = self.cross_attn(x, edit)
        return x


# --- Edit encoder (encodes frame_A + delta to edit features) ---


class EditEncoder(nn.Module):
    def __init__(self, sample_ch, ch1, ch2):
        super().__init__()
        # Input: (frame_A, edit_delta) where delta = edited_frame_A - frame_A
        self.in_conv = nn.Conv2d(sample_ch * 2, ch1, 3, padding=1)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(ch1, ch1, 3, padding=1),
            make_group_norm(ch1),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(ch1, ch2, 3, padding=1),
            make_group_norm(ch2),
            nn.SiLU(),
        )
        # Light self-attention in bottleneck
        self.attn = SelfAttn(ch2, heads=4)

    def forward(self, frame_A, edit_delta):
        x = F.silu(self.in_conv(torch.cat([frame_A, edit_delta], dim=1)))
        ef0 = x
        ef1 = self.down1(ef0)
        ef2 = self.down2(ef1)
        ef2 = self.attn(ef2)
        return ef0, ef1, ef2


# --- Down/Up stages with per-block cross-attention ---


class DualDownStage(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        edit_ch,
        cond_dim,
        num_blocks,
        dropout=0.0,
        use_attn=False,
        heads=4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ContentBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    edit_ch,
                    cond_dim,
                    dropout,
                    use_attn,
                    heads,
                )
                for i in range(num_blocks)
            ]
        )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, edit, cond):
        for blk in self.blocks:
            x = blk(x, edit, cond)
        return self.downsample(x), x


class DualUpStage(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        edit_ch,
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
                ContentBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    edit_ch,
                    cond_dim,
                    dropout,
                    use_attn,
                    heads,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x, skip, edit, cond):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for blk in self.blocks:
            x = blk(x, edit, cond)
        return x


class FlowRefractorModel(nn.Module):
    """
    Dual-stream with explicit edit delta: edit input = (frame_A, edited_A - frame_A).
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 176,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        num_heads: int = 4,
    ):
        super().__init__()
        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)

        ch1 = base_channels
        ch2 = base_channels * 2

        self.edit_encoder = EditEncoder(sample_channels, ch1, ch2)

        self.in_conv = nn.Conv2d(sample_channels * 2, ch1, 1)

        self.down1 = DualDownStage(
            ch1,
            ch1,
            ch1,
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            heads=num_heads,
        )
        self.down2 = DualDownStage(
            ch1,
            ch2,
            ch1,
            cond_dim,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )

        self.mid = nn.ModuleList(
            [
                ContentBlock(
                    ch2, ch2, ch2, cond_dim, dropout, use_attn=True, heads=num_heads
                )
                for _ in range(4)
            ]
        )

        self.up1 = DualUpStage(
            ch2,
            ch2,
            ch1,
            ch1,
            cond_dim,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )
        self.up2 = DualUpStage(
            ch1,
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

        edit_delta = edited_frame_A - frame_A
        ef0, ef1, ef2 = self.edit_encoder(frame_A, edit_delta)

        x = self.in_conv(torch.cat([sample, frame_B], dim=1))

        x, s1 = self.down1(x, ef0, cond)
        x, s2 = self.down2(x, ef1, cond)

        for blk in self.mid:
            x = blk(x, ef2, cond)

        x = self.up1(x, s2, ef1, cond)
        x = self.up2(x, s1, ef0, cond)

        return self.out_conv(x)
