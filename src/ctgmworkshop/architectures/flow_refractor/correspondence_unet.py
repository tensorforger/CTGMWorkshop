"""
Run 4: Edit Transfer via Correspondence Cross-Attention.

Core insight: For each pixel in frame_B, find the corresponding pixel in frame_A
via cross-attention (attention implicitly discovers spatial correspondences),
then transfer the edit applied to that pixel.

Architecture:
1. Edit stream: Encode (frame_A) as reference. Encode (edited_frame_A).
   Produce per-pixel "edit features" = concat(frame_A_feat, edited_frame_A_feat).
2. Content stream: Encode (noisy_sample + frame_B).
   Cross-attend to "edit features" using frame_B-derived queries.
   This lets each position in frame_B look up what happened to the
   corresponding position in frame_A.
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


class ResBlock(nn.Module):
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
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + residual


class SelfAttn(nn.Module):
    def __init__(self, ch: int, heads: int = 4, head_dim: int = 64):
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


class CorrespondenceCrossAttn(nn.Module):
    """
    Cross-attention for spatial correspondence:
    - Queries: from frame_B features (what am I?)
    - Keys: from frame_A features (who are you in A?)
    - Values: from edit features at frame_A locations (what happened to you?)

    This lets each position in frame_B find its corresponding position in
    frame_A and retrieve what edit was applied there.
    """

    def __init__(
        self, content_ch: int, edit_ch: int, heads: int = 4, head_dim: int = 64
    ):
        super().__init__()
        self.norm_content = make_group_norm(content_ch)
        self.norm_edit = make_group_norm(edit_ch)
        self.heads = heads
        self.head_dim = head_dim
        d = heads * head_dim
        self.to_q = nn.Conv2d(content_ch, d, 1)  # queries from frame_B
        self.to_k = nn.Conv2d(edit_ch, d, 1)  # keys from frame_A ref
        self.to_v = nn.Conv2d(edit_ch, d, 1)  # values = edit delta info
        self.to_out = nn.Conv2d(d, content_ch, 1)

    def forward(self, content: torch.Tensor, edit: torch.Tensor) -> torch.Tensor:
        b, c, H, W = content.shape
        if edit.shape[-2:] != content.shape[-2:]:
            edit = F.interpolate(
                edit, size=content.shape[-2:], mode="bilinear", align_corners=False
            )
        q = (
            self.to_q(self.norm_content(content))
            .view(b, self.heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        k = (
            self.to_k(self.norm_edit(edit))
            .view(b, self.heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        v = (
            self.to_v(self.norm_edit(edit))
            .view(b, self.heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.heads * self.head_dim, H, W)
        return content + self.to_out(out)


class EditRefBlock(nn.Module):
    """Res -> self-attn -> correspondence cross-attn"""

    def __init__(
        self,
        in_ch,
        out_ch,
        edit_ch,
        cond_dim,
        dropout=0.0,
        use_attn=False,
        heads=4,
        head_dim=64,
    ):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim, dropout)
        self.self_attn = (
            SelfAttn(out_ch, heads, head_dim) if use_attn else nn.Identity()
        )
        self.cross_attn = (
            CorrespondenceCrossAttn(out_ch, edit_ch, heads, head_dim)
            if use_attn
            else None
        )

    def forward(self, x, edit_feat, cond):
        x = self.res(x, cond)
        x = self.self_attn(x)
        if self.cross_attn is not None:
            x = self.cross_attn(x, edit_feat)
        return x


class DownStage(nn.Module):
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
                EditRefBlock(
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

    def forward(self, x, edit_feat, cond):
        for blk in self.blocks:
            x = blk(x, edit_feat, cond)
        return self.downsample(x), x


class UpStage(nn.Module):
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
                EditRefBlock(
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

    def forward(self, x, skip, edit_feat, cond):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        for blk in self.blocks:
            x = blk(x, edit_feat, cond)
        return x


class FrameEncoder(nn.Module):
    """Encodes a pair of frames (frame_A, edited_frame_A) to produce edit reference features."""

    def __init__(self, sample_channels: int, edit_ch_list: tuple):
        super().__init__()
        self.in_conv = nn.Conv2d(sample_channels * 2, edit_ch_list[0], 3, padding=1)
        self.stages = nn.ModuleList()
        ch = edit_ch_list[0]
        for ch_out in edit_ch_list[1:]:
            self.stages.append(
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(ch, ch_out, 3, padding=1),
                    make_group_norm(ch_out),
                    nn.SiLU(),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                    make_group_norm(ch_out),
                    nn.SiLU(),
                )
            )
            ch = ch_out

    def forward(self, frame_A, edited_frame_A):
        x = self.in_conv(torch.cat([frame_A, edited_frame_A], dim=1))
        feats = [x]
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats  # list: [full_res, half_res, quarter_res]


class FlowRefractorModel(nn.Module):
    """
    Correspondence-based edit transfer:
    - Frame encoder: encodes (frame_A, edited_frame_A) at multiple scales
    - Content UNet: denoises (noisy_sample + frame_B)
    - Cross-attention: content queries frame_A edit features, enabling spatial lookup
    """

    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 192,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.01,
        num_heads: int = 4,
    ):
        super().__init__()
        self.conditioning = TimeCondEncoder(time_dim=time_dim, cond_dim=cond_dim)

        ch1 = base_channels
        ch2 = base_channels * 2
        edit_ch = (ch1, ch1, ch2)

        self.frame_encoder = FrameEncoder(sample_channels, edit_ch)

        self.in_conv = nn.Conv2d(sample_channels * 2, ch1, 1)

        self.down1 = DownStage(
            ch1,
            ch1,
            edit_ch[0],
            cond_dim,
            num_blocks=2,
            dropout=dropout,
            use_attn=False,
            heads=num_heads,
        )
        self.down2 = DownStage(
            ch1,
            ch2,
            edit_ch[1],
            cond_dim,
            num_blocks=4,
            dropout=dropout,
            use_attn=True,
            heads=num_heads,
        )

        self.mid = nn.ModuleList(
            [
                EditRefBlock(
                    ch2,
                    ch2,
                    edit_ch[2],
                    cond_dim,
                    dropout,
                    use_attn=True,
                    heads=num_heads,
                )
                for _ in range(4)
            ]
        )

        self.up1 = UpStage(
            ch2,
            ch2,
            ch1,
            edit_ch[1],
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
            edit_ch[0],
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
        edit_feats = self.frame_encoder(frame_A, edited_frame_A)
        ef0, ef1, ef2 = edit_feats

        x = self.in_conv(torch.cat([sample, frame_B], dim=1))

        x, s1 = self.down1(x, ef0, cond)
        x, s2 = self.down2(x, ef1, cond)

        for blk in self.mid:
            x = blk(x, ef2, cond)

        x = self.up1(x, s2, ef1, cond)
        x = self.up2(x, s1, ef0, cond)

        return self.out_conv(x)
