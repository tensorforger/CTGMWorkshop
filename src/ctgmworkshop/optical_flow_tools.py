import cv2
import torch
import numpy as np

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights,
)

from torchvision.utils import flow_to_image


def get_large_raft(device="cuda"):
    """
    Returns: model, preprocess
    """

    weights = Raft_Large_Weights.DEFAULT

    model = (
        raft_large(
            weights=weights,
            progress=False,
        )
        .to(device)
        .eval()
    )

    preprocess = weights.transforms()

    return model, preprocess


def warp_image(
    image: torch.Tensor, flow: torch.Tensor, interpolation="bilinear"
) -> torch.Tensor:
    """
    Backward warp.

    image:  [B, C, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
    flow: [B, 2, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
    """
    b, c, h, w = image.shape

    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device),
        torch.arange(w, device=image.device),
        indexing="ij",
    )

    grid = torch.stack((xx, yy), dim=0).float()  # [2, H, W]
    grid = grid.unsqueeze(0)  # [1, 2, H, W]

    sample_grid = grid + flow

    sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / (w - 1) - 1.0
    sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / (h - 1) - 1.0

    sample_grid = sample_grid.permute(0, 2, 3, 1)  # [1, H, W, 2]

    warped = torch.nn.functional.grid_sample(
        image,
        sample_grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=True,
    )

    return warped


@torch.no_grad()
def flow_from_numpy(
    frame_A: np.ndarray, frame_B: np.ndarray, model, preprocess, device="cuda"
) -> torch.Tensor:
    """
    Args:
        frame_A: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
        frame_B: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    Returns:
        flow: [1, 2, H, W], dtype = model.dtype, device = device
    """
    img1 = torch.from_numpy(frame_A).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(frame_B).permute(2, 0, 1).unsqueeze(0)

    img1, img2 = preprocess(img1, img2)

    img1 = img1.to(device)
    img2 = img2.to(device)

    flow_predictions = model(img2, img1)
    flow = flow_predictions[-1]

    return flow


def bidir_flow_mask(
    flow_AB: torch.Tensor,
    flow_BA: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        flow_AB: [B, 2, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
        flow_BA: [B, 2, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
    Returns:
        valid_mask: [B, 1, H, W], dtype = model.dtype, device = "cuda"
    """
    warped_BA = warp_image(flow_BA, flow_AB)
    consistency_err = torch.linalg.vector_norm(flow_AB + warped_BA, dim=1)
    valid_mask = consistency_err <= threshold
    valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.to(dtype=flow_AB.dtype)

    return valid_mask


def flow_out_of_bound_mask(flow: torch.Tensor) -> torch.Tensor:
    """
    Args:
        flow: [B, 2, H, W], dtype = float32 (or float16, bfloat16), device = "cuda"
    Returns:
        valid_mask: [B, 1, H, W], dtype = model.dtype, device = "cuda"
    """
    B, C, H, W = flow.shape
    image = torch.ones(B, 1, H, W).to(flow.device, flow.dtype)
    mask = warp_image(image, flow)

    return mask
