#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Geometric augmentation with synchronized action transformation.

When flipping images horizontally, the action's horizontal component must be negated
to maintain image-action correspondence. Random crop does not require action change.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass
class GeometricAugmentationConfig:
    """Configuration for geometric augmentation with action sync."""

    flip_prob: float = 0.5
    crop_ratio_range: tuple[float, float] = (0.95, 1.0)  # keep 95%~100%, i.e. crop 0~5%
    target_size: tuple[int, int] = (128, 128)
    horizontal_axis: int = 0  # 0=dx, 1=dy
    enable_flip: bool = True  # Toggle random horizontal flip for ablation
    enable_crop: bool = True  # Toggle random crop for ablation


def sample_augmentation_params(
    batch_size: int,
    H: int,
    W: int,
    config: GeometricAugmentationConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample per-sample augmentation parameters.

    Returns:
        dict with keys: flip (B,), crop_top (B,), crop_left (B,), crop_h (scalar), crop_w (scalar)
    """
    if config.enable_crop:
        low, high = config.crop_ratio_range
        crop_h = max(1, int(H * (low + (high - low) * torch.rand(1, device=device).item())))
        crop_w = max(1, int(W * (low + (high - low) * torch.rand(1, device=device).item())))
        crop_top = torch.randint(0, max(1, H - crop_h + 1), (batch_size,), device=device)
        crop_left = torch.randint(0, max(1, W - crop_w + 1), (batch_size,), device=device)
    else:
        crop_h, crop_w = H, W
        crop_top = torch.zeros(batch_size, dtype=torch.long, device=device)
        crop_left = torch.zeros(batch_size, dtype=torch.long, device=device)

    if config.enable_flip:
        flip = torch.rand(batch_size, device=device) < config.flip_prob
    else:
        flip = torch.zeros(batch_size, dtype=torch.bool, device=device)

    return {
        "flip": flip,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "crop_h": crop_h,
        "crop_w": crop_w,
    }


def apply_image_augmentation(
    images: torch.Tensor,
    params: dict[str, Any],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Apply flip + crop + resize to images. images: (B, C, H, W)."""
    B, C, H, W = images.shape
    device = images.device
    flip = params["flip"]
    crop_top = params["crop_top"]
    crop_left = params["crop_left"]
    crop_h = params["crop_h"]
    crop_w = params["crop_w"]

    # Per-sample crop: build indices
    batch_idx = torch.arange(B, device=device)
    rows = (
        torch.arange(crop_h, device=device).unsqueeze(0).expand(B, -1)
        + crop_top.unsqueeze(1)
    )
    cols = (
        torch.arange(crop_w, device=device).unsqueeze(0).expand(B, -1)
        + crop_left.unsqueeze(1)
    )
    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)

    images_hwcn = images.permute(0, 2, 3, 1)
    # Advanced indexing: all indices must broadcast to same shape (B, crop_h, crop_w, C)
    batch_idx = batch_idx.view(B, 1, 1, 1)
    rows = rows.unsqueeze(-1)
    cols = cols.unsqueeze(-1)
    channel_idx = torch.arange(C, device=device).view(1, 1, 1, C)
    cropped = images_hwcn[batch_idx, rows, cols, channel_idx]
    cropped = cropped.permute(0, 3, 1, 2)

    # Resize to target_size
    resized = F.interpolate(
        cropped,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.clamp(0.0, 1.0)

    # Per-sample horizontal flip
    flipped = torch.flip(resized, dims=[-1])
    flip_mask = flip.view(B, 1, 1, 1).expand_as(resized)
    out = torch.where(flip_mask, flipped, resized)

    return out


def apply_action_augmentation(
    actions: torch.Tensor,
    params: dict[str, Any],
    horizontal_axis: int,
) -> torch.Tensor:
    """Negate horizontal action component for flipped samples. actions: (B, action_dim)."""
    flip = params["flip"]
    actions = actions.clone()
    actions[flip, horizontal_axis] = -actions[flip, horizontal_axis]
    return actions


def apply_geometric_augmentation(
    batch_state: dict[str, torch.Tensor],
    batch_next_state: dict[str, torch.Tensor],
    batch_actions: torch.Tensor,
    image_keys: list[str],
    config: GeometricAugmentationConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Apply flip + crop + resize to images and sync action for flipped samples.

    Uses the SAME per-sample params for all image keys (state and next_state)
    to ensure multi-camera consistency.
    """
    if not image_keys:
        return batch_state, batch_next_state, batch_actions

    first_key = image_keys[0]
    images = batch_state[first_key]
    B, C, H, W = images.shape
    device = images.device

    params = sample_augmentation_params(B, H, W, config, device)

    new_batch_state = dict(batch_state)
    new_batch_next_state = dict(batch_next_state)

    # Expand params for state + next_state (same params per sample for both)
    params_2b = {
        "flip": torch.cat([params["flip"], params["flip"]]),
        "crop_top": torch.cat([params["crop_top"], params["crop_top"]]),
        "crop_left": torch.cat([params["crop_left"], params["crop_left"]]),
        "crop_h": params["crop_h"],
        "crop_w": params["crop_w"],
    }

    for key in image_keys:
        if key not in batch_state:
            continue
        state_imgs = batch_state[key]
        next_imgs = batch_next_state[key]
        all_imgs = torch.cat([state_imgs, next_imgs], dim=0)
        augmented = apply_image_augmentation(all_imgs, params_2b, config.target_size)
        new_batch_state[key] = augmented[:B]
        new_batch_next_state[key] = augmented[B:]

    new_actions = apply_action_augmentation(
        batch_actions, params, config.horizontal_axis
    )

    return new_batch_state, new_batch_next_state, new_actions
