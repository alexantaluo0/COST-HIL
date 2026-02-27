# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 2: Weighted sampling and batch weights for heterogeneous intervention data.

from __future__ import annotations

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lerobot.configs.hil import WeightedInterventionConfig


def compute_batch_weights(
    batch: dict,
    config: "WeightedInterventionConfig | None",
    device: torch.device | str,
) -> torch.Tensor | None:
    """Compute per-sample weights for Hypothesis 2 weighted loss.

    When config is None or disabled, returns None (use uniform weights).
    Otherwise extracts intervention_quality from complementary_info, clamps to
    min_intervention_weight, and normalizes.

    Args:
        batch: BatchTransition with complementary_info
        config: WeightedInterventionConfig or None
        device: Target device for weight tensor

    Returns:
        Weights tensor of shape (batch_size,) or None for uniform
    """
    if config is None or not config.enabled:
        return None

    comp = batch.get("complementary_info")
    if comp is None:
        return None

    quality = comp.get(config.intervention_quality_key)
    if quality is None:
        return None

    if isinstance(quality, torch.Tensor):
        w = quality.float().to(device)
    else:
        batch_size = batch["reward"].shape[0]
        w = torch.full((batch_size,), float(quality), device=device, dtype=torch.float32)

    # Clamp to min weight for stability
    w = torch.clamp(w, min=config.min_intervention_weight)

    # Normalize so mean weight = 1 (preserves loss scale)
    w = w / (w.mean() + 1e-8)
    return w
