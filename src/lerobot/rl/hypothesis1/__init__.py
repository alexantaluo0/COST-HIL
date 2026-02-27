# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 1: Optimal Stopping Theory - Adaptive Intervention Scheduling
# This module is kept separate from Hypothesis 2 for ablation experiments.

import torch

from lerobot.configs.hil import InterventionDebounceConfig

from .intervention_debounce import InterventionDebouncer
from .intervention_scheduler import InterventionScheduler
from .uncertainty_estimator import estimate_actor_entropy_uncertainty

from .intervention_ui import InterventionUIPrompt
from .value_estimator import estimate_value_no_intervention

__all__ = [
    "InterventionDebouncer",
    "InterventionDebounceConfig",
    "InterventionScheduler",
    "InterventionUIPrompt",
    "estimate_actor_entropy_uncertainty",
    "estimate_value_no_intervention",
    "apply_intervention_cost_to_batch",
]


def apply_intervention_cost_to_batch(batch, intervention_cost: float, is_intervention_key: str = "is_intervention"):
    """Apply intervention penalty to reward when Hypothesis 1 is enabled.

    Modifies batch in-place: reward' = reward + intervention_cost * I(is_intervention).
    intervention_cost is negative (e.g. -0.01).

    Args:
        batch: BatchTransition with keys "reward", "complementary_info"
        intervention_cost: Negative penalty per intervention step
        is_intervention_key: Key in complementary_info for intervention flag
    """
    if intervention_cost is None or intervention_cost == 0:
        return
    comp = batch.get("complementary_info")
    if comp is None:
        return
    is_interv = comp.get(is_intervention_key)
    if is_interv is None:
        return
    # is_interv can be bool tensor (batch_size,) or scalar
    reward = batch["reward"]
    device, dtype = reward.device, reward.dtype
    if hasattr(is_interv, "float"):
        mask = is_interv.float().to(device)
    else:
        mask = torch.tensor(float(is_interv), device=device, dtype=dtype)

    # Handle shape mismatch: concatenated batch (online+offline) may have different
    # complementary_info structure; pad mask to match reward's batch size
    if mask.shape[0] != reward.shape[0]:
        if mask.numel() == 1:
            mask = mask.expand(reward.shape[0])
        else:
            # Pad with zeros; mask typically comes from offline part (last N samples)
            mask_full = torch.zeros(reward.shape[0], device=device, dtype=dtype)
            n = min(mask.shape[0], reward.shape[0])
            mask_full[-n:] = mask[-n:].to(device)  # align to end (offline batch)
            mask = mask_full

    batch["reward"] = reward + intervention_cost * mask
