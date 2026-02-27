# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 1: Uncertainty estimation for adaptive intervention scheduling.
# Uses actor entropy as a proxy for policy uncertainty (Actor-side, no Learner dependency).

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lerobot.policies.sac.modeling_sac import SACPolicy

logger = logging.getLogger(__name__)


def estimate_actor_entropy_uncertainty(
    policy: SACPolicy,
    batch: dict[str, torch.Tensor],
) -> float:
    """Estimate policy uncertainty from actor's action distribution entropy.

    Higher entropy (less negative log_prob) indicates higher uncertainty.
    Returns a normalized score in [0, 1] where 1 = most uncertain.

    Args:
        policy: SAC policy with actor that returns (action, log_prob, _)
        batch: Observation batch for policy input

    Returns:
        uncertainty_score: float in [0, 1], higher = more uncertain
    """
    with torch.no_grad():
        observations_features = None
        if policy.config.shared_encoder and policy.actor.encoder.has_images:
            observations_features = policy.actor.encoder.get_cached_image_features(batch)

        _, log_probs, _ = policy.actor(batch, observations_features)

        # log_prob is negative; more negative = more confident
        # Convert to uncertainty: -log_prob is "surprisal", higher = more uncertain
        # Normalize: use exp(log_prob) = prob, so -log_prob in [0, inf)
        # We use -log_prob as raw score, then normalize via sigmoid or min-max
        surprisal = -log_probs.mean().item()

        # Heuristic normalization: typical log_prob for SAC is in [-5, 0]
        # surprisal in [0, 5]. Map to [0, 1] with soft bound
        # uncertainty = 1 - exp(-surprisal / scale), scale=3 gives reasonable range
        scale = 3.0
        uncertainty_score = 1.0 - torch.exp(torch.tensor(-surprisal / scale)).item()
        uncertainty_score = max(0.0, min(1.0, uncertainty_score))

    return uncertainty_score
