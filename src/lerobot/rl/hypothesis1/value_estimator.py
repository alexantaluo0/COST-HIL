# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 1: Value function estimation for optimal stopping trigger.
# E[V_no_intervene] = Q(s, π(s)) - policy's expected return from current state.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from lerobot.policies.sac.modeling_sac import DISCRETE_DIMENSION_INDEX

if TYPE_CHECKING:
    from lerobot.policies.sac.modeling_sac import SACPolicy

logger = logging.getLogger(__name__)


def estimate_value_no_intervention(
    policy: SACPolicy,
    batch: dict[str, torch.Tensor],
) -> float | None:
    """Estimate E[V_no_intervene] = Q(s, π(s)) - policy's expected return from current state.

    Used for optimal stopping trigger: intervene when benefit > cost, where
    benefit ≈ E[V_intervene] - E[V_no_intervene]. For SAC, we use min over critics
    for conservative Q estimate.

    Args:
        policy: SAC policy with actor and critic
        batch: Observation batch for policy input

    Returns:
        value_estimate: float, or None if policy has no critic (e.g. non-SAC)
    """
    if not hasattr(policy, "critic_forward"):
        return None

    with torch.no_grad():
        observations_features = None
        if policy.config.shared_encoder and policy.actor.encoder.has_images:
            observations_features = policy.actor.encoder.get_cached_image_features(batch)

        actions, _, _ = policy.actor(batch, observations_features)

        # Critic expects continuous actions only (discrete dim excluded)
        if getattr(policy.config, "num_discrete_actions", None) is not None:
            actions_for_critic = actions[..., :DISCRETE_DIMENSION_INDEX]
        else:
            actions_for_critic = actions

        q_values = policy.critic_forward(
            observations=batch,
            actions=actions_for_critic,
            use_target=False,
            observation_features=observations_features,
        )

        # Conservative: min over critic ensemble
        if q_values.dim() > 1:
            min_q = q_values.min(dim=0).values.mean().item()
        else:
            min_q = q_values.mean().item()

    return min_q
