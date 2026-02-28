# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 1: Adaptive intervention scheduling based on optimal stopping theory.
#
# Optimal stopping trigger (use_value_based_trigger=True):
#   Intervene when E[Benefit] > |c|, where
#   E[Benefit] ≈ σ * max(0, V_goal - V(s)),  σ = uncertainty, V = Q(s,π(s))
#   Derived from: E[V_intervene] - E[V_no_intervene] when policy uncertain and V low.
#   c = intervention_cost (negative), |c| = cost magnitude.

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from lerobot.configs.hil import InterventionSchedulerConfig

from .belief_state import BeliefState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InterventionScheduler:
    """Suggests human intervention based on policy uncertainty and recent rewards.

    Used for Hypothesis 1 (Optimal Stopping) - keeps logic separate for ablation.
    When enabled, outputs should_suggest_intervention; actual intervention still
    requires human (RB button) or can be wired to auto-mode if supported.
    """

    def __init__(self, config: InterventionSchedulerConfig | None = None):
        self.config = config or InterventionSchedulerConfig(enabled=False)
        self._steps_since_last_intervention = 0
        self._low_uncertainty_streak = 0
        self._value_ema: float | None = None  # EMA of V(s), auto-initialized on first observation
        use_belief = getattr(self.config, "use_belief_uncertainty", False)
        self._belief_state = (
            BeliefState(
                ema_alpha=getattr(self.config, "belief_ema_alpha", 0.9),
                use_variance_inflation=False,
            )
            if use_belief
            else None
        )

    def reset(self) -> None:
        """Reset episode-level state. Keeps _value_ema across episodes for stable baseline."""
        self._steps_since_last_intervention = 0
        self._low_uncertainty_streak = 0
        if self._belief_state is not None:
            self._belief_state.reset()

    def should_suggest_intervention(
        self,
        uncertainty_score: float,
        is_human_intervention: bool,
        value_estimate: float | None = None,
        interaction_step: int = 0,
    ) -> bool:
        """Decide whether to suggest intervention.

        Trigger modes:
        - Value-based (optimal stopping): benefit = uncertainty * max(0, V_goal - V) > |cost|
        - Heuristic: uncertainty_score >= adaptive threshold (linear ramp over training)

        Args:
            uncertainty_score: From uncertainty_estimator, in [0, 1]
            is_human_intervention: Whether human already intervened this step
            value_estimate: E[V_no_intervene] = Q(s,π(s)) when use_value_based_trigger
            interaction_step: Global interaction step for adaptive threshold

        Returns:
            True if intervention is suggested
        """
        if not self.config.enabled:
            return False

        # Belief update: use EMA of uncertainty when enabled
        if self._belief_state is not None:
            self._belief_state.update(uncertainty_score)
            uncertainty_score = self._belief_state.get_belief_uncertainty()

        self._steps_since_last_intervention += 1

        if is_human_intervention:
            self._steps_since_last_intervention = 0
            self._low_uncertainty_streak = 0
            return False  # Human already intervened, no need to suggest

        # Cooldown: avoid suggesting too frequently
        if self._steps_since_last_intervention < self.config.min_steps_between_interventions:
            return False

        # Low uncertainty streak: policy seems confident, don't suggest
        if uncertainty_score < self.config.uncertainty_threshold_low:
            self._low_uncertainty_streak += 1
            if self._low_uncertainty_streak >= self.config.min_steps_between_interventions:
                return False
        else:
            self._low_uncertainty_streak = 0

        # Update V(s) EMA for adaptive benefit baseline
        value_ema_alpha = getattr(self.config, "value_ema_alpha", 0.05)
        if value_estimate is not None:
            if self._value_ema is None:
                self._value_ema = value_estimate
            else:
                self._value_ema = (1 - value_ema_alpha) * self._value_ema + value_ema_alpha * value_estimate

        # Heuristic trigger: uncertainty >= adaptive threshold
        # Adaptive threshold: linear ramp from start to end over ramp_end_step
        ramp_end = getattr(self.config, "uncertainty_threshold_ramp_end_step", None)
        if ramp_end is not None and ramp_end > 0:
            start = getattr(self.config, "uncertainty_threshold_high_start", 0.2)
            end = getattr(self.config, "uncertainty_threshold_high_end", 0.7)
            t = min(1.0, interaction_step / ramp_end)
            effective_threshold = start + t * (end - start)
        else:
            effective_threshold = self.config.uncertainty_threshold_high
        if uncertainty_score < effective_threshold:
            return False

        # Value-based benefit gate: benefit = σ × max(0, V_ema - V(s))
        # Only intervene if benefit > |intervention_cost| (optimal stopping condition)
        cost_abs = abs(getattr(self.config, "intervention_cost", 0.0))
        if value_estimate is not None and self._value_ema is not None:
            benefit = uncertainty_score * max(0.0, self._value_ema - value_estimate)
            v_ema_str = f", V(s)={value_estimate:.3f}, V_ema={self._value_ema:.3f}, benefit={benefit:.3f}"
            if cost_abs > 0.0 and benefit <= cost_abs:
                return False
        else:
            benefit = None
            v_ema_str = ""

        logger.info(
            "[ACTOR] Intervention suggested: uncertainty=%.3f (threshold=%.2f)%s",
            uncertainty_score,
            effective_threshold,
            v_ema_str,
        )
        return True
