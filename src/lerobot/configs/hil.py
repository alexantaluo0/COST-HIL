# Copyright 2025 The HuggingFace Inc. team.
# HIL-RL config extensions for Hypothesis 1 and Hypothesis 2.
# Kept separate for ablation experiments.

from dataclasses import dataclass


@dataclass
class InterventionSchedulerConfig:
    """Hypothesis 1: Adaptive intervention scheduling (Optimal Stopping)."""

    enabled: bool = False
    uncertainty_threshold_high: float = 0.5
    # Adaptive uncertainty_threshold_high: linear ramp from start to end
    uncertainty_threshold_high_start: float = 0.2  # Early training
    uncertainty_threshold_high_end: float = 0.7  # Late training
    uncertainty_threshold_ramp_end_step: int | None = None  # Step to reach end; None = use fixed uncertainty_threshold_high
    uncertainty_threshold_low: float = 0.1
    min_steps_between_interventions: int = 10
    # Value-based benefit: benefit = σ × max(0, V_ema - V(s)), where V_ema is running mean of Q values
    value_ema_alpha: float = 0.05  # EMA smoothing for V(s) baseline (slow to track long-term trend)
    # Optimal stopping cost: intervene only when benefit > |intervention_cost|
    # Set to 0.0 to disable (trigger on uncertainty threshold alone)
    intervention_cost: float = 0.0
    # Belief state: EMA of uncertainty as Bayesian-style posterior proxy
    use_belief_uncertainty: bool = False
    belief_ema_alpha: float = 0.9
    # Auto intervention: when suggested, try to force env to use teleop (if env supports it)
    use_auto_intervention: bool = False
    max_auto_intervention_steps: int = 50  # Safety: max consecutive auto-intervention steps
    # When suggested, block until human presses intervention key (space) before continuing
    wait_for_intervention_when_suggested: bool = False
    # UI prompt when suggested_intervention is True
    show_ui_prompt: bool = True
    use_sound_prompt: bool = False


@dataclass
class InterventionDebounceConfig:
    """Intervention debounce: fix Space double-trigger and accidental Enter/Backspace."""

    enabled: bool = True
    min_intervention_steps: int = 15  # Min steps to stay in intervention (avoid key repeat toggle off)
    intervention_confirm_steps: int = 2  # Consecutive steps with is_intervention before entering
    success_confirm_steps: int = 2  # Consecutive steps with success before terminating
    terminate_confirm_steps: int = 2  # Consecutive steps with terminate before terminating


@dataclass
class WeightedInterventionConfig:
    """Hypothesis 2: Robust MDP - heterogeneous intervention data fusion."""

    enabled: bool = False
    intervention_quality_key: str = "intervention_quality"
    min_intervention_weight: float = 0.1
