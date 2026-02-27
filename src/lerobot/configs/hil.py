# Copyright 2025 The HuggingFace Inc. team.
# HIL-RL config extensions for Hypothesis 1 and Hypothesis 2.
# Kept separate for ablation experiments.

from dataclasses import dataclass


@dataclass
class InterventionSchedulerConfig:
    """Hypothesis 1: Adaptive intervention scheduling (Optimal Stopping)."""

    enabled: bool = False
    uncertainty_threshold_high: float = 0.5
    uncertainty_threshold_low: float = 0.1
    min_steps_between_interventions: int = 10
    # Penalty per intervention step: negative value (e.g. -0.01) reduces reward
    intervention_cost: float = -0.01
    reward_window_size: int = 20
    baseline_reward_ratio: float = 0.5
    # Value-based trigger (optimal stopping): use Q(s,π(s)) for benefit > cost
    use_value_based_trigger: bool = False
    value_goal: float = 1.0  # Success return for benefit = uncertainty * max(0, goal - V)
    # Belief state: EMA of uncertainty as Bayesian-style posterior proxy
    use_belief_uncertainty: bool = False
    belief_ema_alpha: float = 0.9
    # Auto intervention: when suggested, try to force env to use teleop (if env supports it)
    use_auto_intervention: bool = False
    max_auto_intervention_steps: int = 50  # Safety: max consecutive auto-intervention steps
    # When suggested, block until human presses intervention key (space) before continuing
    wait_for_intervention_when_suggested: bool = False
    wait_timeout_steps: int = 300  # Max steps to wait (e.g. 30s at 10 FPS)
    # UI prompt when suggested_intervention is True
    show_ui_prompt: bool = True
    use_sound_prompt: bool = False
    # Training stage adaptive: suppress intervention when policy converged
    suppress_when_reward_high: bool = True
    episodic_reward_threshold: float = 0.9
    episodic_reward_window: int = 10
    intervention_stop_step: int | None = None
    intervention_ramp_down_step: int | None = None


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
