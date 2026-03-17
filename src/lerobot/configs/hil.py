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


@dataclass
class ImageAugmentationConfig:
    """Image augmentation switches for ablation experiments."""

    enable_flip: bool = True  # Random horizontal flip (with action sync)
    enable_crop: bool = True  # Random crop + resize
    crop_ratio_range: tuple[float, float] = (0.95, 1.0)  # keep 95%~100%, i.e. crop 0~5%


@dataclass
class BISOptimizationConfig:
    """v0.1.7 BIS-inspired data efficiency: PER + TIS + adaptive mixing.

    参数说明（详见 configs/bis_optimization_params.md）:
    - enabled: 是否启用 BIS 优化
    - per_alpha: 优先级指数，0=均匀采样，1=完全优先级
    - per_beta_start/end/frames: 重要性采样 beta 的 anneal 参数
    - enable_tis: 是否用 Q 值作为可靠性（TIS = 信息量 × 可靠性）
    - adaptive_mix_*: 动态 online/offline 混合比例（基于 loss EMA）
    - mix_update_interval: 每隔多少步更新混合比例
    """

    enabled: bool = False
    per_alpha: float = 0.3  # 优先级指数，0=均匀，1=完全优先级
    per_beta_start: float = 0.7  # IS 校正 beta 初始值
    per_beta_end: float = 1.0  # IS 校正 beta 终值
    per_beta_frames: int = 500000  # beta anneal 帧数
    enable_tis: bool = True  # TIS: 信息量(TD) × 可靠性(Q)
    adaptive_mix_ema_decay: float = 0.99  # loss EMA 衰减
    adaptive_mix_min_ratio: float = 0.2  # online 最小比例
    adaptive_mix_max_ratio: float = 0.8  # online 最大比例
    mix_update_interval: int = 500  # 混合比例更新间隔（步）
