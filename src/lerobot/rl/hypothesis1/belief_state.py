# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 1: Belief state for uncertainty - lightweight Bayesian-style update.
# Maintains EMA of uncertainty over time as a belief (posterior proxy).

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class BeliefState:
    """Maintains belief over policy uncertainty via exponential moving average.

    Acts as a lightweight Bayesian-style update: belief_t = α * belief_{t-1} + (1-α) * u_t.
    Optionally inflates by running std for epistemic uncertainty about uncertainty.
    """

    def __init__(self, ema_alpha: float = 0.9, use_variance_inflation: bool = False):
        self.ema_alpha = ema_alpha
        self.use_variance_inflation = use_variance_inflation
        self._belief: float | None = None
        self._m2: float = 0.0  # For Welford's online variance
        self._count: int = 0
        self._recent: deque[float] = deque(maxlen=20)

    def reset(self) -> None:
        """Reset episode-level state."""
        self._belief = None
        self._m2 = 0.0
        self._count = 0
        self._recent.clear()

    def update(self, uncertainty: float) -> None:
        """Update belief with new uncertainty observation."""
        self._recent.append(uncertainty)
        if self._belief is None:
            self._belief = uncertainty
            self._m2 = 0.0
            self._count = 1
        else:
            self._belief = self.ema_alpha * self._belief + (1 - self.ema_alpha) * uncertainty
            self._count += 1
            delta = uncertainty - self._belief
            self._m2 += (1 - self.ema_alpha) * (delta * delta)

    def get_belief_uncertainty(self) -> float:
        """Return belief (EMA) uncertainty, optionally inflated by running std."""
        if self._belief is None:
            return 0.0
        if not self.use_variance_inflation or self._count < 2:
            return max(0.0, min(1.0, self._belief))
        # Welford's variance: var ≈ m2 / (count - 1)
        var = self._m2 / max(1, self._count - 1)
        std = math.sqrt(max(0, var))
        # Inflate: when we're uncertain about our uncertainty, report higher
        inflated = self._belief + 0.5 * std
        return max(0.0, min(1.0, inflated))
