# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 1: Intervention debounce to fix Space double-trigger and accidental Enter/Backspace.

from __future__ import annotations

import logging
from typing import Any

from lerobot.configs.hil import InterventionDebounceConfig
from lerobot.teleoperators.utils import TeleopEvents

logger = logging.getLogger(__name__)


class InterventionDebouncer:
    """Debounces intervention and success/terminate to fix Space double-trigger and accidental keys."""

    def __init__(self, config: InterventionDebounceConfig | None = None):
        self.config = config or InterventionDebounceConfig(enabled=False)
        self._intervention_streak: int = 0
        self._in_intervention: bool = False
        self._steps_in_intervention: int = 0
        self._success_streak: int = 0
        self._terminate_streak: int = 0

    def reset(self) -> None:
        """Reset episode-level state."""
        self._intervention_streak = 0
        self._in_intervention = False
        self._steps_in_intervention = 0
        self._success_streak = 0
        self._terminate_streak = 0

    def process_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Process raw info from env.step(), return debounced info. Modifies in-place and returns."""
        if not self.config.enabled:
            return info

        raw_is_intervention = info.get(TeleopEvents.IS_INTERVENTION.value, False)
        raw_success = info.get(TeleopEvents.SUCCESS.value, False)
        raw_terminate = info.get(TeleopEvents.TERMINATE_EPISODE.value, False)

        # Intervention debounce: require confirm_steps to enter, min_steps before exit
        if raw_is_intervention:
            self._intervention_streak += 1
            if self._intervention_streak >= self.config.intervention_confirm_steps:
                if not self._in_intervention:
                    self._in_intervention = True
                    self._steps_in_intervention = 0
                    logger.debug("[DEBOUNCE] Entered intervention mode")
        else:
            self._intervention_streak = 0
            if self._in_intervention:
                self._steps_in_intervention += 1
                if self._steps_in_intervention >= self.config.min_intervention_steps:
                    self._in_intervention = False
                    logger.debug("[DEBOUNCE] Exited intervention mode after %d steps", self._steps_in_intervention)
                else:
                    # Force keep intervention during min period
                    info[TeleopEvents.IS_INTERVENTION.value] = True
                    return info

        info[TeleopEvents.IS_INTERVENTION.value] = self._in_intervention

        # Success/Terminate debounce: only when in intervention (avoid accidental Enter/Backspace)
        # When NOT in intervention, use raw values so env physics (object at height, etc.) ends episode immediately
        if self._in_intervention:
            if raw_success:
                self._success_streak += 1
            else:
                self._success_streak = 0
            if self._success_streak < self.config.success_confirm_steps:
                info[TeleopEvents.SUCCESS.value] = False

            if raw_terminate:
                self._terminate_streak += 1
            else:
                self._terminate_streak = 0
            if self._terminate_streak < self.config.terminate_confirm_steps:
                info[TeleopEvents.TERMINATE_EPISODE.value] = False
        else:
            self._success_streak = 1 if raw_success else 0
            self._terminate_streak = 1 if raw_terminate else 0

        return info
