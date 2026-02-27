# Copyright 2025 The HuggingFace Inc. team.
# Hypothesis 1: UI prompt when suggested_intervention is True.

from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default prompt text (gym_hil uses space, real robot uses RB button)
DEFAULT_PROMPT_TEXT = "请按空格键干预 / Press SPACE to intervene"
SOUND_DEBOUNCE_STEPS = 15  # Min steps between beeps to avoid spam


class InterventionUIPrompt:
    """Shows UI overlay and optional sound when intervention is suggested."""

    def __init__(self, show_ui: bool = True, use_sound: bool = False, prompt_text: str | None = None):
        self.show_ui = show_ui
        self.use_sound = use_sound
        self.prompt_text = prompt_text or DEFAULT_PROMPT_TEXT
        self._root = None
        self._label = None
        self._last_suggested = False
        self._steps_since_beep = SOUND_DEBOUNCE_STEPS

    def _ensure_window(self) -> bool:
        """Create overlay window if needed. Returns False if tkinter unavailable."""
        if self._root is not None:
            return True
        try:
            import tkinter as tk

            self._root = tk.Tk()
            self._root.title("HIL-SERL")
            self._root.attributes("-topmost", True)
            self._root.overrideredirect(True)
            self._root.geometry("+20+20")
            self._root.configure(bg="black")

            self._label = tk.Label(
                self._root,
                text="",
                font=("Arial", 14, "bold"),
                fg="yellow",
                bg="black",
                padx=12,
                pady=8,
            )
            self._label.pack()
            self._root.withdraw()  # Start hidden
            return True
        except Exception as e:
            logger.warning("Intervention UI overlay unavailable: %s", e)
            return False

    def update(self, suggested_intervention: bool) -> None:
        """Update UI and optionally play sound when suggested_intervention is True."""
        if not self.show_ui and not self.use_sound:
            return

        # Sound with debounce
        if self.use_sound and suggested_intervention:
            self._steps_since_beep += 1
            if self._steps_since_beep >= SOUND_DEBOUNCE_STEPS:
                self._play_beep()
                self._steps_since_beep = 0
        elif not suggested_intervention:
            self._steps_since_beep = SOUND_DEBOUNCE_STEPS

        # UI overlay
        if self.show_ui and suggested_intervention != self._last_suggested:
            self._last_suggested = suggested_intervention
            if self._ensure_window():
                if suggested_intervention:
                    self._label.config(text=self.prompt_text, fg="yellow", bg="darkred")
                    self._root.deiconify()
                else:
                    self._root.withdraw()

        if self._root is not None and self.show_ui:
            try:
                self._root.update()
            except Exception:
                pass

    def _play_beep(self) -> None:
        """Play a short beep (platform-dependent)."""
        try:
            if platform.system() == "Windows":
                import winsound

                winsound.Beep(800, 150)
            else:
                import sys

                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception as e:
            logger.debug("Beep failed: %s", e)

    def close(self) -> None:
        """Close the overlay window."""
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
            self._label = None
