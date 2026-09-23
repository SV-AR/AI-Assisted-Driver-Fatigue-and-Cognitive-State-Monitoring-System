"""
modules/alarm.py
------------------
MODULE 12: ALARM SYSTEM

Plays an audible alarm sound when the Fatigue Score (Module 11)
crosses a configurable threshold, with a cooldown so the alarm
doesn't blare continuously on every single frame above threshold.

Uses `pygame.mixer` for playback because it is cross-platform
(Windows/macOS/Linux), does not block the main thread while a sound
plays, and requires no native OS-level audio dependencies beyond the
`pygame` pip package.
"""

import os
import time
from typing import Optional

from config import ALARM_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)


class AlarmError(Exception):
    """Raised when the alarm system cannot initialize or load its sound."""
    pass


class AlarmSystem:
    """Threshold-triggered, cooldown-limited audible alarm.

    Usage
    -----
    >>> alarm = AlarmSystem()
    >>> alarm.check_and_trigger(fatigue_score=85.0)   # plays if >= threshold
    >>> alarm.stop()
    >>> alarm.shutdown()
    """

    def __init__(
        self,
        sound_path: str = ALARM_CONFIG.ALARM_SOUND_PATH,
        fatigue_threshold: float = ALARM_CONFIG.FATIGUE_ALARM_THRESHOLD,
        cooldown_seconds: float = ALARM_CONFIG.ALARM_COOLDOWN_SECONDS,
        volume: float = ALARM_CONFIG.ALARM_VOLUME,
    ) -> None:
        self.sound_path = sound_path
        self.fatigue_threshold = fatigue_threshold
        self.cooldown_seconds = cooldown_seconds
        self.volume = volume

        self._last_triggered_at: float = 0.0
        self._is_playing: bool = False
        self._mixer_ready: bool = False
        self._sound = None

        self._init_mixer()

    def _init_mixer(self) -> None:
        """Initialize pygame's mixer and load the alarm sound file."""
        try:
            import pygame  # imported here so the whole project doesn't
                            # hard-fail if pygame/audio backend is missing
                            # on a headless test machine.
            self._pygame = pygame

            if not os.path.exists(self.sound_path):
                raise AlarmError(f"Alarm sound file not found at '{self.sound_path}'.")

            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(self.sound_path)
            self._sound.set_volume(self.volume)
            self._mixer_ready = True
            logger.info("AlarmSystem initialized with sound '%s'.", self.sound_path)

        except Exception as error:  # noqa: BLE001 - deliberately broad: audio
            # backends fail in many environment-specific ways (no sound
            # card, missing driver, headless CI, etc.), and the whole
            # monitoring pipeline should keep running even if the
            # alarm cannot play audio -- it should just log loudly.
            self._mixer_ready = False
            logger.error(
                "AlarmSystem failed to initialize audio playback (%s). "
                "The system will continue running with a SILENT/visual-only alarm.",
                error,
            )

    def check_and_trigger(self, fatigue_score: float) -> bool:
        """Evaluate the current fatigue score and play the alarm if warranted.

        Parameters
        ----------
        fatigue_score : float
            The current 0-100 fatigue score from FatigueEngine.

        Returns
        -------
        bool
            True if the alarm was triggered on this call, False otherwise.
        """
        if fatigue_score < self.fatigue_threshold:
            return False

        now = time.time()
        if now - self._last_triggered_at < self.cooldown_seconds:
            return False  # still in cooldown; don't re-trigger yet

        self._last_triggered_at = now
        self._play()
        logger.warning(
            "ALARM TRIGGERED: fatigue_score=%.1f >= threshold=%.1f",
            fatigue_score, self.fatigue_threshold,
        )
        return True

    def _play(self) -> None:
        """Play the alarm sound (non-blocking) if audio is available."""
        if not self._mixer_ready or self._sound is None:
            logger.warning("Audio unavailable - alarm event logged only (no sound played).")
            return
        self._sound.play()
        self._is_playing = True

    def stop(self) -> None:
        """Immediately stop any currently playing alarm sound."""
        if self._mixer_ready and self._sound is not None:
            self._sound.stop()
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        """Whether the alarm sound is (believed to be) currently playing."""
        if self._mixer_ready and hasattr(self, "_pygame"):
            try:
                return bool(self._pygame.mixer.get_busy())
            except Exception:  # noqa: BLE001
                return self._is_playing
        return self._is_playing

    def shutdown(self) -> None:
        """Release audio resources. Call when the application exits."""
        if self._mixer_ready and hasattr(self, "_pygame"):
            try:
                self._pygame.mixer.quit()
            except Exception:  # noqa: BLE001
                pass
        logger.info("AlarmSystem shut down.")
