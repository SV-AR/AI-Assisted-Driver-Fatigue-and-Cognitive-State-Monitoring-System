"""
modules/blink_detector.py
---------------------------
MODULE 6: BLINK DETECTION

Consumes the per-frame average EAR value and runs a small state
machine to distinguish:
  - Normal, brief blinks (a few consecutive low-EAR frames)
  - Prolonged eye closure / microsleep (many consecutive low-EAR
    frames -- a strong fatigue indicator)

It also maintains a rolling blink-rate (blinks per minute), which
feeds into the Fatigue Score Engine (Module 11): both an
abnormally LOW blink rate (staring / microsleep) and an abnormally
HIGH blink rate (eye strain) can indicate reduced alertness.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

from config import EAR_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BlinkState:
    """Snapshot of the blink detector's output for the current frame."""

    ear: float
    eye_status: str          # "OPEN" or "CLOSED"
    is_blink_event: bool     # True on the frame a blink is confirmed
    blink_count: int         # total blinks since start
    blink_rate_per_min: float
    is_prolonged_closure: bool  # True while eyes have been closed too long


class BlinkDetector:
    """Stateful blink / prolonged-closure detector driven by EAR values."""

    def __init__(
        self,
        ear_threshold: float = EAR_CONFIG.EAR_THRESHOLD,
        blink_consec_frames: int = EAR_CONFIG.BLINK_CONSEC_FRAMES,
        prolonged_closure_consec_frames: int = EAR_CONFIG.PROLONGED_CLOSURE_CONSEC_FRAMES,
        blink_rate_window_seconds: float = EAR_CONFIG.BLINK_RATE_WINDOW_SECONDS,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.blink_consec_frames = blink_consec_frames
        self.prolonged_closure_consec_frames = prolonged_closure_consec_frames
        self.blink_rate_window_seconds = blink_rate_window_seconds

        self._closed_frame_counter: int = 0
        self._blink_count: int = 0
        self._blink_timestamps: Deque[float] = deque()
        self._prolonged_closure_active: bool = False

        logger.info(
            "BlinkDetector initialized (ear_threshold=%s, blink_frames=%s, prolonged_frames=%s)",
            ear_threshold, blink_consec_frames, prolonged_closure_consec_frames,
        )

    def update(self, ear: float) -> BlinkState:
        """Feed the current frame's EAR value and get the updated blink state.

        Parameters
        ----------
        ear : float
            The current average Eye Aspect Ratio (see EARCalculator).

        Returns
        -------
        BlinkState
            Structured snapshot of blink/closure status for this frame.
        """
        is_blink_event = False
        eyes_closed_this_frame = ear < self.ear_threshold

        if eyes_closed_this_frame:
            self._closed_frame_counter += 1
        else:
            # Eyes just opened after having been closed for a bit.
            if self._closed_frame_counter >= self.blink_consec_frames:
                self._blink_count += 1
                self._blink_timestamps.append(time.time())
                is_blink_event = True
                logger.debug("Blink #%s confirmed (was closed for %s frames).",
                             self._blink_count, self._closed_frame_counter)

            self._closed_frame_counter = 0
            self._prolonged_closure_active = False

        is_prolonged = self._closed_frame_counter >= self.prolonged_closure_consec_frames
        if is_prolonged and not self._prolonged_closure_active:
            logger.warning(
                "Prolonged eye closure detected (%s consecutive frames below EAR threshold).",
                self._closed_frame_counter,
            )
        self._prolonged_closure_active = is_prolonged

        self._prune_old_blinks()
        blink_rate = self._current_blink_rate()

        eye_status = "CLOSED" if eyes_closed_this_frame else "OPEN"

        return BlinkState(
            ear=ear,
            eye_status=eye_status,
            is_blink_event=is_blink_event,
            blink_count=self._blink_count,
            blink_rate_per_min=blink_rate,
            is_prolonged_closure=is_prolonged,
        )

    def _prune_old_blinks(self) -> None:
        """Drop blink timestamps that have fallen outside the rolling window."""
        cutoff = time.time() - self.blink_rate_window_seconds
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

    def _current_blink_rate(self) -> float:
        """Extrapolate blinks-in-window to a blinks-per-minute rate."""
        count_in_window = len(self._blink_timestamps)
        minutes = self.blink_rate_window_seconds / 60.0
        if minutes <= 0:
            return 0.0
        return round(count_in_window / minutes, 1)

    def reset(self) -> None:
        """Reset all counters (e.g. when starting a new monitoring session)."""
        self._closed_frame_counter = 0
        self._blink_count = 0
        self._blink_timestamps.clear()
        self._prolonged_closure_active = False
        logger.info("BlinkDetector state reset.")
