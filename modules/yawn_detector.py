"""
modules/yawn_detector.py
---------------------------
MODULE 9: YAWN DETECTION

Mirrors the design of BlinkDetector (Module 6), but for the mouth:
tracks consecutive high-MAR frames to confirm a genuine yawn (as
opposed to talking or a brief mouth movement), counts total yawns,
and maintains a rolling yawn rate for the Fatigue Score Engine.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

from config import MAR_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class YawnState:
    """Snapshot of the yawn detector's output for the current frame."""

    mar: float
    is_yawning: bool          # True while mouth is currently held open past threshold
    is_yawn_event: bool       # True on the frame a full yawn is confirmed complete
    yawn_count: int
    yawn_rate_per_min: float


class YawnDetector:
    """Stateful yawn detector driven by MAR values."""

    def __init__(
        self,
        mar_threshold: float = MAR_CONFIG.MAR_THRESHOLD,
        yawn_consec_frames: int = MAR_CONFIG.YAWN_CONSEC_FRAMES,
        yawn_rate_window_seconds: float = MAR_CONFIG.YAWN_RATE_WINDOW_SECONDS,
    ) -> None:
        self.mar_threshold = mar_threshold
        self.yawn_consec_frames = yawn_consec_frames
        self.yawn_rate_window_seconds = yawn_rate_window_seconds

        self._open_frame_counter: int = 0
        self._yawn_count: int = 0
        self._yawn_timestamps: Deque[float] = deque()
        self._yawn_confirmed_this_episode: bool = False

        logger.info(
            "YawnDetector initialized (mar_threshold=%s, consec_frames=%s)",
            mar_threshold, yawn_consec_frames,
        )

    def update(self, mar: float) -> YawnState:
        """Feed the current frame's MAR value and get the updated yawn state.

        Parameters
        ----------
        mar : float
            The current Mouth Aspect Ratio (see MARCalculator).

        Returns
        -------
        YawnState
            Structured snapshot of yawning status for this frame.
        """
        is_yawn_event = False
        mouth_open_this_frame = mar > self.mar_threshold

        if mouth_open_this_frame:
            self._open_frame_counter += 1

            if (
                self._open_frame_counter >= self.yawn_consec_frames
                and not self._yawn_confirmed_this_episode
            ):
                self._yawn_count += 1
                self._yawn_timestamps.append(time.time())
                self._yawn_confirmed_this_episode = True
                is_yawn_event = True
                logger.info("Yawn #%s confirmed.", self._yawn_count)
        else:
            self._open_frame_counter = 0
            self._yawn_confirmed_this_episode = False

        self._prune_old_yawns()
        yawn_rate = self._current_yawn_rate()

        is_yawning = self._open_frame_counter >= self.yawn_consec_frames

        return YawnState(
            mar=mar,
            is_yawning=is_yawning,
            is_yawn_event=is_yawn_event,
            yawn_count=self._yawn_count,
            yawn_rate_per_min=yawn_rate,
        )

    def _prune_old_yawns(self) -> None:
        cutoff = time.time() - self.yawn_rate_window_seconds
        while self._yawn_timestamps and self._yawn_timestamps[0] < cutoff:
            self._yawn_timestamps.popleft()

    def _current_yawn_rate(self) -> float:
        count_in_window = len(self._yawn_timestamps)
        minutes = self.yawn_rate_window_seconds / 60.0
        if minutes <= 0:
            return 0.0
        return round(count_in_window / minutes, 2)

    def reset(self) -> None:
        """Reset all counters (e.g. when starting a new monitoring session)."""
        self._open_frame_counter = 0
        self._yawn_count = 0
        self._yawn_timestamps.clear()
        self._yawn_confirmed_this_episode = False
        logger.info("YawnDetector state reset.")
