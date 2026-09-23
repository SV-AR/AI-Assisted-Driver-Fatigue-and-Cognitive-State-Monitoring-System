"""
modules/fatigue_engine.py
---------------------------
MODULE 11: FATIGUE SCORE ENGINE

Instead of a binary "sleeping / not sleeping" classification, this
module computes a continuous Fatigue Score in [0, 100] by combining
four weighted behavioural sub-scores:

  1. Eye Closure (PERCLOS)  - % of recent frames with eyes closed
  2. Blink Rate             - deviation from a normal blink-rate band
  3. Yawning                - recent yawn frequency / active yawning
  4. Head Pose              - sustained non-"straight" head orientation

    FatigueScore = W_eye * S_eye + W_blink * S_blink
                 + W_yawn * S_yawn + W_pose * S_pose

Each sub-score S_x is itself normalized to [0, 100] before weighting
(weights sum to 1.0, configured in config.FatigueConfig). The final
score is smoothed with an exponential moving average so it doesn't
jitter frame-to-frame, then mapped to a risk-level label.

This design is intentionally explainable: every sub-score can be
reported individually (as required by the IEEE-style output example
in the project brief: EAR, blink rate, yawning, head pose -> score).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

from config import EAR_CONFIG, FATIGUE_CONFIG
from logger.logger import get_logger
from utils.math_utils import clamp, exponential_moving_average

logger = get_logger(__name__)


@dataclass
class FatigueResult:
    """Structured output of the fatigue engine for the current frame."""

    fatigue_score: float          # 0-100, smoothed
    risk_level: str                # "LOW RISK", "MEDIUM RISK", "HIGH RISK"
    sub_scores: Dict[str, float]   # individual component scores, for transparency/debugging


class FatigueEngine:
    """Combines eye, blink, yawn, and head-pose signals into one score."""

    def __init__(
        self,
        weight_eye_closure: float = FATIGUE_CONFIG.WEIGHT_EYE_CLOSURE,
        weight_blink_rate: float = FATIGUE_CONFIG.WEIGHT_BLINK_RATE,
        weight_yawn: float = FATIGUE_CONFIG.WEIGHT_YAWN,
        weight_head_pose: float = FATIGUE_CONFIG.WEIGHT_HEAD_POSE,
        perclos_window_size: int = EAR_CONFIG.PERCLOS_WINDOW_SIZE,
        normal_blink_rate_min: float = FATIGUE_CONFIG.NORMAL_BLINK_RATE_MIN,
        normal_blink_rate_max: float = FATIGUE_CONFIG.NORMAL_BLINK_RATE_MAX,
        low_risk_max: float = FATIGUE_CONFIG.LOW_RISK_MAX,
        medium_risk_max: float = FATIGUE_CONFIG.MEDIUM_RISK_MAX,
        smoothing_alpha: float = FATIGUE_CONFIG.SCORE_SMOOTHING_ALPHA,
    ) -> None:
        self.weight_eye_closure = weight_eye_closure
        self.weight_blink_rate = weight_blink_rate
        self.weight_yawn = weight_yawn
        self.weight_head_pose = weight_head_pose

        self.normal_blink_rate_min = normal_blink_rate_min
        self.normal_blink_rate_max = normal_blink_rate_max
        self.low_risk_max = low_risk_max
        self.medium_risk_max = medium_risk_max
        self.smoothing_alpha = smoothing_alpha

        self._eye_closed_history: Deque[bool] = deque(maxlen=perclos_window_size)
        self._smoothed_score: float = 0.0

        total_weight = weight_eye_closure + weight_blink_rate + weight_yawn + weight_head_pose
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(
                "FatigueEngine weights sum to %.3f, not 1.0. Scores may exceed [0, 100].",
                total_weight,
            )

        logger.info(
            "FatigueEngine initialized (weights: eye=%.2f, blink=%.2f, yawn=%.2f, pose=%.2f)",
            weight_eye_closure, weight_blink_rate, weight_yawn, weight_head_pose,
        )

    def update(
        self,
        eye_status: str,
        blink_rate_per_min: float,
        is_yawning: bool,
        yawn_rate_per_min: float,
        head_direction: str,
        is_prolonged_closure: bool,
    ) -> FatigueResult:
        """Compute the updated fatigue score for the current frame.

        Parameters
        ----------
        eye_status : str
            "OPEN" or "CLOSED", from BlinkDetector.
        blink_rate_per_min : float
            Rolling blink rate, from BlinkDetector.
        is_yawning : bool
            Whether a yawn is currently in progress, from YawnDetector.
        yawn_rate_per_min : float
            Rolling yawn rate, from YawnDetector.
        head_direction : str
            "STRAIGHT", "LEFT", "RIGHT", "UP", or "DOWN", from HeadPoseEstimator.
        is_prolonged_closure : bool
            Whether eyes have been closed for a fatigue-indicative
            duration, from BlinkDetector.

        Returns
        -------
        FatigueResult
            Smoothed fatigue score, risk label, and component sub-scores.
        """
        self._eye_closed_history.append(eye_status == "CLOSED")

        s_eye = self._eye_closure_subscore(is_prolonged_closure)
        s_blink = self._blink_rate_subscore(blink_rate_per_min)
        s_yawn = self._yawn_subscore(is_yawning, yawn_rate_per_min)
        s_pose = self._head_pose_subscore(head_direction)

        raw_score = (
            self.weight_eye_closure * s_eye
            + self.weight_blink_rate * s_blink
            + self.weight_yawn * s_yawn
            + self.weight_head_pose * s_pose
        )
        raw_score = clamp(raw_score, 0.0, 100.0)

        self._smoothed_score = exponential_moving_average(
            previous=self._smoothed_score, new_value=raw_score, alpha=self.smoothing_alpha
        )
        final_score = round(clamp(self._smoothed_score, 0.0, 100.0), 1)

        risk_level = self._classify_risk(final_score)

        return FatigueResult(
            fatigue_score=final_score,
            risk_level=risk_level,
            sub_scores={
                "eye_closure": round(s_eye, 1),
                "blink_rate": round(s_blink, 1),
                "yawn": round(s_yawn, 1),
                "head_pose": round(s_pose, 1),
            },
        )

    # ------------------------------------------------------------------
    # Sub-score calculators (each normalized to 0-100)
    # ------------------------------------------------------------------

    def _eye_closure_subscore(self, is_prolonged_closure: bool) -> float:
        """PERCLOS: percentage of recent frames with eyes closed, 0-100.
        A prolonged closure event adds an extra penalty on top.
        """
        if not self._eye_closed_history:
            return 0.0
        perclos = 100.0 * sum(self._eye_closed_history) / len(self._eye_closed_history)
        if is_prolonged_closure:
            perclos = clamp(perclos + 25.0, 0.0, 100.0)
        return perclos

    def _blink_rate_subscore(self, blink_rate_per_min: float) -> float:
        """Score rises the further blink rate deviates from the normal band."""
        if self.normal_blink_rate_min <= blink_rate_per_min <= self.normal_blink_rate_max:
            return 0.0

        if blink_rate_per_min < self.normal_blink_rate_min:
            # Very low blink rate -> approaching microsleep / staring.
            deficit = self.normal_blink_rate_min - blink_rate_per_min
            return clamp(deficit * 15.0, 0.0, 100.0)

        excess = blink_rate_per_min - self.normal_blink_rate_max
        return clamp(excess * 5.0, 0.0, 100.0)

    def _yawn_subscore(self, is_yawning: bool, yawn_rate_per_min: float) -> float:
        """Active yawning contributes strongly; frequent recent yawns add up too."""
        score = 40.0 if is_yawning else 0.0
        score += clamp(yawn_rate_per_min * 20.0, 0.0, 60.0)
        return clamp(score, 0.0, 100.0)

    def _head_pose_subscore(self, head_direction: str) -> float:
        """Any non-"STRAIGHT" direction indicates distraction/drowsiness risk."""
        mapping = {
            "STRAIGHT": 0.0,
            "LEFT": 40.0,
            "RIGHT": 40.0,
            "UP": 50.0,
            "DOWN": 70.0,  # looking down is the strongest drowsiness signal
        }
        return mapping.get(head_direction.upper(), 0.0)

    def _classify_risk(self, score: float) -> str:
        if score <= self.low_risk_max:
            return "LOW RISK"
        if score <= self.medium_risk_max:
            return "MEDIUM RISK"
        return "HIGH RISK"

    def reset(self) -> None:
        """Reset all rolling state (e.g. for a new monitoring session)."""
        self._eye_closed_history.clear()
        self._smoothed_score = 0.0
        logger.info("FatigueEngine state reset.")
