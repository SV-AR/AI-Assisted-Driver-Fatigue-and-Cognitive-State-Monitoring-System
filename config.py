"""
config.py
---------
Central configuration file for the Driver Cognitive State Monitoring System.

Design Rule:
No module in this project should hardcode values. Every tunable
parameter (camera index, resolution, thresholds, paths, etc.) must
live here so that the whole system can be re-tuned from a single
location.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# --------------------------------------------------------------------------
# MODULE 1: CAMERA CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    """Configuration parameters for the video capture module."""

    DEVICE_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    TARGET_FPS: int = 30
    MAX_FAILED_READS: int = 30
    FLIP_HORIZONTAL: bool = True
    USE_AUTO_BACKEND: bool = True


# --------------------------------------------------------------------------
# MODULE 2: FACE DETECTION CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceDetectionConfig:
    """Configuration for MediaPipe's lightweight Face Detection model."""

    # 0 = short-range model (best for faces within ~2m, e.g. selfie/driver cam)
    # 1 = full-range model (faces further away)
    MODEL_SELECTION: int = 0
    MIN_DETECTION_CONFIDENCE: float = 0.6


# --------------------------------------------------------------------------
# MODULE 3: FACE MESH CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceMeshConfig:
    """Configuration for MediaPipe's dense 468-point Face Mesh model."""

    STATIC_IMAGE_MODE: bool = False
    MAX_NUM_FACES: int = 1
    REFINE_LANDMARKS: bool = True  # adds iris landmarks (478 total points)
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5


# --------------------------------------------------------------------------
# MODULE 4/5: EYE LANDMARK + EAR CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EyeConfig:
    """MediaPipe Face Mesh landmark indices used for EAR computation.

    Each list follows the canonical 6-point ordering used by the
    Soukupova & Cech Eye Aspect Ratio formula:
        [p1 (outer corner), p2 (upper-outer), p3 (upper-inner),
         p4 (inner corner), p5 (lower-inner), p6 (lower-outer)]
    """

    LEFT_EYE_IDX: Tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE_IDX: Tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)


@dataclass(frozen=True)
class EARConfig:
    """Thresholds for eye-closure / blink logic based on EAR."""

    # EAR falls below this value when the eye is considered closed.
    EAR_THRESHOLD: float = 0.21

    # Number of consecutive closed-eye frames required before we count
    # a genuine "blink" (filters out single-frame detection noise).
    BLINK_CONSEC_FRAMES: int = 2

    # Number of consecutive closed-eye frames before we escalate a
    # blink into a "prolonged eye closure" (microsleep) event.
    PROLONGED_CLOSURE_CONSEC_FRAMES: int = 15

    # Size (in frames) of the rolling window used to compute PERCLOS
    # (percentage of eye closure over time) for the fatigue engine.
    PERCLOS_WINDOW_SIZE: int = 150  # ~5 seconds at 30 FPS

    # Rolling time window (seconds) for blink-rate (blinks/min) calc.
    BLINK_RATE_WINDOW_SECONDS: float = 60.0


# --------------------------------------------------------------------------
# MODULE 7/8/9: MOUTH LANDMARK + MAR + YAWN CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MouthConfig:
    """MediaPipe Face Mesh landmark indices used for MAR computation."""

    TOP_LIP_IDX: int = 13        # inner upper lip center
    BOTTOM_LIP_IDX: int = 14     # inner lower lip center
    LEFT_CORNER_IDX: int = 78    # inner left mouth corner
    RIGHT_CORNER_IDX: int = 308  # inner right mouth corner

    # Wider outer-lip set used for drawing the mouth contour on screen.
    CONTOUR_IDX: Tuple[int, ...] = (
        61, 291, 39, 181, 0, 17, 269, 405, 78, 308, 13, 14,
    )


@dataclass(frozen=True)
class MARConfig:
    """Thresholds for mouth-open / yawn logic based on MAR."""

    MAR_THRESHOLD: float = 0.55

    # Consecutive frames the mouth must stay open before a yawn is
    # confirmed (filters out talking / brief mouth movement).
    YAWN_CONSEC_FRAMES: int = 15

    # Rolling time window (seconds) for yawn-rate calculation.
    YAWN_RATE_WINDOW_SECONDS: float = 60.0


# --------------------------------------------------------------------------
# MODULE 10: HEAD POSE ESTIMATION CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadPoseConfig:
    """Configuration for solvePnP-based 3D head pose estimation."""

    # MediaPipe Face Mesh landmark indices corresponding to the 6
    # classical head-pose reference points.
    NOSE_TIP_IDX: int = 1
    CHIN_IDX: int = 152
    LEFT_EYE_CORNER_IDX: int = 33
    RIGHT_EYE_CORNER_IDX: int = 263
    LEFT_MOUTH_CORNER_IDX: int = 61
    RIGHT_MOUTH_CORNER_IDX: int = 291

    # Generic 3D face model coordinates (arbitrary units, mm-like),
    # centered on the nose tip. This is a widely used approximate
    # anthropometric model sufficient for coarse pose classification.
    MODEL_POINTS_3D: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    )

    # Angle thresholds (degrees) beyond which the head is classified
    # as looking away from "Straight".
    YAW_THRESHOLD_DEG: float = 15.0
    PITCH_UP_THRESHOLD_DEG: float = 12.0
    PITCH_DOWN_THRESHOLD_DEG: float = 12.0


# --------------------------------------------------------------------------
# MODULE 11: FATIGUE SCORE ENGINE CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FatigueConfig:
    """Weights and thresholds for the composite Fatigue Score (0-100)."""

    # Weighted contribution of each behavioural signal. Must sum to 1.0.
    WEIGHT_EYE_CLOSURE: float = 0.40   # PERCLOS-based
    WEIGHT_BLINK_RATE: float = 0.15
    WEIGHT_YAWN: float = 0.25
    WEIGHT_HEAD_POSE: float = 0.20

    # "Normal" alert blink rate range (blinks/min). Rates far outside
    # this band (too few = staring/microsleep, too many = strain)
    # increase the blink-rate fatigue sub-score.
    NORMAL_BLINK_RATE_MIN: float = 10.0
    NORMAL_BLINK_RATE_MAX: float = 20.0

    # Risk-level classification thresholds on the final 0-100 score.
    LOW_RISK_MAX: float = 40.0
    MEDIUM_RISK_MAX: float = 70.0
    # anything above MEDIUM_RISK_MAX is HIGH RISK

    # Smoothing factor (EMA) applied to the final fatigue score so it
    # doesn't jump erratically frame to frame.
    SCORE_SMOOTHING_ALPHA: float = 0.15


# --------------------------------------------------------------------------
# MODULE 12: ALARM SYSTEM CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AlarmConfig:
    """Configuration for the audible fatigue alarm."""

    ALARM_SOUND_PATH: str = "assets/alarm.wav"

    # Fatigue score (0-100) at/above which the alarm is triggered.
    FATIGUE_ALARM_THRESHOLD: float = 70.0

    # Minimum seconds between two alarm triggers, to avoid a
    # continuously blaring siren while still keeping the driver alert.
    ALARM_COOLDOWN_SECONDS: float = 5.0

    ALARM_VOLUME: float = 1.0  # 0.0 - 1.0


# --------------------------------------------------------------------------
# MODULE 13: DASHBOARD CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for the PyQt6 professional dashboard."""

    WINDOW_TITLE: str = "AI-Based Driver Cognitive State Monitoring System"
    WINDOW_MIN_WIDTH: int = 1200
    WINDOW_MIN_HEIGHT: int = 720
    VIDEO_PANEL_WIDTH: int = 800
    REFRESH_INTERVAL_MS: int = 15  # ~66 Hz UI refresh cap (actual FPS limited by camera)

    RISK_COLOR_LOW: str = "#2ecc71"      # green
    RISK_COLOR_MEDIUM: str = "#f39c12"   # amber
    RISK_COLOR_HIGH: str = "#e74c3c"     # red

    BACKGROUND_COLOR: str = "#1e1e2f"
    PANEL_COLOR: str = "#282a3a"
    TEXT_COLOR: str = "#eaeaea"
    ACCENT_COLOR: str = "#4d8dff"


# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoggingConfig:
    """Configuration parameters for the logging subsystem."""

    LOG_DIR: str = "logs"
    LOG_FILE_NAME: str = "driver_monitoring_system.log"
    LOG_LEVEL: str = "DEBUG"
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------
# DISPLAY / WINDOW CONFIGURATION (used for standalone module testing)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DisplayConfig:
    """Configuration for the raw OpenCV test/debug windows."""

    WINDOW_NAME: str = "Driver Monitoring System - Debug View"
    SHOW_FPS_OVERLAY: bool = True
    QUIT_KEY: str = "q"


# --------------------------------------------------------------------------
# Single shared instances (import these in other modules)
# --------------------------------------------------------------------------

CAMERA_CONFIG = CameraConfig()
FACE_DETECTION_CONFIG = FaceDetectionConfig()
FACE_MESH_CONFIG = FaceMeshConfig()
EYE_CONFIG = EyeConfig()
EAR_CONFIG = EARConfig()
MOUTH_CONFIG = MouthConfig()
MAR_CONFIG = MARConfig()
HEAD_POSE_CONFIG = HeadPoseConfig()
FATIGUE_CONFIG = FatigueConfig()
ALARM_CONFIG = AlarmConfig()
DASHBOARD_CONFIG = DashboardConfig()
LOGGING_CONFIG = LoggingConfig()
DISPLAY_CONFIG = DisplayConfig()
