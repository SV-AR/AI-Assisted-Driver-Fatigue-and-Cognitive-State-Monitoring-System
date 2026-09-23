"""
modules/head_pose.py
----------------------
MODULE 10: HEAD POSE ESTIMATION

Estimates the driver's 3D head orientation (pitch, yaw, roll) from
6 correspondence points between a generic 3D face model and their
2D projections in the current frame (from the face mesh), using
OpenCV's solvePnP (Perspective-n-Point) algorithm.

Algorithm
---------
1. Take 6 known 3D model points of a generic face (nose tip, chin,
   eye corners, mouth corners), defined in an arbitrary face-centric
   3D coordinate system (config.HeadPoseConfig.MODEL_POINTS_3D).
2. Take the corresponding 2D pixel locations of those same 6 points
   from the current frame's face mesh.
3. Estimate a camera matrix from the frame dimensions (approximating
   focal length as the frame width -- a common simplification when
   the true camera calibration is unknown).
4. Run cv2.solvePnP to recover a rotation vector and translation
   vector that best explains how the 3D model would need to be
   rotated/translated to produce the observed 2D points.
5. Convert the rotation vector to a rotation matrix (cv2.Rodrigues),
   then decompose it into pitch/yaw/roll Euler angles.
6. Classify the head direction (Straight / Left / Right / Up / Down)
   by comparing yaw/pitch against configurable thresholds.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from config import HEAD_POSE_CONFIG
from logger.logger import get_logger
from utils.math_utils import rotation_matrix_to_euler_angles

logger = get_logger(__name__)

Point2D = Tuple[int, int]


@dataclass
class HeadPoseResult:
    """Structured output of head pose estimation for one frame."""

    pitch: float          # degrees; positive = looking down (sign depends on convention)
    yaw: float            # degrees; positive = turned to one side
    roll: float           # degrees; head tilt
    direction: str        # "STRAIGHT", "LEFT", "RIGHT", "UP", "DOWN"
    rotation_vector: np.ndarray
    translation_vector: np.ndarray


class HeadPoseEstimator:
    """Estimates 3D head pose from 2D face-mesh landmarks via solvePnP."""

    def __init__(
        self,
        yaw_threshold_deg: float = HEAD_POSE_CONFIG.YAW_THRESHOLD_DEG,
        pitch_up_threshold_deg: float = HEAD_POSE_CONFIG.PITCH_UP_THRESHOLD_DEG,
        pitch_down_threshold_deg: float = HEAD_POSE_CONFIG.PITCH_DOWN_THRESHOLD_DEG,
    ) -> None:
        self.yaw_threshold_deg = yaw_threshold_deg
        self.pitch_up_threshold_deg = pitch_up_threshold_deg
        self.pitch_down_threshold_deg = pitch_down_threshold_deg

        self.model_points_3d = np.array(HEAD_POSE_CONFIG.MODEL_POINTS_3D, dtype=np.float64)

        self._landmark_indices = (
            HEAD_POSE_CONFIG.NOSE_TIP_IDX,
            HEAD_POSE_CONFIG.CHIN_IDX,
            HEAD_POSE_CONFIG.LEFT_EYE_CORNER_IDX,
            HEAD_POSE_CONFIG.RIGHT_EYE_CORNER_IDX,
            HEAD_POSE_CONFIG.LEFT_MOUTH_CORNER_IDX,
            HEAD_POSE_CONFIG.RIGHT_MOUTH_CORNER_IDX,
        )

        logger.info("HeadPoseEstimator initialized with 6-point PnP model.")

    def estimate(
        self, mesh_points: Sequence[Point2D], frame_shape: Tuple[int, int]
    ) -> Optional[HeadPoseResult]:
        """Estimate head pose for the current frame.

        Parameters
        ----------
        mesh_points : Sequence[Point2D]
            Full list of pixel-space face mesh landmarks (Module 3 output).
        frame_shape : Tuple[int, int]
            (frame_height, frame_width) of the source frame, used to
            build an approximate camera intrinsics matrix.

        Returns
        -------
        Optional[HeadPoseResult]
            None if solvePnP fails to converge; otherwise the
            estimated pose + direction classification.
        """
        try:
            image_points = np.array(
                [mesh_points[i] for i in self._landmark_indices], dtype=np.float64
            )
        except IndexError as error:
            logger.error("Mesh point list too short for head pose landmarks: %s", error)
            return None

        frame_height, frame_width = frame_shape
        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.warning("solvePnP failed to converge for this frame.")
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)

        direction = self._classify_direction(pitch, yaw)

        return HeadPoseResult(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            direction=direction,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
        )

    def _classify_direction(self, pitch: float, yaw: float) -> str:
        """Classify head direction from pitch/yaw angles using configured thresholds.

        Note: exact sign conventions depend on the camera/model point
        setup; thresholds in config.py should be calibrated against
        your own webcam if directions appear inverted.
        """
        if yaw > self.yaw_threshold_deg:
            return "RIGHT"
        if yaw < -self.yaw_threshold_deg:
            return "LEFT"
        if pitch > self.pitch_down_threshold_deg:
            return "DOWN"
        if pitch < -self.pitch_up_threshold_deg:
            return "UP"
        return "STRAIGHT"
