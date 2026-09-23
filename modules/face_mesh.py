"""
modules/face_mesh.py
---------------------
MODULE 3: FACE MESH

Runs MediaPipe's Face Mesh model to extract 468 (or 478 with iris
refinement) dense facial landmarks. This is the foundation that
every downstream module (eyes, mouth, head pose) builds on: they
simply index into the pixel-coordinate landmark list this module
produces, rather than each re-running their own detector.
"""

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import FACE_MESH_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)

Point2D = Tuple[int, int]


class FaceMeshDetector:
    """Object-oriented wrapper around MediaPipe Face Mesh.

    Usage
    -----
    >>> mesh = FaceMeshDetector()
    >>> landmarks = mesh.process(frame)
    >>> if landmarks is not None:
    ...     nose_tip = landmarks[1]
    >>> mesh.close()
    """

    def __init__(
        self,
        static_image_mode: bool = FACE_MESH_CONFIG.STATIC_IMAGE_MODE,
        max_num_faces: int = FACE_MESH_CONFIG.MAX_NUM_FACES,
        refine_landmarks: bool = FACE_MESH_CONFIG.REFINE_LANDMARKS,
        min_detection_confidence: float = FACE_MESH_CONFIG.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = FACE_MESH_CONFIG.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.num_landmarks_expected = 478 if refine_landmarks else 468
        logger.info(
            "FaceMeshDetector initialized (max_faces=%s, refine_landmarks=%s -> %s points expected)",
            max_num_faces, refine_landmarks, self.num_landmarks_expected,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[List[Point2D]]:
        """Run Face Mesh on a single BGR frame and return pixel-space landmarks.

        Parameters
        ----------
        frame_bgr : np.ndarray
            A BGR image, as produced by `CameraStream.read()`.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            A list of (x, y) pixel coordinates, one per landmark, for
            the first detected face. Returns None if no face mesh
            could be computed (e.g. no face in frame).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Received an empty frame; skipping face mesh.")
            return None

        frame_height, frame_width = frame_bgr.shape[:2]

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # max_num_faces is typically 1 for driver monitoring; take the first.
        face_landmarks = results.multi_face_landmarks[0]

        pixel_points: List[Point2D] = [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in face_landmarks.landmark
        ]
        return pixel_points

    def close(self) -> None:
        """Release MediaPipe's internal resources. Call when shutting down."""
        self._mesh.close()
        logger.info("FaceMeshDetector resources released.")

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
