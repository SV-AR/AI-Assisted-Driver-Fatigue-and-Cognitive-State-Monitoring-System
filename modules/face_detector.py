"""
modules/face_detector.py
-------------------------
MODULE 2: FACE DETECTION

Uses MediaPipe's lightweight BlazeFace-based Face Detection model to
quickly determine WHETHER a face is present in the frame and WHERE
(bounding box + confidence). This is intentionally a cheap, fast
"is there a driver in front of the camera at all" check, separate
from the much heavier 468-point Face Mesh (Module 3), which is only
worth running once we know a face exists.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import FACE_DETECTION_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FaceBoundingBox:
    """A single detected face's bounding box and confidence."""

    x: int
    y: int
    width: int
    height: int
    confidence: float

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height) as a plain tuple, e.g. for drawing."""
        return self.x, self.y, self.width, self.height


class FaceDetector:
    """Object-oriented wrapper around MediaPipe Face Detection.

    Usage
    -----
    >>> detector = FaceDetector()
    >>> faces = detector.detect(frame)
    >>> if faces:
    ...     print(f"Found {len(faces)} face(s), best confidence: {faces[0].confidence}")
    >>> detector.close()
    """

    def __init__(
        self,
        model_selection: int = FACE_DETECTION_CONFIG.MODEL_SELECTION,
        min_detection_confidence: float = FACE_DETECTION_CONFIG.MIN_DETECTION_CONFIDENCE,
    ) -> None:
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )
        logger.info(
            "FaceDetector initialized (model_selection=%s, min_confidence=%s)",
            model_selection, min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBoundingBox]:
        """Detect all faces present in a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            A BGR image, as produced by `CameraStream.read()`.

        Returns
        -------
        List[FaceBoundingBox]
            Zero or more detected faces, sorted by confidence
            descending. Empty list if no face is found.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Received an empty frame; skipping face detection.")
            return []

        frame_height, frame_width = frame_bgr.shape[:2]

        # MediaPipe expects RGB input, OpenCV gives us BGR.
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # perf: mark as read-only for MediaPipe
        results = self._detector.process(rgb_frame)

        faces: List[FaceBoundingBox] = []

        if results.detections:
            for detection in results.detections:
                relative_box = detection.location_data.relative_bounding_box
                x = int(relative_box.xmin * frame_width)
                y = int(relative_box.ymin * frame_height)
                w = int(relative_box.width * frame_width)
                h = int(relative_box.height * frame_height)

                # Clip to frame boundaries; MediaPipe can return
                # slightly out-of-frame boxes near the edges.
                x = max(0, x)
                y = max(0, y)
                w = max(0, min(w, frame_width - x))
                h = max(0, min(h, frame_height - y))

                confidence = float(detection.score[0]) if detection.score else 0.0

                faces.append(FaceBoundingBox(x=x, y=y, width=w, height=h, confidence=confidence))

        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces

    def close(self) -> None:
        """Release MediaPipe's internal resources. Call when shutting down."""
        self._detector.close()
        logger.info("FaceDetector resources released.")

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
