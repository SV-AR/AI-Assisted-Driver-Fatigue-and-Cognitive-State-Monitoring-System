"""
modules/camera.py
------------------
MODULE 1: CAMERA

Responsible for capturing a live video stream from the laptop's
webcam and exposing it to the rest of the pipeline as a simple,
reusable, object-oriented interface.

This module deliberately does ONLY ONE job: reliable frame
acquisition. Face detection, landmark extraction, EAR/MAR
calculation, etc. are handled by later modules that will consume
frames produced here. This separation of concerns keeps the system
modular, testable, and scalable.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CAMERA_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)


class CameraError(Exception):
    """Raised when the camera cannot be opened or read from."""
    pass


class CameraStream:
    """Object-oriented wrapper around OpenCV's VideoCapture.

    This class encapsulates:
      - Opening/closing the webcam device.
      - Applying requested resolution/FPS settings.
      - Reading frames with error handling and automatic recovery
        tracking (counting consecutive failed reads).
      - Optional horizontal mirroring.
      - Real-time FPS measurement.

    Usage
    -----
    >>> cam = CameraStream()
    >>> cam.start()
    >>> ret, frame = cam.read()
    >>> cam.stop()

    Or, as a context manager:
    >>> with CameraStream() as cam:
    ...     ret, frame = cam.read()
    """

    def __init__(
        self,
        device_index: int = CAMERA_CONFIG.DEVICE_INDEX,
        frame_width: int = CAMERA_CONFIG.FRAME_WIDTH,
        frame_height: int = CAMERA_CONFIG.FRAME_HEIGHT,
        target_fps: int = CAMERA_CONFIG.TARGET_FPS,
        flip_horizontal: bool = CAMERA_CONFIG.FLIP_HORIZONTAL,
    ) -> None:
        self.device_index: int = device_index
        self.frame_width: int = frame_width
        self.frame_height: int = frame_height
        self.target_fps: int = target_fps
        self.flip_horizontal: bool = flip_horizontal

        self._capture: Optional[cv2.VideoCapture] = None
        self._is_running: bool = False
        self._consecutive_failed_reads: int = 0

        # FPS measurement state
        self._prev_frame_time: float = 0.0
        self._current_fps: float = 0.0

        logger.debug(
            "CameraStream initialized (device_index=%s, resolution=%sx%s, target_fps=%s)",
            self.device_index, self.frame_width, self.frame_height, self.target_fps,
        )

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the webcam device and apply the configured settings.

        Raises
        ------
        CameraError
            If the camera device cannot be opened at all.
        """
        backend = cv2.CAP_ANY if CAMERA_CONFIG.USE_AUTO_BACKEND else cv2.CAP_DSHOW

        logger.info("Attempting to open camera device index %s...", self.device_index)
        self._capture = cv2.VideoCapture(self.device_index, backend)

        if not self._capture.isOpened():
            logger.error("Could not open camera at device index %s", self.device_index)
            raise CameraError(
                f"Unable to open webcam at device index {self.device_index}. "
                f"Check that no other application is using the camera, "
                f"and that the correct device index is set in config.py."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_width = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Camera opened successfully. Requested %sx%s@%sfps -> Actual %sx%s@%sfps",
            self.frame_width, self.frame_height, self.target_fps,
            int(actual_width), int(actual_height), actual_fps,
        )

        self._is_running = True
        self._consecutive_failed_reads = 0
        self._prev_frame_time = time.time()

    def stop(self) -> None:
        """Release the camera device and clean up resources."""
        if self._capture is not None:
            self._capture.release()
            logger.info("Camera device released.")
        self._is_running = False

    # Support "with CameraStream() as cam:" usage
    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.

        Returns
        -------
        (success, frame) : Tuple[bool, Optional[np.ndarray]]
            success is True if a frame was read correctly.
            frame is a BGR numpy array, or None on failure.

        Raises
        ------
        CameraError
            If too many consecutive frame reads fail, indicating the
            camera has likely been disconnected.
        """
        if not self._is_running or self._capture is None:
            raise CameraError("Camera is not started. Call start() first.")

        success, frame = self._capture.read()

        if not success or frame is None:
            self._consecutive_failed_reads += 1
            logger.warning(
                "Failed to read frame (%s/%s consecutive failures).",
                self._consecutive_failed_reads, CAMERA_CONFIG.MAX_FAILED_READS,
            )

            if self._consecutive_failed_reads >= CAMERA_CONFIG.MAX_FAILED_READS:
                logger.error("Camera appears disconnected. Too many failed reads.")
                raise CameraError(
                    "Camera stopped responding. It may have been disconnected "
                    "or is being used by another application."
                )
            return False, None

        # Reset failure counter on a successful read.
        self._consecutive_failed_reads = 0

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        self._update_fps()

        return True, frame

    # ------------------------------------------------------------------
    # FPS measurement
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        """Update the internal, smoothed FPS estimate based on the
        time elapsed since the previous frame."""
        current_time = time.time()
        elapsed = current_time - self._prev_frame_time
        self._prev_frame_time = current_time

        if elapsed > 0:
            instantaneous_fps = 1.0 / elapsed
            # Exponential moving average for a stable, non-jittery reading.
            smoothing_factor = 0.9
            self._current_fps = (
                smoothing_factor * self._current_fps
                + (1 - smoothing_factor) * instantaneous_fps
            )

    @property
    def fps(self) -> float:
        """Current smoothed frames-per-second estimate."""
        return round(self._current_fps, 1)

    @property
    def is_running(self) -> bool:
        """Whether the camera is currently active."""
        return self._is_running
