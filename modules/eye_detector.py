"""
modules/eye_detector.py
------------------------
MODULE 4: EYE LANDMARK DETECTION

Given the full 468/478-point face mesh from Module 3, this module
extracts just the 6 landmarks per eye needed for Eye Aspect Ratio
(EAR) computation, using the canonical index mapping defined in
config.py (EyeConfig).
"""

from typing import List, NamedTuple, Sequence, Tuple

from config import EYE_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)

Point2D = Tuple[int, int]


class EyeLandmarks(NamedTuple):
    """The 6 EAR-relevant points for a single eye, in canonical order:
    [outer_corner, upper_outer, upper_inner, inner_corner, lower_inner, lower_outer]
    """

    outer_corner: Point2D
    upper_outer: Point2D
    upper_inner: Point2D
    inner_corner: Point2D
    lower_inner: Point2D
    lower_outer: Point2D

    def as_list(self) -> List[Point2D]:
        return [
            self.outer_corner, self.upper_outer, self.upper_inner,
            self.inner_corner, self.lower_inner, self.lower_outer,
        ]


class EyeLandmarkExtractor:
    """Extracts left/right eye landmark subsets from full mesh output."""

    def __init__(
        self,
        left_eye_indices: Sequence[int] = EYE_CONFIG.LEFT_EYE_IDX,
        right_eye_indices: Sequence[int] = EYE_CONFIG.RIGHT_EYE_IDX,
    ) -> None:
        self.left_eye_indices = list(left_eye_indices)
        self.right_eye_indices = list(right_eye_indices)
        logger.debug(
            "EyeLandmarkExtractor initialized (left_idx=%s, right_idx=%s)",
            self.left_eye_indices, self.right_eye_indices,
        )

    def get_left_eye(self, mesh_points: Sequence[Point2D]) -> EyeLandmarks:
        """Extract the 6 left-eye points from the full mesh point list."""
        return self._extract(mesh_points, self.left_eye_indices)

    def get_right_eye(self, mesh_points: Sequence[Point2D]) -> EyeLandmarks:
        """Extract the 6 right-eye points from the full mesh point list."""
        return self._extract(mesh_points, self.right_eye_indices)

    @staticmethod
    def _extract(mesh_points: Sequence[Point2D], indices: Sequence[int]) -> EyeLandmarks:
        try:
            points = [mesh_points[i] for i in indices]
        except IndexError as error:
            logger.error(
                "Mesh point list too short for expected eye indices %s: %s",
                indices, error,
            )
            raise
        return EyeLandmarks(*points)
