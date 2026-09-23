"""
modules/mouth_detector.py
---------------------------
MODULE 7: MOUTH LANDMARK DETECTION

Extracts the specific mouth landmarks needed for Mouth Aspect Ratio
(MAR) computation, plus a wider contour for on-screen drawing, from
the full face mesh produced by Module 3.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from config import MOUTH_CONFIG
from logger.logger import get_logger

logger = get_logger(__name__)

Point2D = Tuple[int, int]


@dataclass
class MouthLandmarks:
    """The 4 MAR-relevant mouth points plus a drawable contour."""

    top_lip: Point2D
    bottom_lip: Point2D
    left_corner: Point2D
    right_corner: Point2D
    contour: List[Point2D]


class MouthLandmarkExtractor:
    """Extracts mouth landmark subsets from full mesh output."""

    def __init__(
        self,
        top_lip_idx: int = MOUTH_CONFIG.TOP_LIP_IDX,
        bottom_lip_idx: int = MOUTH_CONFIG.BOTTOM_LIP_IDX,
        left_corner_idx: int = MOUTH_CONFIG.LEFT_CORNER_IDX,
        right_corner_idx: int = MOUTH_CONFIG.RIGHT_CORNER_IDX,
        contour_idx: Sequence[int] = MOUTH_CONFIG.CONTOUR_IDX,
    ) -> None:
        self.top_lip_idx = top_lip_idx
        self.bottom_lip_idx = bottom_lip_idx
        self.left_corner_idx = left_corner_idx
        self.right_corner_idx = right_corner_idx
        self.contour_idx = list(contour_idx)
        logger.debug("MouthLandmarkExtractor initialized.")

    def get_mouth(self, mesh_points: Sequence[Point2D]) -> MouthLandmarks:
        """Extract MAR points + drawable contour from the full mesh point list."""
        try:
            top_lip = mesh_points[self.top_lip_idx]
            bottom_lip = mesh_points[self.bottom_lip_idx]
            left_corner = mesh_points[self.left_corner_idx]
            right_corner = mesh_points[self.right_corner_idx]
            contour = [mesh_points[i] for i in self.contour_idx]
        except IndexError as error:
            logger.error("Mesh point list too short for expected mouth indices: %s", error)
            raise

        return MouthLandmarks(
            top_lip=top_lip,
            bottom_lip=bottom_lip,
            left_corner=left_corner,
            right_corner=right_corner,
            contour=contour,
        )
