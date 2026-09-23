"""
modules/ear_calculator.py
---------------------------
MODULE 5: EYE ASPECT RATIO (EAR)

Implements the Eye Aspect Ratio metric from:
    Soukupova, T. & Cech, J. (2016). "Real-Time Eye Blink Detection
    using Facial Landmarks."

Mathematical definition (6 points p1..p6 per eye, in order
[outer corner, upper-outer, upper-inner, inner corner, lower-inner,
lower-outer]):

    EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 * ||p1 - p4|| )

Intuition: the numerator sums two roughly-vertical eyelid distances,
the denominator is the (roughly constant) horizontal eye width. When
the eye is open, EAR sits around 0.25-0.35. When the eye closes, the
vertical distances shrink toward zero while the horizontal width
stays roughly constant, so EAR drops sharply toward 0.
"""

from modules.eye_detector import EyeLandmarks
from logger.logger import get_logger
from utils.math_utils import euclidean_distance

logger = get_logger(__name__)


class EARCalculator:
    """Stateless calculator for the Eye Aspect Ratio of a single eye."""

    @staticmethod
    def compute(eye: EyeLandmarks) -> float:
        """Compute the EAR for one eye given its 6 landmark points.

        Parameters
        ----------
        eye : EyeLandmarks
            The 6-point landmark set for one eye (see modules.eye_detector).

        Returns
        -------
        float
            The Eye Aspect Ratio. Typically in [0.0, ~0.4].
        """
        vertical_1 = euclidean_distance(eye.upper_outer, eye.lower_outer)
        vertical_2 = euclidean_distance(eye.upper_inner, eye.lower_inner)
        horizontal = euclidean_distance(eye.outer_corner, eye.inner_corner)

        if horizontal == 0:
            logger.warning("Degenerate eye width (0 px) encountered; returning EAR=0.0")
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)

    @classmethod
    def compute_average(cls, left_eye: EyeLandmarks, right_eye: EyeLandmarks) -> float:
        """Compute the average EAR across both eyes.

        Averaging both eyes is standard practice: it smooths out
        landmark noise and remains valid even if the driver's head is
        slightly turned (one eye is often clearer than the other).
        """
        left_ear = cls.compute(left_eye)
        right_ear = cls.compute(right_eye)
        return (left_ear + right_ear) / 2.0
