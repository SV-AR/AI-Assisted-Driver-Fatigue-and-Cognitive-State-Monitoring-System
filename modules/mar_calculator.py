"""
modules/mar_calculator.py
---------------------------
MODULE 8: MOUTH ASPECT RATIO (MAR)

Analogous to EAR, but for the mouth. Given the inner-lip top/bottom
points and the two mouth corners:

    MAR = ||top_lip - bottom_lip|| / ||left_corner - right_corner||

Intuition: during normal speech/expression the mouth opens and
closes quickly and rarely very wide relative to its width. During a
yawn, the vertical opening becomes large relative to the mouth's
horizontal width, causing MAR to spike well above its resting value.
"""

from modules.mouth_detector import MouthLandmarks
from logger.logger import get_logger
from utils.math_utils import euclidean_distance

logger = get_logger(__name__)


class MARCalculator:
    """Stateless calculator for the Mouth Aspect Ratio."""

    @staticmethod
    def compute(mouth: MouthLandmarks) -> float:
        """Compute the MAR given a MouthLandmarks instance.

        Parameters
        ----------
        mouth : MouthLandmarks
            The mouth landmark set (see modules.mouth_detector).

        Returns
        -------
        float
            The Mouth Aspect Ratio. Typically ~0.1-0.3 at rest and
            0.6+ during a yawn (thresholds are tunable in config.py).
        """
        vertical = euclidean_distance(mouth.top_lip, mouth.bottom_lip)
        horizontal = euclidean_distance(mouth.left_corner, mouth.right_corner)

        if horizontal == 0:
            logger.warning("Degenerate mouth width (0 px) encountered; returning MAR=0.0")
            return 0.0

        mar = vertical / horizontal
        return float(mar)
