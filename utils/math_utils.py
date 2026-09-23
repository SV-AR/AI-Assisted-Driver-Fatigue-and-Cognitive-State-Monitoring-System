"""
utils/math_utils.py
--------------------
Small, pure mathematical helper functions shared across the computer
vision modules (EAR, MAR, head pose, fatigue engine).

Keeping these in one place avoids duplicating distance/geometry code
in every module and makes unit testing straightforward.
"""

from typing import Sequence, Tuple

import numpy as np

Point2D = Tuple[float, float]


def euclidean_distance(point_a: Point2D, point_b: Point2D) -> float:
    """Compute the Euclidean distance between two 2D points.

    Parameters
    ----------
    point_a, point_b : Tuple[float, float]
        (x, y) pixel or normalized coordinates.

    Returns
    -------
    float
        The straight-line distance between the two points.
    """
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


def midpoint(point_a: Point2D, point_b: Point2D) -> Point2D:
    """Return the midpoint between two 2D points."""
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp `value` into the inclusive range [min_value, max_value]."""
    return max(min_value, min(value, max_value))


def exponential_moving_average(previous: float, new_value: float, alpha: float) -> float:
    """Compute a single-step exponential moving average.

    result = alpha * previous + (1 - alpha) * new_value

    A higher `alpha` favors the historical value (smoother, slower to
    react); a lower `alpha` favors the new reading (more responsive,
    noisier).
    """
    return alpha * previous + (1 - alpha) * new_value


def rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix into (pitch, yaw, roll) in degrees.

    Uses the standard extrinsic X-Y-Z Euler decomposition. This is a
    common approach for consuming the rotation matrix returned by
    `cv2.Rodrigues` after `cv2.solvePnP` in head-pose pipelines.

    Parameters
    ----------
    rotation_matrix : np.ndarray
        A 3x3 rotation matrix.

    Returns
    -------
    (pitch, yaw, roll) : Tuple[float, float, float]
        Rotation angles in degrees.
    """
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0.0

    pitch = np.degrees(x)
    yaw = np.degrees(y)
    roll = np.degrees(z)
    return float(pitch), float(yaw), float(roll)


def average_points(points: Sequence[Point2D]) -> Point2D:
    """Return the centroid (average x, average y) of a list of points."""
    arr = np.array(points, dtype=np.float64)
    centroid = arr.mean(axis=0)
    return float(centroid[0]), float(centroid[1])
