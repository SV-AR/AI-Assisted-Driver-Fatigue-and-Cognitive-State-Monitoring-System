"""
utils/drawing.py
-----------------
Reusable OpenCV drawing helpers for visualizing detection results:
face bounding boxes, landmark points, eye/mouth contours, and a
heads-up status panel. Both `main.py` (raw OpenCV debug window) and
`dashboard/dashboard.py` (PyQt6 GUI) call into these functions so the
visual style stays consistent everywhere.
"""

from typing import Sequence, Tuple

import cv2
import numpy as np

from utils.constants import COLOR_BLACK, COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_YELLOW

Point2D = Tuple[int, int]


def draw_face_box(frame: np.ndarray, box: Tuple[int, int, int, int], confidence: float) -> None:
    """Draw a bounding box and confidence label around a detected face.

    Parameters
    ----------
    frame : np.ndarray
        The BGR frame to draw on (modified in place).
    box : Tuple[int, int, int, int]
        (x, y, width, height) of the face bounding box in pixels.
    confidence : float
        Detection confidence score in [0, 1].
    """
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)
    label = f"Face {confidence * 100:.0f}%"
    cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)


def draw_landmark_points(frame: np.ndarray, points: Sequence[Point2D],
                          color: Tuple[int, int, int] = COLOR_YELLOW, radius: int = 1) -> None:
    """Draw a small filled circle at each landmark point."""
    for (x, y) in points:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def draw_eye_contour(frame: np.ndarray, eye_points: Sequence[Point2D],
                      color: Tuple[int, int, int] = COLOR_GREEN) -> None:
    """Draw the closed polygon contour of an eye given its 6 EAR points."""
    pts = np.array(eye_points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def draw_mouth_contour(frame: np.ndarray, mouth_points: Sequence[Point2D],
                        color: Tuple[int, int, int] = COLOR_YELLOW) -> None:
    """Draw the contour polygon of the mouth given its outline points."""
    pts = np.array(mouth_points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def draw_text_with_background(frame: np.ndarray, text: str, origin: Point2D,
                               font_scale: float = 0.6,
                               text_color: Tuple[int, int, int] = COLOR_WHITE,
                               bg_color: Tuple[int, int, int] = COLOR_BLACK,
                               thickness: int = 1, padding: int = 4) -> None:
    """Draw text on a solid background rectangle for readability over video."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_status_panel(frame: np.ndarray, status_lines: Sequence[str],
                       origin: Point2D = (20, 30), line_height: int = 26) -> None:
    """Draw a stacked list of status strings in the top-left corner,
    each on its own readable text-with-background line.

    Parameters
    ----------
    frame : np.ndarray
        Frame to draw on (modified in place).
    status_lines : Sequence[str]
        Ordered list of strings, e.g. ["EAR: 0.24", "Blinks: 5", ...].
    origin : Tuple[int, int]
        (x, y) pixel location of the first line.
    line_height : int
        Vertical spacing between lines, in pixels.
    """
    x, y = origin
    for i, line in enumerate(status_lines):
        draw_text_with_background(frame, line, (x, y + i * line_height))


def risk_level_color(risk_level: str) -> Tuple[int, int, int]:
    """Map a risk-level string to a BGR color for on-screen display."""
    mapping = {
        "LOW RISK": COLOR_GREEN,
        "MEDIUM RISK": COLOR_YELLOW,
        "HIGH RISK": COLOR_RED,
    }
    return mapping.get(risk_level.upper(), COLOR_WHITE)
