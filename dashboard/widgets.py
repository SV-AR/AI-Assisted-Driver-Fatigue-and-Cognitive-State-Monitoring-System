"""
dashboard/widgets.py
----------------------
Reusable, custom-drawn PyQt6 widgets used by the main dashboard
(Module 13). Kept separate from dashboard.py so individual widgets
can be unit-tested / reused (e.g. in a future analytics screen)
without importing the whole application window.
"""

from typing import Optional

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from config import DASHBOARD_CONFIG


class StatusCard(QFrame):
    """A small labeled metric card, e.g. "EAR" -> "0.24".

    Used for every simple text metric on the dashboard (EAR, MAR,
    blink count, head pose, FPS, etc.) so they all share one
    consistent visual style.
    """

    def __init__(self, title: str, initial_value: str = "--", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusCard")
        self.setStyleSheet(
            f"""
            #StatusCard {{
                background-color: {DASHBOARD_CONFIG.PANEL_COLOR};
                border-radius: 10px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self._title_label = QLabel(title.upper())
        self._title_label.setStyleSheet(
            f"color: #9a9ab0; font-size: 11px; font-weight: 600; letter-spacing: 1px;"
        )

        self._value_label = QLabel(initial_value)
        self._value_label.setStyleSheet(
            f"color: {DASHBOARD_CONFIG.TEXT_COLOR}; font-size: 20px; font-weight: 700;"
        )

        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)

    def set_value(self, value: str, color: Optional[str] = None) -> None:
        """Update the displayed value, optionally overriding its text color."""
        self._value_label.setText(value)
        if color:
            self._value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 700;")
        else:
            self._value_label.setStyleSheet(
                f"color: {DASHBOARD_CONFIG.TEXT_COLOR}; font-size: 20px; font-weight: 700;"
            )


class RiskBadge(QLabel):
    """A pill-shaped, color-coded label showing the current risk level."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("LOW RISK", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(36)
        self.set_risk_level("LOW RISK")

    def set_risk_level(self, risk_level: str) -> None:
        """Update the badge text and background color for the given risk level."""
        color_map = {
            "LOW RISK": DASHBOARD_CONFIG.RISK_COLOR_LOW,
            "MEDIUM RISK": DASHBOARD_CONFIG.RISK_COLOR_MEDIUM,
            "HIGH RISK": DASHBOARD_CONFIG.RISK_COLOR_HIGH,
        }
        color = color_map.get(risk_level.upper(), DASHBOARD_CONFIG.RISK_COLOR_LOW)
        self.setText(risk_level.upper())
        self.setStyleSheet(
            f"""
            background-color: {color};
            color: #101018;
            font-weight: 800;
            font-size: 14px;
            border-radius: 18px;
            padding: 4px 12px;
            """
        )


class FatigueGauge(QWidget):
    """A custom-painted circular gauge showing the 0-100 Fatigue Score.

    Implemented with QPainter rather than a 3rd-party gauge widget so
    the project keeps a minimal, standard dependency footprint
    (PyQt6 only), which matters for reproducibility in an IEEE
    publication / artifact evaluation context.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self._score: float = 0.0
        self._risk_level: str = "LOW RISK"

    def set_score(self, score: float, risk_level: str) -> None:
        self._score = max(0.0, min(100.0, score))
        self._risk_level = risk_level
        self.update()  # trigger a repaint

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height())
        rect = QRectF(
            (self.width() - side) / 2 + 10,
            (self.height() - side) / 2 + 10,
            side - 20,
            side - 20,
        )

        color_map = {
            "LOW RISK": DASHBOARD_CONFIG.RISK_COLOR_LOW,
            "MEDIUM RISK": DASHBOARD_CONFIG.RISK_COLOR_MEDIUM,
            "HIGH RISK": DASHBOARD_CONFIG.RISK_COLOR_HIGH,
        }
        arc_color = QColor(color_map.get(self._risk_level.upper(), DASHBOARD_CONFIG.RISK_COLOR_LOW))

        # Background track (full circle, dim)
        track_pen = QPen(QColor("#3a3a4d"), 14, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0, 360 * 16)

        # Foreground arc proportional to fatigue score (0-100 -> 0-360 degrees)
        progress_pen = QPen(arc_color, 14, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(progress_pen)
        span_angle = int((self._score / 100.0) * 360 * 16)
        start_angle = 90 * 16  # start from the top, clockwise
        painter.drawArc(rect, start_angle, -span_angle)

        # Center score text
        painter.setPen(QColor(DASHBOARD_CONFIG.TEXT_COLOR))
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self._score:.0f}")

        painter.end()
