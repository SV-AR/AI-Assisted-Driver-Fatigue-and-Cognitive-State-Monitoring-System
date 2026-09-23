"""
dashboard/dashboard.py
------------------------
MODULE 13: PROFESSIONAL DASHBOARD

The main application window. Wires together every previous module
into a single real-time pipeline driven by a QTimer:

    Camera -> FaceDetector -> FaceMesh -> Eye/Mouth extraction
    -> EAR/MAR -> Blink/Yawn detection -> Head Pose -> FatigueEngine
    -> AlarmSystem -> on-screen video + live status panel

Layout
------
Left  : live annotated camera feed (face box, mesh, eye/mouth contours)
Right : status cards (Face, Eye, Blink Count, EAR, MAR, Yawning,
        Head Pose, FPS, Time, Alarm Status) + a circular Fatigue
        Score gauge + a color-coded Risk Level badge.
"""

import sys
import time
from datetime import datetime

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QVBoxLayout, QWidget,
)

from config import DASHBOARD_CONFIG
from dashboard.widgets import FatigueGauge, RiskBadge, StatusCard
from logger.logger import get_logger
from modules.alarm import AlarmSystem
from modules.blink_detector import BlinkDetector
from modules.camera import CameraError, CameraStream
from modules.ear_calculator import EARCalculator
from modules.eye_detector import EyeLandmarkExtractor
from modules.face_detector import FaceDetector
from modules.face_mesh import FaceMeshDetector
from modules.fatigue_engine import FatigueEngine
from modules.head_pose import HeadPoseEstimator
from modules.mar_calculator import MARCalculator
from modules.mouth_detector import MouthLandmarkExtractor
from modules.yawn_detector import YawnDetector
from utils.drawing import draw_eye_contour, draw_face_box, draw_mouth_contour

logger = get_logger(__name__)


class DriverMonitoringDashboard(QMainWindow):
    """Main application window for the Driver Cognitive State Monitoring System."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(DASHBOARD_CONFIG.WINDOW_TITLE)
        self.setMinimumSize(DASHBOARD_CONFIG.WINDOW_MIN_WIDTH, DASHBOARD_CONFIG.WINDOW_MIN_HEIGHT)
        self.setStyleSheet(f"background-color: {DASHBOARD_CONFIG.BACKGROUND_COLOR};")

        # ------------------------------------------------------------
        # Initialize the full CV pipeline (Modules 1-12)
        # ------------------------------------------------------------
        self.camera = CameraStream()
        self.face_detector = FaceDetector()
        self.face_mesh = FaceMeshDetector()
        self.eye_extractor = EyeLandmarkExtractor()
        self.mouth_extractor = MouthLandmarkExtractor()
        self.blink_detector = BlinkDetector()
        self.yawn_detector = YawnDetector()
        self.head_pose_estimator = HeadPoseEstimator()
        self.fatigue_engine = FatigueEngine()
        self.alarm_system = AlarmSystem()

        self._camera_started = False
        self._prev_frame_time = time.time()
        self._fps = 0.0

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)

    # ----------------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # --- Left: video panel ---
        self.video_label = QLabel("Starting camera...")
        self.video_label.setFixedWidth(DASHBOARD_CONFIG.VIDEO_PANEL_WIDTH)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color: black; color: {DASHBOARD_CONFIG.TEXT_COLOR}; "
            f"border-radius: 12px; font-size: 16px;"
        )
        root_layout.addWidget(self.video_label)

        # --- Right: status panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(14)

        title = QLabel(DASHBOARD_CONFIG.WINDOW_TITLE)
        title.setWordWrap(True)
        title.setStyleSheet(
            f"color: {DASHBOARD_CONFIG.ACCENT_COLOR}; font-size: 16px; font-weight: 800;"
        )
        right_panel.addWidget(title)

        # Fatigue gauge label
        gauge_label = QLabel("FATIGUE SCORE")
        gauge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gauge_label.setStyleSheet("color: #9a9ab0; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
        right_panel.addWidget(gauge_label)

        # Fatigue gauge (centered)
        gauge_row = QHBoxLayout()
        gauge_row.addStretch(1)
        self.fatigue_gauge = FatigueGauge()
        gauge_row.addWidget(self.fatigue_gauge)
        gauge_row.addStretch(1)
        right_panel.addLayout(gauge_row)

        self.risk_badge = RiskBadge()
        right_panel.addWidget(self.risk_badge)

        # Status cards grid
        grid = QGridLayout()
        grid.setSpacing(10)

        self.card_face = StatusCard("Face Status")
        self.card_eye = StatusCard("Eye Status")
        self.card_blinks = StatusCard("Blink Count")
        self.card_ear = StatusCard("EAR")
        self.card_mar = StatusCard("MAR")
        self.card_yawn = StatusCard("Yawning")
        self.card_pose = StatusCard("Head Pose")
        self.card_fps = StatusCard("FPS")
        self.card_time = StatusCard("Current Time")
        self.card_alarm = StatusCard("Alarm Status")

        cards = [
            self.card_face, self.card_eye, self.card_blinks, self.card_ear,
            self.card_mar, self.card_yawn, self.card_pose, self.card_fps,
            self.card_time, self.card_alarm,
        ]
        for i, card in enumerate(cards):
            grid.addWidget(card, i // 2, i % 2)

        right_panel.addLayout(grid)
        right_panel.addStretch(1)

        right_container = QWidget()
        right_container.setLayout(right_panel)
        right_container.setFixedWidth(380)
        root_layout.addWidget(right_container)

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def start(self) -> None:
        """Start the camera and begin the processing loop."""
        try:
            self.camera.start()
            self._camera_started = True
        except CameraError as error:
            logger.critical("Dashboard failed to start camera: %s", error)
            self.video_label.setText(f"Camera error:\n{error}")
            return

        self.timer.start(DASHBOARD_CONFIG.REFRESH_INTERVAL_MS)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        """Clean shutdown of all pipeline resources when the window closes."""
        logger.info("Dashboard closing. Releasing all resources...")
        self.timer.stop()
        if self._camera_started:
            self.camera.stop()
        self.face_detector.close()
        self.face_mesh.close()
        self.alarm_system.shutdown()
        event.accept()

    # ----------------------------------------------------------------
    # Main per-frame processing pipeline
    # ----------------------------------------------------------------

    def _process_frame(self) -> None:
        try:
            success, frame = self.camera.read()
        except CameraError as error:
            logger.critical("Camera disconnected: %s", error)
            self.timer.stop()
            self.video_label.setText(f"Camera disconnected:\n{error}")
            return

        if not success or frame is None:
            return

        self._update_fps()
        frame_height, frame_width = frame.shape[:2]

        faces = self.face_detector.detect(frame)
        face_found = len(faces) > 0

        ear_value = 0.0
        mar_value = 0.0
        eye_status = "N/A"
        head_direction = "N/A"
        is_yawning = False
        blink_count = 0
        blink_rate = 0.0
        yawn_rate = 0.0
        is_prolonged_closure = False
        fatigue_score = 0.0
        risk_level = "LOW RISK"

        if face_found:
            best_face = faces[0]
            draw_face_box(frame, best_face.as_tuple(), best_face.confidence)

            mesh_points = self.face_mesh.process(frame)

            if mesh_points is not None:
                left_eye = self.eye_extractor.get_left_eye(mesh_points)
                right_eye = self.eye_extractor.get_right_eye(mesh_points)
                ear_value = EARCalculator.compute_average(left_eye, right_eye)
                draw_eye_contour(frame, left_eye.as_list())
                draw_eye_contour(frame, right_eye.as_list())

                mouth = self.mouth_extractor.get_mouth(mesh_points)
                mar_value = MARCalculator.compute(mouth)
                draw_mouth_contour(frame, mouth.contour)

                blink_state = self.blink_detector.update(ear_value)
                eye_status = blink_state.eye_status
                blink_count = blink_state.blink_count
                blink_rate = blink_state.blink_rate_per_min
                is_prolonged_closure = blink_state.is_prolonged_closure

                yawn_state = self.yawn_detector.update(mar_value)
                is_yawning = yawn_state.is_yawning
                yawn_rate = yawn_state.yawn_rate_per_min

                pose_result = self.head_pose_estimator.estimate(mesh_points, (frame_height, frame_width))
                head_direction = pose_result.direction if pose_result else "N/A"

                fatigue_result = self.fatigue_engine.update(
                    eye_status=eye_status,
                    blink_rate_per_min=blink_rate,
                    is_yawning=is_yawning,
                    yawn_rate_per_min=yawn_rate,
                    head_direction=head_direction,
                    is_prolonged_closure=is_prolonged_closure,
                )
                fatigue_score = fatigue_result.fatigue_score
                risk_level = fatigue_result.risk_level

                self.alarm_system.check_and_trigger(fatigue_score)

        self._render_video_frame(frame)
        self._update_status_panel(
            face_found=face_found,
            eye_status=eye_status,
            blink_count=blink_count,
            ear_value=ear_value,
            mar_value=mar_value,
            is_yawning=is_yawning,
            head_direction=head_direction,
            fatigue_score=fatigue_score,
            risk_level=risk_level,
            alarm_active=self.alarm_system.is_playing,
        )

    # ----------------------------------------------------------------
    # Rendering helpers
    # ----------------------------------------------------------------

    def _render_video_frame(self, frame: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaledToWidth(
            DASHBOARD_CONFIG.VIDEO_PANEL_WIDTH, Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _update_status_panel(
        self, face_found: bool, eye_status: str, blink_count: int, ear_value: float,
        mar_value: float, is_yawning: bool, head_direction: str, fatigue_score: float,
        risk_level: str, alarm_active: bool,
    ) -> None:
        self.card_face.set_value("DETECTED" if face_found else "NOT FOUND",
                                  color="#2ecc71" if face_found else "#e74c3c")
        self.card_eye.set_value(eye_status)
        self.card_blinks.set_value(str(blink_count))
        self.card_ear.set_value(f"{ear_value:.2f}")
        self.card_mar.set_value(f"{mar_value:.2f}")
        self.card_yawn.set_value("YES" if is_yawning else "NO",
                                  color="#e74c3c" if is_yawning else None)
        self.card_pose.set_value(head_direction)
        self.card_fps.set_value(f"{self._fps:.1f}")
        self.card_time.set_value(datetime.now().strftime("%H:%M:%S"))
        self.card_alarm.set_value("ACTIVE" if alarm_active else "OFF",
                                   color="#e74c3c" if alarm_active else "#2ecc71")

        self.fatigue_gauge.set_score(fatigue_score, risk_level)
        self.risk_badge.set_risk_level(risk_level)

    def _update_fps(self) -> None:
        now = time.time()
        elapsed = now - self._prev_frame_time
        self._prev_frame_time = now
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self._fps = 0.9 * self._fps + 0.1 * instant_fps


def run_dashboard() -> None:
    """Application entry point: create the QApplication and show the dashboard."""
    app = QApplication(sys.argv)
    window = DriverMonitoringDashboard()
    window.show()
    window.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_dashboard()
