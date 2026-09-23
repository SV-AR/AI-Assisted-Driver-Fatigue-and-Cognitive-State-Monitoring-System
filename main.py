"""
main.py
-------
Entry point for the Driver Cognitive State Monitoring System.

STATUS: All 13 modules complete. Launches the full PyQt6 dashboard
(Module 13), which internally wires together the camera, face
detection, face mesh, EAR/MAR, blink/yawn detection, head pose
estimation, fatigue engine, and alarm system (Modules 1-12).

For a lightweight, GUI-free debug view (raw OpenCV window with
overlays but no PyQt6 dependency), run:

    python main.py --debug

This is useful for quickly sanity-checking the CV pipeline on a
machine where PyQt6 isn't installed yet, or over SSH/X11 forwarding.
"""

import sys

import cv2

from config import DISPLAY_CONFIG
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
from utils.constants import COLOR_GREEN
from utils.drawing import draw_eye_contour, draw_face_box, draw_mouth_contour, draw_status_panel

logger = get_logger(__name__)


def run_dashboard_mode() -> None:
    """Launch the full PyQt6 dashboard (Module 13)."""
    from dashboard.dashboard import run_dashboard
    run_dashboard()


def run_debug_mode() -> None:
    """Run the full CV pipeline in a plain OpenCV window (no PyQt6 needed)."""
    logger.info("Starting Driver Monitoring System in DEBUG (OpenCV-only) mode.")

    camera = CameraStream()
    face_detector = FaceDetector()
    face_mesh = FaceMeshDetector()
    eye_extractor = EyeLandmarkExtractor()
    mouth_extractor = MouthLandmarkExtractor()
    blink_detector = BlinkDetector()
    yawn_detector = YawnDetector()
    head_pose_estimator = HeadPoseEstimator()
    fatigue_engine = FatigueEngine()
    alarm_system = AlarmSystem()

    try:
        camera.start()
    except CameraError as error:
        logger.critical("Fatal camera error: %s", error)
        sys.exit(1)

    try:
        while True:
            try:
                success, frame = camera.read()
            except CameraError as error:
                logger.critical("Camera disconnected during runtime: %s", error)
                break

            if not success:
                continue

            faces = face_detector.detect(frame)
            status_lines = [f"FPS: {camera.fps}"]

            if faces:
                best = faces[0]
                draw_face_box(frame, best.as_tuple(), best.confidence)
                mesh_points = face_mesh.process(frame)

                if mesh_points is not None:
                    left_eye = eye_extractor.get_left_eye(mesh_points)
                    right_eye = eye_extractor.get_right_eye(mesh_points)
                    ear = EARCalculator.compute_average(left_eye, right_eye)
                    draw_eye_contour(frame, left_eye.as_list())
                    draw_eye_contour(frame, right_eye.as_list())

                    mouth = mouth_extractor.get_mouth(mesh_points)
                    mar = MARCalculator.compute(mouth)
                    draw_mouth_contour(frame, mouth.contour)

                    blink_state = blink_detector.update(ear)
                    yawn_state = yawn_detector.update(mar)
                    pose = head_pose_estimator.estimate(mesh_points, frame.shape[:2])
                    direction = pose.direction if pose else "N/A"

                    fatigue = fatigue_engine.update(
                        eye_status=blink_state.eye_status,
                        blink_rate_per_min=blink_state.blink_rate_per_min,
                        is_yawning=yawn_state.is_yawning,
                        yawn_rate_per_min=yawn_state.yawn_rate_per_min,
                        head_direction=direction,
                        is_prolonged_closure=blink_state.is_prolonged_closure,
                    )
                    alarm_system.check_and_trigger(fatigue.fatigue_score)

                    status_lines += [
                        f"EAR: {ear:.2f}  MAR: {mar:.2f}",
                        f"Blinks: {blink_state.blink_count}  Rate: {blink_state.blink_rate_per_min}/min",
                        f"Yawning: {'YES' if yawn_state.is_yawning else 'NO'}",
                        f"Head Pose: {direction}",
                        f"Fatigue: {fatigue.fatigue_score:.0f}%  {fatigue.risk_level}",
                    ]
            else:
                status_lines.append("Face: NOT FOUND")

            draw_status_panel(frame, status_lines)
            cv2.imshow(DISPLAY_CONFIG.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(DISPLAY_CONFIG.QUIT_KEY):
                logger.info("Quit key pressed. Shutting down.")
                break
    finally:
        camera.stop()
        face_detector.close()
        face_mesh.close()
        alarm_system.shutdown()
        cv2.destroyAllWindows()
        logger.info("Debug mode finished cleanly.")


if __name__ == "__main__":
    if "--debug" in sys.argv:
        run_debug_mode()
    else:
        try:
            run_dashboard_mode()
        except ImportError as error:
            logger.error("PyQt6 not available (%s). Falling back to --debug mode.", error)
            run_debug_mode()
