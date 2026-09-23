# AI-Based Driver Cognitive State Monitoring System

A modular, software-only (webcam-based) computer vision system for
monitoring driver fatigue and cognitive state 

**Build status: ALL 13 MODULES COMPLETE.**

## Project Structure

```
DriverMonitoringSystem/
├── main.py                 # Entry point (dashboard by default, --debug for OpenCV-only view)
├── config.py                # ALL tunable parameters, one dataclass per module
├── requirements.txt
├── README.md
├── modules/
│   ├── camera.py             # Module 1  - CameraStream
│   ├── face_detector.py      # Module 2  - FaceDetector
│   ├── face_mesh.py          # Module 3  - FaceMeshDetector
│   ├── eye_detector.py       # Module 4  - EyeLandmarkExtractor
│   ├── ear_calculator.py     # Module 5  - EARCalculator
│   ├── blink_detector.py     # Module 6  - BlinkDetector
│   ├── mouth_detector.py     # Module 7  - MouthLandmarkExtractor
│   ├── mar_calculator.py     # Module 8  - MARCalculator
│   ├── yawn_detector.py      # Module 9  - YawnDetector
│   ├── head_pose.py          # Module 10 - HeadPoseEstimator
│   ├── fatigue_engine.py     # Module 11 - FatigueEngine
│   └── alarm.py               # Module 12 - AlarmSystem
├── dashboard/
│   ├── dashboard.py           # Module 13 - PyQt6 main window (wires everything together)
│   └── widgets.py             # Reusable StatusCard / FatigueGauge / RiskBadge widgets
├── utils/
│   ├── drawing.py             # OpenCV overlay helpers (boxes, contours, status text)
│   ├── math_utils.py          # Distance, EMA, Euler-angle decomposition helpers
│   └── constants.py           # Fixed colors / labels / metadata
├── logger/
│   └── logger.py              # Centralized logging utility
├── assets/
│   └── alarm.wav               # Synthesized two-tone alarm sound (no external audio file needed)
└── models/                     # Reserved for any future trained model files
```

## Setup

1. Create and activate a virtual environment (recommended):

   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   > **Note:** `mediapipe` is pinned to `0.10.14` in requirements.txt.
   > Newer mediapipe releases (0.10.15+) removed the legacy
   > `mp.solutions.face_detection` / `mp.solutions.face_mesh` API this
   > project uses, in favor of a different Tasks API. Installing a
   > newer version will cause `AttributeError: module 'mediapipe' has
   > no attribute 'solutions'`.

3. Run the full dashboard:

   ```
   python main.py
   ```

   Or run the lightweight OpenCV-only debug view (no PyQt6 GUI, useful
   for quick pipeline sanity checks):

   ```
   python main.py --debug
   ```

## What You Should See

- **Dashboard mode**: a window with your live camera feed on the left
  and a status panel on the right showing Face Status, Eye Status,
  Blink Count, EAR, MAR, Yawning, Head Pose, FPS, Current Time, and
  Alarm Status, plus a circular **Fatigue Score** gauge (0-100) and a
  color-coded **Risk Level** badge (green/amber/red).
- **Debug mode**: a single OpenCV window with the same information
  overlaid as text on the video feed directly.
- When the Fatigue Score crosses the configured threshold (default
  70%), you should hear the alarm sound and see "Alarm Status: ACTIVE".

## Configuration

Every threshold, index, weight, and color lives in `config.py`,
organized into one dataclass per module (e.g. `EARConfig`,
`FatigueConfig`, `AlarmConfig`). Tune behavior there — never hardcode
values inside the `modules/` files.

Key thresholds you may want to calibrate for your own webcam/lighting:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `EAR_THRESHOLD` | `EARConfig` | 0.21 | Below this, eye is "closed" |
| `MAR_THRESHOLD` | `MARConfig` | 0.55 | Above this, mouth is "open" (yawn candidate) |
| `YAW_THRESHOLD_DEG` | `HeadPoseConfig` | 15.0 | Head-turn angle to trigger LEFT/RIGHT |
| `FATIGUE_ALARM_THRESHOLD` | `AlarmConfig` | 70.0 | Fatigue score that triggers the alarm |

## Module Roadmap

| # | Module | Status |
|---|--------|--------|
| 1 | Camera | ✅ Complete |
| 2 | Face Detection | ✅ Complete |
| 3 | Face Mesh | ✅ Complete |
| 4 | Eye Landmark Detection | ✅ Complete |
| 5 | Eye Aspect Ratio | ✅ Complete |
| 6 | Blink Detection | ✅ Complete |
| 7 | Mouth Landmark Detection | ✅ Complete |
| 8 | Mouth Aspect Ratio | ✅ Complete |
| 9 | Yawn Detection | ✅ Complete |
| 10 | Head Pose Estimation | ✅ Complete |
| 11 | Fatigue Score Engine | ✅ Complete |
| 12 | Alarm System | ✅ Complete |
| 13 | Professional Dashboard | ✅ Complete |

## Testing Notes

Every module was individually import-tested and exercised with
synthetic frames/landmark data during development (see chat history
for full test transcripts: blank-frame face-detection graceful
failure, EAR/MAR math on synthetic points, alarm cooldown logic, and
a full offscreen PyQt6 dashboard render). **You should still run
`python main.py` on your own machine with a real webcam** to verify
end-to-end accuracy and calibrate thresholds for your lighting/camera.

## Notes

This is a software-only, laptop/webcam-based system. No hardware,
microcontrollers, IoT, or embedded components are used at any stage.
