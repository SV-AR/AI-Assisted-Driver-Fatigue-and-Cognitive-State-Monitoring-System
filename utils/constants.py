"""
utils/constants.py
-------------------
Structural/static constants shared across the project.

Difference from config.py:
- config.py holds TUNABLE parameters (thresholds, resolutions,
  device indices) that a user or researcher may want to change.
- constants.py holds FIXED, non-tunable values such as colors used
  for drawing overlays, text labels, and other literals that are
  part of the program's identity rather than its configuration.
"""

# BGR color tuples (OpenCV uses BGR, not RGB)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)

# Text overlay defaults
FONT_FACE = "FONT_HERSHEY_SIMPLEX"   # resolved via getattr(cv2, ...) where used
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Project metadata
PROJECT_NAME = "AI-Based Driver Cognitive State Monitoring System"
PROJECT_VERSION = "0.1.0-module1"
