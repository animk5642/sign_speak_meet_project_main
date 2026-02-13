"""
Configuration for Sign Language Detection Service
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Sign Language Recognition paths
SIGN_LANGUAGE_DIR = BASE_DIR / "sign-language-recognition"
MODEL_PATH = SIGN_LANGUAGE_DIR / "weights" / "model.tflite"
TRAIN_CSV_PATH = SIGN_LANGUAGE_DIR / "asl-signs" / "train.csv"

# Model parameters
SEQUENCE_LENGTH = 30  # Number of frames needed for prediction
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for displaying predictions
FRAME_SKIP = 2  # Process every N frames to reduce load

# MediaPipe parameters
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# WebSocket settings
MAX_FRAME_SIZE = 500000  # 500KB max per frame
FRAME_QUALITY = 70  # JPEG quality for frame transmission

# Logging
ENABLE_DEBUG_LOGGING = True
