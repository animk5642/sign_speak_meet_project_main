# Word3 Gesture Recognition - American Sign Language Detector

A hand gesture recognition system that detects American Sign Language (ASL) letters using MediaPipe and a TensorFlow Lite model with advanced swipe gesture detection.

## 📁 Folder Structure

```
word3_gesture_recognition/
├── word3.py                              # Main application
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── model/
│   └── keypoint_classifier/
│       ├── keypoint_classifier.py        # Classifier module
│       ├── keypoint_classifier.tflite    # Pre-trained TFLite model
│       └── keypoint_classifier_label.csv # ASL letter labels (A-Z)
└── utils/
    └── cvfpscalc.py                     # FPS calculator utility
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python word3.py
```

## ⌨️ Controls

| Key | Action |
|-----|--------|
| **A-Z** | Log keypoints for a specific letter |
| **N** | Normal mode (gesture detection) |
| **K** | Capture landmarks from dataset |
| **D** | Dataset processing mode |
| **B** | Delete last letter (backspace) |
| **C** | Clear current word and sentence |
| **Left Hand Swipe** | Delete last letter (backspace) |
| **No hand for 2s** | Add space between words |
| **ESC** | Quit application |

## 🎯 Features

### ✅ Real-time Hand Detection
- Dual-hand support (left & right)
- MediaPipe hand landmark detection
- Bounding rectangle visualization

### ✅ Letter Recognition
- Right hand: ASL letter classification
- Supports A-Z letters
- Hold gesture for 1 second to register
- Real-time feedback with progress bar

### ✅ Backspace Swipe Gesture
- Left hand: Swipe LEFT to delete (backspace)
- Visual feedback with orange-red trail
- Configurable swipe parameters:
  - Minimum distance: 85px
  - Max vertical movement: 50px
  - Minimum velocity: 15px/frame
  - Consistency ratio: 72%
  - Time window: 0.08-1.20 seconds

### ✅ Word/Sentence Building
- Automatic word break after 2 seconds without hands
- Displays current word and sentence
- Real-time text output

## 📊 Configuration

Edit swipe detection parameters in `word3.py`:

```python
SWIPE_HISTORY_FRAMES    = 14      # Frames to track
SWIPE_MIN_DISTANCE_PX   = 85      # Minimum swipe distance
SWIPE_MAX_VERTICAL_PX   = 50      # Max vertical deviation
SWIPE_MIN_VELOCITY_PX   = 15      # Min velocity per frame
SWIPE_CONSISTENCY_RATIO = 0.72    # % frames moving left
SWIPE_MIN_TIME_SEC      = 0.08    # Min swipe duration
SWIPE_MAX_TIME_SEC      = 1.20    # Max swipe duration
SWIPE_MIN_R2            = 0.80    # Min path linearity
SWIPE_COOLDOWN_SEC      = 0.95    # Cooldown between swipes
```

## 🎬 Camera Parameters

Edit in `word3.py`:

```python
--device              0         # Camera device (0=default webcam)
--width               960       # Frame width
--height              540       # Frame height
--use_static_image_mode False   # Static vs dynamic mode
--min_detection_confidence 0.7  # Hand detection confidence
--min_tracking_confidence  0.5  # Hand tracking confidence
```

## 📋 Example Usage

```bash
# Run with default settings
python word3.py

# Run with custom camera and resolution
python word3.py --device 0 --width 1280 --height 720

# Run in static image mode
python word3.py --use_static_image_mode
```

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8.1.78 | Image processing & visualization |
| mediapipe | 0.10.14 | Hand detection & landmarks |
| numpy | 1.26.4 | Numerical computations |
| tensorflow-cpu | 2.15.1 | TFLite interpreter |
| Pillow | 12.1.1 | Image utilities |
| matplotlib | 3.10.8 | Optional visualization |

## 📝 Supported Gestures

- **A-Z Letters**: ASL hand alphabet
- **Left Swipe**: Backspace/Delete
- **Hand Pause**: Word space

## 🖼️ Output Display

The application shows:
- Real-time hand landmarks
- Bounding rectangles
- Current letter being held
- Letter hold progress bar
- Detected word and sentence
- FPS counter
- Swipe status indicator

## ✨ Notes

- Ensure adequate lighting for best hand detection
- Keep hands within camera frame
- The model is optimized for ASL alphabet (A-Z)
- Swipe detection works on physical LEFT hand only
- Letter recognition uses physical RIGHT hand

## 🐛 Troubleshooting

1. **No hand detected**: Improve lighting, ensure hands in frame
2. **Poor accuracy**: Adjust `min_detection_confidence` parameter
3. **Slow performance**: Reduce frame resolution
4. **Swipe not registering**: Adjust swipe parameters in code

## 📄 License

See parent repository for license information.

## 👤 Author

Original implementation for American Sign Language Detection project
