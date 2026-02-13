"""
Visual System Architecture Diagram
Sign Language Recognition Integration
"""

SYSTEM_ARCHITECTURE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SIGN LANGUAGE RECOGNITION SYSTEM                      ║
║                         Real-Time Video Conferencing                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Browser)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Meeting Room UI (meeting_room.html)                                 │    │
│  │                                                                       │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │    │
│  │  │  You       │  │ Participant│  │ Participant│  │ Participant│   │    │
│  │  │  [VIDEO]   │  │  [VIDEO]   │  │  [VIDEO]   │  │  [VIDEO]   │   │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │    │
│  │                                                                       │    │
│  │  Controls: [🎤 Mute] [📹 Video] [🤲 Signs] [📞 Leave]               │    │
│  │                                                                       │    │
│  │  ┌──────────────────────────────────────┐                           │    │
│  │  │  Sign Language Overlay (Top-Right)   │                           │    │
│  │  │  ┌────────────────────────────────┐  │                           │    │
│  │  │  │ user@example.com               │  │                           │    │
│  │  │  │ Hello                          │  │                           │    │
│  │  │  │ 87% confident                  │  │                           │    │
│  │  │  ├────────────────────────────────┤  │                           │    │
│  │  │  │ another@user.com               │  │                           │    │
│  │  │  │ Thank You                      │  │                           │    │
│  │  │  │ 92% confident                  │  │                           │    │
│  │  │  └────────────────────────────────┘  │                           │    │
│  │  └──────────────────────────────────────┘                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↕                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SignLanguageManager (main.js)                                       │    │
│  │                                                                       │    │
│  │  • Captures video frames from localVideo element                     │    │
│  │  • Rate: 5 FPS (every 200ms)                                        │    │
│  │  • Encodes: Canvas → JPEG → Base64                                  │    │
│  │  • Sends via WebSocket                                              │    │
│  │  • Receives predictions from all users                              │    │
│  │  • Updates overlay UI                                               │    │
│  │  • Auto-clears predictions after 3 seconds                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↕ WebSocket
                    wss://domain/ws/sign-language/ROOM_ID/
                                    ↕
┌──────────────────────────────────────────────────────────────────────────────┐
│                         BACKEND (Django Server)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SignLanguageConsumer (consumers.py)                                 │    │
│  │                                                                       │    │
│  │  • Handles WebSocket connections                                     │    │
│  │  • Receives: Base64 encoded frames                                   │    │
│  │  • Routes to detector pool                                           │    │
│  │  • Broadcasts predictions to room group                              │    │
│  │  • Manages lifecycle (connect/disconnect)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↕                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SignLanguageDetectorPool (sign_language_detector.py)                │    │
│  │                                                                       │    │
│  │  • Manages per-user detector instances                               │    │
│  │  • Thread-safe operations                                            │    │
│  │  • get_detector(user_id) → Returns detector                          │    │
│  │  • remove_detector(user_id) → Cleanup                                │    │
│  │                                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │    │
│  │  │  Detector    │  │  Detector    │  │  Detector    │             │    │
│  │  │  User 1      │  │  User 2      │  │  User 3      │  ...        │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↕                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SignLanguageDetector (Per User)                                     │    │
│  │                                                                       │    │
│  │  1. Frame Input (Base64) → Decode → NumPy Array                     │    │
│  │     ↓                                                                │    │
│  │  2. MediaPipe Holistic Processing                                    │    │
│  │     ├─ Face Landmarks: 468 points × 3 coords = 1404 values         │    │
│  │     ├─ Left Hand: 21 points × 3 coords = 63 values                 │    │
│  │     ├─ Pose: 33 points × 3 coords = 99 values                      │    │
│  │     └─ Right Hand: 21 points × 3 coords = 63 values                │    │
│  │     ↓                                                                │    │
│  │  3. Extract Keypoints → Shape: (543, 3)                             │    │
│  │     ↓                                                                │    │
│  │  4. Sliding Window Buffer (30 frames)                               │    │
│  │     [frame-29, frame-28, ..., frame-1, frame-0]                    │    │
│  │     Shape: (30, 543, 3)                                             │    │
│  │     ↓                                                                │    │
│  │  5. TensorFlow Lite Inference                                       │    │
│  │     Input: (30, 543, 3) float32                                     │    │
│  │     Model: model.tflite                                             │    │
│  │     Output: Probability distribution over N sign classes            │    │
│  │     ↓                                                                │    │
│  │  6. Get Best Prediction                                             │    │
│  │     class_id = argmax(probabilities)                                │    │
│  │     confidence = probabilities[class_id]                            │    │
│  │     ↓                                                                │    │
│  │  7. Threshold Check (confidence > 0.7)                              │    │
│  │     ✓ Pass → Map class_id to sign label from train.csv            │    │
│  │     ✗ Fail → Return None                                           │    │
│  │     ↓                                                                │    │
│  │  8. Return Prediction                                               │    │
│  │     { sign: "Hello", confidence: 0.87, class_id: 42 }              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Configuration (config.py)                                           │    │
│  │                                                                       │    │
│  │  MODEL_PATH = "sign-language-recognition/weights/model.tflite"      │    │
│  │  TRAIN_CSV_PATH = "sign-language-recognition/asl-signs/train.csv"   │    │
│  │  SEQUENCE_LENGTH = 30 frames                                        │    │
│  │  CONFIDENCE_THRESHOLD = 0.7                                         │    │
│  │  MIN_DETECTION_CONFIDENCE = 0.5                                     │    │
│  │  MIN_TRACKING_CONFIDENCE = 0.5                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                            DATA FLOW DIAGRAM                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

User A (Browser)                  Server                    User B (Browser)
─────────────────                ────────                   ─────────────────

1. Enable Sign Detection
   [Click 🤲 Button]
        │
        ├──────────> WebSocket Connect
        │            ├─> Create Detector for User A
        │            └─> Join Room Group
        │
2. Capture Frame @ 5 FPS
   Canvas.toDataURL()
        │
        ├──────────> { type: "video_frame",
        │              frame: "data:image/jpeg;base64..." }
        │
        │            3. Process Frame
        │            ├─> Decode Base64
        │            ├─> MediaPipe Extract Keypoints
        │            ├─> Add to 30-frame buffer
        │            ├─> TFLite Inference
        │            └─> Get Prediction: "Hello" (87%)
        │
        │            4. Broadcast to Room
        │            ├─────────────────────────────────> Receive
        │            │                                   { sign: "Hello",
        │            │                                     user: "User A",
        │            │                                     confidence: 0.87 }
        │            │                                          │
5. Receive Own Prediction    │                                          │
   Display in Overlay   <────┘                                          │
                                                             6. Display User A's
                                                                Sign in Overlay


                        [User B Also Signing]
                                                            
                                                                   Capture Frame
                                                                        │
                        { type: "video_frame", ... } <──────────────────┤
                                      │
                        Process → "Thank You" (92%)
                                      │
7. Receive User B's                   │
   Prediction           <─────────────┼─> Broadcast to Room
   Display in Overlay                 │
                                      └─────────────────────────────> Display
                                                                      in Overlay

╔══════════════════════════════════════════════════════════════════════════════╗
║                         PERFORMANCE CHARACTERISTICS                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Frame Capture:     5 FPS (200ms interval)
Detection Latency: 6 seconds (30 frames @ 5 FPS)
CPU Usage:         20-30% per active detector
Memory:            ~500MB per detector instance
Network:           ~10 KB/s per user
Model Size:        ~50MB (loaded once, shared)
Concurrent Users:  Supports 10+ simultaneous detectors

╔══════════════════════════════════════════════════════════════════════════════╗
║                           TECHNOLOGY STACK                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Frontend:
  • HTML5 Canvas API        → Frame capture
  • WebSocket API          → Real-time communication
  • JavaScript ES6+        → SignLanguageManager
  • Bootstrap 5            → UI styling
  • Font Awesome          → Icons

Backend:
  • Django 4.2.7          → Web framework
  • Channels 4.0.0        → WebSocket support
  • Daphne 4.0.0          → ASGI server
  • Python 3.x            → Runtime

Machine Learning:
  • TensorFlow 2.13.0     → ML framework
  • TensorFlow Lite       → Optimized inference
  • MediaPipe 0.10.9      → Landmark detection
  • OpenCV 4.8.1.78       → Image processing
  • NumPy 1.24.3          → Numerical operations
  • Pandas 2.0.3          → Label management

Video Conferencing:
  • Agora RTC SDK         → Video/audio streaming
  • WebRTC                → Peer connections

╔══════════════════════════════════════════════════════════════════════════════╗
║                            FILE STRUCTURE                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

videoconf/
├── video_app/
│   ├── ml_service/                    ← 🆕 NEW MODULE
│   │   ├── __init__.py
│   │   ├── config.py                  ← Configuration
│   │   └── sign_language_detector.py  ← Core ML logic
│   ├── consumers.py                   ← ✏️ SignLanguageConsumer added
│   ├── routing.py                     ← WebSocket routes
│   └── ...
├── sign-language-recognition/
│   ├── weights/
│   │   └── model.tflite              ← TFLite model
│   └── asl-signs/
│       └── train.csv                 ← Sign labels
├── static/js/
│   └── main.js                       ← ✏️ SignLanguageManager added
├── templates/video_app/
│   └── meeting_room.html             ← ✏️ UI components added
├── requirements.txt                  ← ✏️ ML dependencies added
├── setup_sign_language.sh            ← 🆕 Setup script
├── SIGN_LANGUAGE_INTEGRATION_GUIDE.md    ← 🆕 Full guide
├── SIGN_LANGUAGE_QUICK_REFERENCE.md      ← 🆕 Quick ref
└── IMPLEMENTATION_SUMMARY.md             ← 🆕 Summary

"""

if __name__ == "__main__":
    print(SYSTEM_ARCHITECTURE)
