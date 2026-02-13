# Sign Language Recognition - Quick Reference

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run setup script
./setup_sign_language.sh

# 3. Start server
python manage.py runserver

# 4. Open browser and join a meeting
# 5. Click the 🤲 button to enable sign language detection
```

---

## 📦 Files Created/Modified

### New Files
```
video_app/ml_service/__init__.py
video_app/ml_service/config.py
video_app/ml_service/sign_language_detector.py
setup_sign_language.sh
SIGN_LANGUAGE_INTEGRATION_GUIDE.md
SIGN_LANGUAGE_QUICK_REFERENCE.md (this file)
```

### Modified Files
```
video_app/consumers.py        → Added SignLanguageConsumer
static/js/main.js            → Added SignLanguageManager class
templates/video_app/meeting_room.html → Added toggle button & overlay
requirements.txt             → Added ML dependencies
```

---

## 🎯 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Browser   │────▶│  WebSocket   │────▶│  ML Service │
│             │     │  Consumer    │     │  Detector   │
│ (main.js)   │◀────│ (consumers.py)│◀────│ (.py file)  │
└─────────────┘     └──────────────┘     └─────────────┘
      │                     │                     │
   Captures             Broadcasts           Processes
   Video                Predictions          with TFLite
   @ 5 FPS              to all users         + MediaPipe
```

---

## 🔧 Configuration

Edit `video_app/ml_service/config.py`:

```python
# Model paths (auto-detected)
MODEL_PATH = BASE_DIR / "sign-language-recognition/weights/model.tflite"
TRAIN_CSV_PATH = BASE_DIR / "sign-language-recognition/asl-signs/train.csv"

# Performance tuning
SEQUENCE_LENGTH = 30          # Frames needed for prediction
CONFIDENCE_THRESHOLD = 0.7    # Min confidence to show (0.0-1.0)
FRAME_SKIP = 2               # Process every N frames

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
```

---

## 🎮 User Interface

### Meeting Controls
```
┌────────────────────────────────────────────────────┐
│  🎤      📹      🤲        📞                       │
│ Audio  Video   Signs    Leave                      │
└────────────────────────────────────────────────────┘
```

### Sign Language Overlay (when active)
```
┌──────────────────────────────┐
│ 🤲 Sign Language Detected    │
│ ─────────────────────────    │
│                               │
│ user@example.com              │
│ Hello                         │
│ 87% confident                 │
│                               │
│ another@user.com              │
│ Thank You                     │
│ 92% confident                 │
└──────────────────────────────┘
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install tensorflow mediapipe opencv-python` |
| Model not found | Check paths in `config.py` |
| No predictions | Lower `CONFIDENCE_THRESHOLD` in config |
| High CPU usage | Increase frame interval in `main.js` (line 438) |
| WebSocket fails | Verify `routing.py` includes sign language route |

---

## 📊 Performance Tips

**Reduce Latency:**
- Decrease `SEQUENCE_LENGTH` (e.g., 20 frames)
- Increase frame rate in `main.js` (e.g., 150ms)

**Reduce CPU Usage:**
- Increase frame interval (e.g., 400ms)
- Increase `FRAME_SKIP` (e.g., 3)

**Improve Accuracy:**
- Increase `SEQUENCE_LENGTH` (e.g., 40 frames)
- Lower `CONFIDENCE_THRESHOLD` (e.g., 0.6)

---

## 🧪 Testing Commands

```bash
# Test model loading
python -c "from video_app.ml_service.sign_language_detector import SignLanguageDetector; print('✅ OK')"

# Test imports
python -c "import tensorflow, mediapipe, cv2, numpy, pandas; print('✅ All imports OK')"

# Check WebSocket route
python manage.py show_urls | grep sign-language

# Run development server
python manage.py runserver

# Check logs for errors
python manage.py runserver 2>&1 | grep -i error
```

---

## 🔍 Debug Mode

### Enable Debug Logging
In `video_app/ml_service/config.py`:
```python
ENABLE_DEBUG_LOGGING = True
```

### Check Browser Console
Press F12 → Console tab, look for:
- "Sign language WebSocket connected" ✅
- "Sign language detection started" ✅
- Frame capture errors ❌

### Check Django Logs
Look for:
- "Sign language detector initialized for user..." ✅
- "Failed to load model" ❌
- WebSocket connection errors ❌

---

## 📝 Code Snippets

### Manually trigger detection
```javascript
// In browser console
window.app.signLanguage.start()
```

### Reset detector
```javascript
// In browser console
window.app.signLanguage.stop()
window.app.signLanguage.start()
```

### Check active connections
```python
# In Django shell
from channels.layers import get_channel_layer
channel_layer = get_channel_layer()
```

---

## 🔗 Architecture Overview

```
video_app/
├── ml_service/
│   ├── __init__.py
│   ├── config.py                    # Settings & paths
│   └── sign_language_detector.py    # Core ML logic
│       ├── SignLanguageDetector     # Per-user detector
│       └── SignLanguageDetectorPool # Multi-user manager
│
├── consumers.py
│   └── SignLanguageConsumer         # WebSocket handler
│       ├── connect()                # Initialize detector
│       ├── receive()                # Process frames
│       ├── disconnect()             # Cleanup
│       └── sign_prediction()        # Broadcast results
│
└── routing.py
    └── ws/sign-language/<room_id>/  # WebSocket route
```

```
static/js/main.js
└── SignLanguageManager              # Frontend manager
    ├── initialize()                 # Connect WebSocket
    ├── start()                      # Begin capturing
    ├── stop()                       # Stop capturing
    ├── captureAndSendFrame()        # Encode & send
    ├── handleMessage()              # Receive predictions
    └── updateUI()                   # Display results
```

---

## ⚡ API Reference

### WebSocket Messages

**Client → Server:**
```json
{
  "type": "video_frame",
  "frame": "data:image/jpeg;base64,..."
}
```

**Server → Client:**
```json
{
  "type": "sign_prediction",
  "user_id": "123",
  "username": "user@example.com",
  "sign": "Hello",
  "confidence": 0.87
}
```

---

## 📞 Need Help?

1. Read full guide: `SIGN_LANGUAGE_INTEGRATION_GUIDE.md`
2. Check Django logs: Terminal output when running server
3. Check browser console: F12 → Console
4. Test model standalone: Run `experiments/model_inference/local_inference_test1_chat.py`

---

## ✅ Verification Checklist

- [ ] Setup script runs without errors
- [ ] Server starts successfully
- [ ] Meeting page loads
- [ ] Hands button visible in controls
- [ ] Button turns green when clicked
- [ ] Overlay appears when active
- [ ] Console shows "WebSocket connected"
- [ ] Making signs shows predictions
- [ ] Multiple users work simultaneously
- [ ] Existing features still work

---

**Ready to go! 🎉**

For detailed documentation, see `SIGN_LANGUAGE_INTEGRATION_GUIDE.md`
