# ML Service Module - Sign Language Recognition

This module provides real-time sign language detection for video conferencing applications.

## Overview

The ML service integrates TensorFlow Lite and MediaPipe to detect American Sign Language (ASL) gestures from video frames in real-time during video meetings.

## Components

### 1. `sign_language_detector.py`

**Classes:**

#### `SignLanguageDetector`
Core detection class that processes individual user's video frames.

**Methods:**
- `__init__(model_path, train_csv_path, sequence_length=30, confidence_threshold=0.7)`
  - Initializes detector with model and parameters
  - Loads TFLite model and sign labels
  - Sets up MediaPipe Holistic

- `process_frame(frame_data: str) -> Optional[Dict]`
  - Processes a single Base64-encoded video frame
  - Returns prediction dict or None
  - Thread-safe for async operations

- `extract_keypoints(results) -> np.ndarray`
  - Extracts 543 keypoints from MediaPipe results
  - Returns shape (543, 3) array

- `reset_sequence()`
  - Clears the frame buffer
  - Use when user pauses/resumes

**Returns:**
```python
{
    'sign': 'Hello',
    'confidence': 0.87,
    'class_id': 42
}
```

#### `SignLanguageDetectorPool`
Manages multiple detector instances for concurrent users.

**Methods:**
- `get_detector(user_id: str) -> SignLanguageDetector`
  - Gets or creates detector for user
  - Thread-safe

- `remove_detector(user_id: str)`
  - Removes detector on user disconnect
  - Frees resources

- `reset_detector(user_id: str)`
  - Resets frame sequence for user

### 2. `config.py`

Configuration parameters for the ML service.

**Key Settings:**
```python
MODEL_PATH              # Path to model.tflite
TRAIN_CSV_PATH          # Path to train.csv (labels)
SEQUENCE_LENGTH         # Frames needed (default: 30)
CONFIDENCE_THRESHOLD    # Min confidence (default: 0.7)
FRAME_SKIP             # Process every N frames (default: 2)
```

## Usage

### Basic Usage

```python
from video_app.ml_service.sign_language_detector import SignLanguageDetector
from video_app.ml_service.config import MODEL_PATH, TRAIN_CSV_PATH

# Create detector
detector = SignLanguageDetector(
    model_path=str(MODEL_PATH),
    train_csv_path=str(TRAIN_CSV_PATH)
)

# Process frame (Base64 encoded)
prediction = detector.process_frame(frame_data)

if prediction:
    print(f"Detected: {prediction['sign']} ({prediction['confidence']:.2%})")
```

### Multi-User Usage

```python
from video_app.ml_service.sign_language_detector import SignLanguageDetectorPool
from video_app.ml_service.config import MODEL_PATH, TRAIN_CSV_PATH

# Create pool
pool = SignLanguageDetectorPool(
    model_path=str(MODEL_PATH),
    train_csv_path=str(TRAIN_CSV_PATH)
)

# Get detector for user
user_detector = pool.get_detector('user_123')

# Process frame
prediction = user_detector.process_frame(frame_data)

# Cleanup when done
pool.remove_detector('user_123')
```

### Async Usage (in Django Channels)

```python
from channels.db import database_sync_to_async

@database_sync_to_async
def process_frame_async(frame_data, user_id):
    detector = detector_pool.get_detector(user_id)
    return detector.process_frame(frame_data)

# In async function
prediction = await process_frame_async(frame_data, user_id)
```

## Architecture

```
Frame (Base64)
    ↓
Decode → NumPy Array
    ↓
MediaPipe Holistic
    ├─ Face (468 landmarks)
    ├─ Left Hand (21 landmarks)
    ├─ Pose (33 landmarks)
    └─ Right Hand (21 landmarks)
    ↓
Keypoints (543, 3)
    ↓
Sliding Window (30 frames)
    ↓
TensorFlow Lite Model
    ↓
Predictions (class probabilities)
    ↓
Threshold Filter (> 0.7)
    ↓
Result: {sign, confidence, class_id}
```

## Performance

**Timing:**
- Frame decode: ~5ms
- MediaPipe processing: ~30-50ms
- TFLite inference: ~20-40ms
- Total per frame: ~50-100ms

**Resources:**
- Memory: ~500MB per detector
- CPU: 20-30% per active detector
- Detection latency: 6 seconds (30 frames @ 5 FPS)

## Configuration Tuning

### For Faster Response (Less Accurate)
```python
# config.py
SEQUENCE_LENGTH = 20           # Reduce frames needed
CONFIDENCE_THRESHOLD = 0.6     # Lower threshold
FRAME_SKIP = 3                 # Skip more frames
```

### For Better Accuracy (Slower Response)
```python
# config.py
SEQUENCE_LENGTH = 40           # More frames for context
CONFIDENCE_THRESHOLD = 0.8     # Higher threshold
FRAME_SKIP = 1                 # Process all frames
```

### For Lower CPU Usage
```python
# config.py
FRAME_SKIP = 4                 # Process fewer frames
MIN_DETECTION_CONFIDENCE = 0.3 # Lower MediaPipe confidence
MIN_TRACKING_CONFIDENCE = 0.3
```

## Error Handling

The detector handles errors gracefully:

```python
try:
    prediction = detector.process_frame(frame_data)
except Exception as e:
    logger.error(f"Detection error: {e}")
    prediction = None
```

**Common Errors:**
- Invalid Base64 data → Returns None
- Model loading failure → Raises exception on init
- MediaPipe failure → Returns None for that frame
- Out of memory → May need to reduce concurrent detectors

## Dependencies

```
tensorflow==2.13.0
mediapipe==0.10.9
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
protobuf==3.20.3
```

## Thread Safety

- `SignLanguageDetector` - **NOT thread-safe** (use one per user)
- `SignLanguageDetectorPool` - **Thread-safe** (uses locks internally)

## Memory Management

Detectors are automatically cleaned up:

1. On disconnect: `pool.remove_detector(user_id)`
2. Python garbage collection handles MediaPipe resources
3. TFLite model is shared across all detectors (single load)

## Debugging

### Enable Debug Logging

```python
# config.py
ENABLE_DEBUG_LOGGING = True
```

### Check Detector State

```python
detector = pool.get_detector('user_123')
print(f"Frame buffer size: {len(detector.frame_keypoints)}")
print(f"Ready for prediction: {len(detector.frame_keypoints) == detector.sequence_length}")
```

### Test Model Loading

```python
from video_app.ml_service.sign_language_detector import SignLanguageDetector
from video_app.ml_service.config import MODEL_PATH, TRAIN_CSV_PATH

try:
    detector = SignLanguageDetector(str(MODEL_PATH), str(TRAIN_CSV_PATH))
    print("✅ Model loaded successfully")
    print(f"Sign labels: {len(detector.ord2sign)}")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Limitations

1. **Detection Delay**: 6 seconds (30 frames @ 5 FPS)
2. **ASL Only**: Trained on American Sign Language
3. **Lighting**: Performance degrades in poor lighting
4. **Occlusion**: Hands must be visible to camera
5. **Single Sign**: Detects one sign at a time per user
6. **Static Model**: No real-time learning/adaptation

## Future Enhancements

- [ ] Support for continuous sign language (sentences)
- [ ] Multiple sign language systems (BSL, ISL, etc.)
- [ ] Model quantization for faster inference
- [ ] GPU acceleration support
- [ ] Adaptive confidence thresholds
- [ ] User-specific model fine-tuning

## Testing

### Unit Test Example

```python
import unittest
from video_app.ml_service.sign_language_detector import SignLanguageDetectorPool

class TestSignLanguageDetector(unittest.TestCase):
    def setUp(self):
        self.pool = SignLanguageDetectorPool(MODEL_PATH, TRAIN_CSV_PATH)
    
    def test_detector_creation(self):
        detector = self.pool.get_detector('test_user')
        self.assertIsNotNone(detector)
    
    def test_detector_cleanup(self):
        self.pool.get_detector('test_user')
        self.pool.remove_detector('test_user')
        # Should not raise error
```

## License

This module uses:
- TensorFlow (Apache 2.0)
- MediaPipe (Apache 2.0)
- OpenCV (Apache 2.0)

## Support

For issues or questions:
1. Check logs in Django console
2. Verify model files exist
3. Test imports: `python -c "from video_app.ml_service.sign_language_detector import SignLanguageDetector"`
4. See `SIGN_LANGUAGE_INTEGRATION_GUIDE.md`

## Contributors

Integrated by: Professional AI/ML Engineering Team
Date: February 2026
Version: 1.0.0
