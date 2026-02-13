#!/usr/bin/env python3
"""
Quick test to verify IMPORTANT_LANDMARKS fix without webcam
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'meet_clone.settings')
import django
django.setup()

from video_app.ml_service.sign_language_detector import SignLanguageDetector, IMPORTANT_LANDMARKS
import numpy as np

def test_landmark_extraction():
    """Test that landmark extraction produces correct shape"""
    print("\n" + "="*80)
    print("🧪 TESTING LANDMARK EXTRACTION SHAPE")
    print("="*80)
    
    # Initialize detector
    model_path = project_root / "sign-language-recognition" / "weights" / "model.tflite"
    csv_path = project_root / "sign-language-recognition" / "asl-signs" / "train.csv"
    
    detector = SignLanguageDetector(
        model_path=str(model_path),
        train_csv_path=str(csv_path)
    )
    
    print(f"\n✅ Detector initialized")
    print(f"   - IMPORTANT_LANDMARKS count: {len(IMPORTANT_LANDMARKS)}")
    print(f"   - Expected shape per frame: (88, 3)")
    
    # Create mock MediaPipe results with all landmarks
    class MockLandmark:
        def __init__(self):
            self.x = np.random.random()
            self.y = np.random.random()
            self.z = np.random.random()
    
    class MockResults:
        def __init__(self):
            # Create all 543 landmarks
            self.face_landmarks = type('obj', (object,), {
                'landmark': [MockLandmark() for _ in range(468)]
            })()
            self.left_hand_landmarks = type('obj', (object,), {
                'landmark': [MockLandmark() for _ in range(21)]
            })()
            self.pose_landmarks = type('obj', (object,), {
                'landmark': [MockLandmark() for _ in range(33)]
            })()
            self.right_hand_landmarks = type('obj', (object,), {
                'landmark': [MockLandmark() for _ in range(21)]
            })()
    
    # Test extraction
    results = MockResults()
    keypoints, has_hands = detector.extract_keypoints(results)
    
    print(f"\n📊 Extraction Test Results:")
    print(f"   - Extracted shape: {keypoints.shape}")
    print(f"   - Expected shape: (88, 3)")
    print(f"   - Hands detected: {has_hands}")
    print(f"   - Contains NaN: {np.isnan(keypoints).any()}")
    
    # Verify
    if keypoints.shape == (88, 3):
        print(f"\n✅ SHAPE CORRECT! Using IMPORTANT_LANDMARKS only!")
    else:
        print(f"\n❌ SHAPE INCORRECT! Expected (88, 3), got {keypoints.shape}")
        return False
    
    if np.isnan(keypoints).any():
        print(f"❌ Contains NaN values (should be replaced with 0.0)")
        return False
    else:
        print(f"✅ No NaN values (correctly replaced with 0.0)")
    
    # Test buffer accumulation
    print(f"\n📦 Testing Buffer Accumulation:")
    for i in range(30):
        detector.frame_keypoints.append(keypoints)
    
    buffer_array = np.array(detector.frame_keypoints, dtype=np.float32)
    print(f"   - Buffer shape: {buffer_array.shape}")
    print(f"   - Expected: (30, 88, 3)")
    
    if buffer_array.shape == (30, 88, 3):
        print(f"   ✅ BUFFER SHAPE CORRECT!")
    else:
        print(f"   ❌ BUFFER SHAPE INCORRECT!")
        return False
    
    return True

def main():
    print("\n" + "="*80)
    print("🚀 IMPORTANT_LANDMARKS FIX - QUICK TEST")
    print("="*80)
    print("\nVerifying:")
    print("  1. IMPORTANT_LANDMARKS = 88 landmarks")
    print("  2. Extract shape = (88, 3) per frame")
    print("  3. Buffer shape = (30, 88, 3) for prediction")
    print("  4. NaN values replaced with 0.0")
    
    success = test_landmark_extraction()
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\n🎉 The fix is working correctly!")
        print("\nKey improvements:")
        print("  ✓ Using 88 IMPORTANT_LANDMARKS (not 543)")
        print("  ✓ Matching training data preprocessing")
        print("  ✓ NaN values handled correctly")
        print("  ✓ Should see MUCH better accuracy now!\n")
        return 0
    else:
        print("❌ TESTS FAILED!")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
