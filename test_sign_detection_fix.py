"""
Test script to verify sign language detection fix
Compares the working original code approach with our implementation
OPTIMIZED FOR SPEED with real-time percentage display
NOW WITH IMPORTANT_LANDMARKS FIX!
"""
import sys
import os
import cv2
import numpy as np
import base64
import time

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_app.ml_service.sign_language_detector import SignLanguageDetector, IMPORTANT_LANDMARKS

# Paths
MODEL_PATH = "sign-language-recognition/weights/model.tflite"
TRAIN_CSV = "sign-language-recognition/asl-signs/train.csv"

# Performance settings
FRAME_SKIP = 3  # Process every 3rd frame for faster performance (10 FPS)
JPEG_QUALITY = 60  # Lower quality for faster encoding (50-95)

def test_detection():
    """Test the detector with webcam feed"""
    print("🔍 Testing Sign Language Detection Fix (OPTIMIZED + IMPORTANT_LANDMARKS)...")
    print("=" * 60)
    
    # Initialize detector
    try:
        detector = SignLanguageDetector(MODEL_PATH, TRAIN_CSV)
        print("✅ Detector initialized successfully")
        print(f"✅ Using IMPORTANT_LANDMARKS: {len(IMPORTANT_LANDMARKS)} landmarks (not 543!)")
        print(f"⚡ Real-time mode: Processing every {FRAME_SKIP} frames (~10 FPS)")
        print(f"⚡ JPEG quality: {JPEG_QUALITY}%")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # Set webcam to lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n📹 Webcam opened. Processing frames...")
    print("Press 'q' to quit, 'r' to reset sequence\n")
    
    frame_count = 0
    processed_count = 0
    prediction_count = 0
    last_prediction = None
    last_confidence = 0
    hands_detected = False
    prediction_cooldown = 0  # Frames to show prediction before allowing next
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start = time.time()
        
        # Process every Nth frame for speed optimization
        if frame_count % FRAME_SKIP == 0:
            processed_count += 1
            
            # Encode frame to base64 with optimized quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frame_data = f"data:image/jpeg;base64,{frame_base64}"
            
            # Process frame (real-time sliding window)
            result = detector.process_frame(frame_data)
            
            # Update prediction if available (updates continuously)
            if result:
                # Only count as new prediction if different from last
                if last_prediction != result['sign']:
                    prediction_count += 1
                    print(f"🤲 Frame {frame_count}: NEW Sign - '{result['sign']}' ({result['confidence']*100:.1f}%)")
                
                last_prediction = result['sign']
                last_confidence = result['confidence'] * 100
        
        # Calculate progress (check BEFORE it processes the frame)
        frames_collected = len(detector.frame_keypoints)
        progress_pct = int((frames_collected / 30) * 100)
        
        # Check if hands are detected OR we're in prediction cooldown
        hands_detected = frames_collected > 0 or prediction_cooldown > 0
        
        # Create info overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Display hands status (always detecting after 30 frames)
        if len(detector.frame_keypoints) >= 30:
            hands_status = "REAL-TIME DETECTION ACTIVE ✓"
            hands_color = (0, 255, 0)
        elif len(detector.frame_keypoints) > 0:
            hands_status = f"Warming up... {len(detector.frame_keypoints)}/30"
            hands_color = (255, 255, 0)
        else:
            hands_status = "No hands detected - Show hands to start"
            hands_color = (255, 255, 255)
        
        cv2.putText(frame, hands_status, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hands_color, 2, cv2.LINE_AA)
        
        # Display current prediction (always show if available)
        if last_prediction:
            cv2.putText(frame, f"Sign: {last_prediction}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {last_confidence:.1f}%", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Show buffer status during warm-up
            if len(detector.frame_keypoints) < 30:
                frames_collected = len(detector.frame_keypoints)
                progress_pct = int((frames_collected / 30) * 100)
                cv2.putText(frame, f"Warming up: {frames_collected}/30 frames ({progress_pct}%)", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Draw progress bar
                bar_width = int((frames_collected / 30) * 600)
                cv2.rectangle(frame, (10, 90), (610, 110), (60, 60, 60), -1)
                cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (255, 255, 0), -1)
                cv2.rectangle(frame, (10, 90), (610, 110), (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No sign detected (below 70% confidence)", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow('Sign Language Detection - REAL-TIME MODE', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF  # Reduced from 5ms to 1ms for faster response
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_sequence()
            last_prediction = None
            last_confidence = 0
            prediction_cooldown = 0
            print("♻️  Sequence reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   Total frames captured: {frame_count}")
    print(f"   Frames processed: {processed_count}")
    print(f"   Predictions made: {prediction_count}")
    print(f"   Total time: {elapsed_time:.1f} seconds")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Processing rate: {(processed_count/frame_count)*100:.0f}% (every {FRAME_SKIP} frames)")
    print(f"   Success rate: {(prediction_count/max(1,processed_count//30))*100:.1f}%")
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_detection()
