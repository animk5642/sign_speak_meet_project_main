"""
Simple diagnostic to test if MediaPipe can detect hands
"""
import cv2
import mediapipe as mp
import time

print("="*80)
print("MEDIAPIPE HAND DETECTION DIAGNOSTIC")
print("="*80)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.3,  # Lower threshold for testing
    min_tracking_confidence=0.3,
    model_complexity=0  # Faster model
)

print("\n✅ MediaPipe initialized")
print("   - min_detection_confidence: 0.3")
print("   - min_tracking_confidence: 0.3")
print("   - model_complexity: 0 (fast)")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n📹 Webcam opened (640x480)")
print("\nInstructions:")
print("  - Show your hands to the camera")
print("  - Move them around")
print("  - Try different distances")
print("  - Press 'q' to quit")
print("\n" + "="*80 + "\n")

frame_count = 0
detection_count = 0
fps_start = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Calculate FPS
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_start)
        fps_start = time.time()
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = holistic.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    # Check what was detected
    has_pose = results.pose_landmarks is not None
    has_face = results.face_landmarks is not None
    has_left_hand = results.left_hand_landmarks is not None
    has_right_hand = results.right_hand_landmarks is not None
    has_any_hand = has_left_hand or has_right_hand
    
    if has_any_hand:
        detection_count += 1
        if detection_count % 30 == 1:  # Print every 30 detections
            print(f"✅ Frame {frame_count}: HANDS DETECTED!")
            print(f"   - Left hand: {has_left_hand}")
            print(f"   - Right hand: {has_right_hand}")
            print(f"   - Pose: {has_pose}")
            print(f"   - Face: {has_face}\n")
    
    # Draw detection results
    frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Draw landmarks if detected
    if has_left_hand:
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    if has_right_hand:
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    if has_pose:
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )
    
    # Status overlay
    status_color = (0, 255, 0) if has_any_hand else (0, 0, 255)
    status_text = "HANDS DETECTED ✓" if has_any_hand else "NO HANDS - Show hands!"
    
    cv2.rectangle(frame_bgr, (0, 0), (640, 100), (0, 0, 0), -1)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_bgr, status_text, (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Show detection stats
    cv2.putText(frame_bgr, f"Detections: {detection_count}/{frame_count}", (10, 95),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('MediaPipe Diagnostic', frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
print(f"Total frames: {frame_count}")
print(f"Hands detected: {detection_count}")
print(f"Detection rate: {(detection_count/max(1,frame_count))*100:.1f}%")

if detection_count == 0:
    print("\n❌ NO HANDS DETECTED!")
    print("\nPossible issues:")
    print("  1. Lighting too dark - try better lighting")
    print("  2. Hands too far from camera - move closer")
    print("  3. Camera quality poor - try different camera")
    print("  4. MediaPipe model issue - reinstall mediapipe")
    print("\nTry:")
    print("  pip install --upgrade mediapipe")
else:
    print(f"\n✅ MediaPipe is working! Detected hands in {detection_count} frames")
    print("The issue might be in the integration code, not MediaPipe itself")

print("\n")
