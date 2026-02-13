"""
EXACT COPY of the original working code - minimal changes
This is the PROVEN WORKING implementation
"""
import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# ================== CONFIG ==================
MODEL_PATH = "sign-language-recognition/weights/model.tflite"
TRAIN_CSV = "sign-language-recognition/asl-signs/train.csv"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.5  # Lower threshold for better detection
FONT = cv2.FONT_HERSHEY_SIMPLEX

os.environ["QT_QPA_PLATFORM"] = "xcb"

# ================== LOAD MODEL ==================
print("Loading model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
prediction_fn = interpreter.get_signature_runner("serving_default")
print("✅ Model loaded")

# ================== LOAD LABELS ==================
print("Loading labels...")
train = pd.read_csv(TRAIN_CSV)
train["sign_ord"] = train["sign"].astype("category").cat.codes
ORD2SIGN = train[["sign_ord", "sign"]].set_index("sign_ord").squeeze().to_dict()
print(f"✅ Loaded {len(ORD2SIGN)} sign labels")

# ================== MEDIAPIPE ==================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ================== HELPERS ==================
def mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              landmark_drawing_spec=None,
                              connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                              connection_drawing_spec=mp_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                              connection_drawing_spec=mp_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    """Extract keypoints EXACTLY as original - 543 landmarks in specific order"""
    pose = np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.full(33 * 3, np.nan)

    face = np.array([[l.x, l.y, l.z] for l in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.full(468 * 3, np.nan)

    lh = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.full(21 * 3, np.nan)

    rh = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.full(21 * 3, np.nan)

    # EXACT order: face, left_hand, pose, right_hand
    keypoints = np.concatenate([face, lh, pose, rh])
    return np.reshape(keypoints, (543, 3))

# ================== MAIN LOOP ==================
print("\n📹 Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_keypoints = []
latest_prediction = ""
frame_count = 0

print("✅ Webcam opened")
print("🎯 Press ESC to quit\n")

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Extract and collect keypoints
        keypoints = extract_keypoints(results)
        frame_keypoints.append(keypoints)
        frame_keypoints = frame_keypoints[-SEQUENCE_LENGTH:]  # Sliding window

        # Make prediction when buffer is full
        if len(frame_keypoints) == SEQUENCE_LENGTH:
            # EXACT format from original code
            input_data = np.expand_dims(frame_keypoints, axis=0).astype(np.float32)[0]
            prediction = prediction_fn(inputs=input_data)
            probs = prediction["outputs"][0]

            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])

            if confidence > CONFIDENCE_THRESHOLD:
                latest_prediction = f"{ORD2SIGN[class_id]} ({int(confidence*100)}%)"
                if frame_count % 10 == 0:  # Print every 10th frame
                    print(f"🤲 Detected: {ORD2SIGN[class_id]} ({confidence*100:.1f}%)")
            else:
                latest_prediction = f"Low confidence ({int(confidence*100)}%)"

        # Display info
        # Background for text
        cv2.rectangle(image, (0, 0), (640, 100), (0, 0, 0), -1)
        
        # Show buffer status
        buffer_status = f"Buffer: {len(frame_keypoints)}/{SEQUENCE_LENGTH}"
        cv2.putText(image, buffer_status, (10, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show prediction
        cv2.putText(image, f"Sign: {latest_prediction}", (10, 70), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Sign Language - ORIGINAL WORKING CODE", image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Done!")
