"""
Sign Language Detection Service
Adapted for real-time video conferencing integration
"""
import os
import base64
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import pandas as pd
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# IMPORTANT: Model was trained on these specific landmarks only!
# 13 face/pose landmarks + 75 hand landmarks (indices 468-543)
IMPORTANT_LANDMARKS = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348] + list(range(468, 543))


class SignLanguageDetector:
    """
    Real-time sign language detection using TensorFlow Lite and MediaPipe
    """
    
    def __init__(self, model_path: str, train_csv_path: str, 
                 sequence_length: int = 30, confidence_threshold: float = 0.5):
        """
        Initialize the sign language detector
        
        Args:
            model_path: Path to the TFLite model file
            train_csv_path: Path to training CSV with sign labels
            sequence_length: Number of frames needed for prediction
            confidence_threshold: Minimum confidence for predictions (0.5 = 50%)
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.frame_keypoints = []
        
        # Load TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load sign labels
        try:
            train = pd.read_csv(train_csv_path)
            train["sign_ord"] = train["sign"].astype("category").cat.codes
            self.ord2sign = train[["sign_ord", "sign"]].set_index("sign_ord").squeeze().to_dict()
            logger.info(f"Loaded {len(self.ord2sign)} sign labels")
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            raise
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("Sign language detector initialized successfully")
    
    def extract_keypoints(self, results) -> tuple:
        """
        Extract keypoints from MediaPipe results - MATCHING ORIGINAL WORKING CODE
        
        Args:
            results: MediaPipe holistic results
            
        Returns:
            Tuple of (keypoints array of shape (543, 3), has_hands boolean)
        """
        # Check if we have hand landmarks (required for sign language)
        has_hands = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
        
        # Extract pose landmarks (33 points * 3 coords)
        pose = np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.full(33 * 3, np.nan)
        
        # Extract face landmarks (468 points * 3 coords)
        face = np.array([[l.x, l.y, l.z] for l in results.face_landmarks.landmark]).flatten() \
            if results.face_landmarks else np.full(468 * 3, np.nan)
        
        # Extract left hand landmarks (21 points * 3 coords)
        lh = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.full(21 * 3, np.nan)
        
        # Extract right hand landmarks (21 points * 3 coords)
        rh = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.full(21 * 3, np.nan)
        
        # IMPORTANT: Exact order from original working code - face, lh, pose, rh
        keypoints = np.concatenate([face, lh, pose, rh])
        
        # Return as (543, 3) shape - model will do IMPORTANT_LANDMARKS selection internally!
        return np.reshape(keypoints, (543, 3)), has_hands
    
    def process_frame(self, frame_data: str) -> Optional[Dict]:
        """
        Process a single frame and return prediction if available
        
        Args:
            frame_data: Base64 encoded frame data
            
        Returns:
            Dictionary with prediction and confidence, or None
        """
        try:
            # Decode base64 frame
            frame_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning("Failed to decode frame")
                return None
            
            # Convert to RGB for MediaPipe (matching working code pattern)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False  # Optimize performance
            
            # Process with MediaPipe
            results = self.holistic.process(frame_rgb)
            
            frame_rgb.flags.writeable = True  # Restore writeable flag
            
            # Extract keypoints - matching original working code
            keypoints, has_hands = self.extract_keypoints(results)
            
            # Add to sequence (original code doesn't require hands - model handles NaN)
            self.frame_keypoints.append(keypoints)
            self.frame_keypoints = self.frame_keypoints[-self.sequence_length:]
            
            # Make prediction if we have enough frames
            if len(self.frame_keypoints) >= self.sequence_length:
                prediction = self._make_prediction()
                # Keep sliding window (original code doesn't reset buffer)
                return prediction
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def _make_prediction(self) -> Optional[Dict]:
        """
        Make prediction from accumulated frames - MATCHING ORIGINAL WORKING CODE
        
        Returns:
            Dictionary with sign, confidence, and class_id
        """
        try:
            # Prepare input - EXACT format from original working code
            # Shape: (1, 30, 543, 3) -> but we remove batch dimension with [0]
            # So: (30, 543, 3) - 30 frames, 543 landmarks, 3 coords each
            input_data = np.expand_dims(self.frame_keypoints, axis=0).astype(np.float32)[0]
            
            # Run inference using the exact format from the working code
            prediction = self.prediction_fn(inputs=input_data)
            probs = prediction["outputs"][0]
            
            # Get best prediction
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            
            # Only return if confidence is high enough
            if confidence > self.confidence_threshold:
                return {
                    'sign': self.ord2sign[class_id],
                    'confidence': confidence,
                    'class_id': class_id
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def reset_sequence(self):
        """Reset the frame sequence"""
        self.frame_keypoints = []
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


class SignLanguageDetectorPool:
    """
    Manages multiple sign language detector instances for concurrent users
    """
    
    def __init__(self, model_path: str, train_csv_path: str):
        self.model_path = model_path
        self.train_csv_path = train_csv_path
        self.detectors: Dict[str, SignLanguageDetector] = {}
    
    def get_detector(self, user_id: str) -> SignLanguageDetector:
        """
        Get or create a detector for a specific user
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            SignLanguageDetector instance
        """
        if user_id not in self.detectors:
            self.detectors[user_id] = SignLanguageDetector(
                self.model_path, 
                self.train_csv_path
            )
            logger.info(f"Created detector for user {user_id}")
        
        return self.detectors[user_id]
    
    def remove_detector(self, user_id: str):
        """
        Remove detector for a user (cleanup on disconnect)
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id in self.detectors:
            del self.detectors[user_id]
            logger.info(f"Removed detector for user {user_id}")
    
    def reset_detector(self, user_id: str):
        """
        Reset detector sequence for a user
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id in self.detectors:
            self.detectors[user_id].reset_sequence()
