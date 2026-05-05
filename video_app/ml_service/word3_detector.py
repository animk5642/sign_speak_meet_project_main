"""
Word3 Gesture Recognition Detector
Server-side port of word3.py logic for real-time letter detection (A-Z).
Uses MediaPipe Hands + TFLite keypoint classifier.
"""
import os
import csv
import copy
import time
import base64
import itertools
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# ── Swipe config — dead simple: any small left movement = backspace ───────────
SWIPE_MIN_LEFT_PX  = 6        # Just 6px leftward motion at 320×240 triggers backspace
SWIPE_COOLDOWN_SEC = 0.30     # Cooldown between backspaces to prevent rapid-fire

LEFT_HAND_LABEL  = "Left"
RIGHT_HAND_LABEL = "Right"
TRACK_LANDMARKS  = [0, 5, 9, 13, 17, 8]  # wrist + 4 knuckles + index tip


class BackspaceSwipeDetector:
    """
    Watches ONLY the physical left hand.
    Fires True when ANY small leftward motion is detected — dead simple.
    No state machine, no smoothing, no linearity checks.
    """

    def __init__(self):
        self.prev_x: float = None
        self.last_fire_t: float = 0.0
        self._tracking: bool = False

    @staticmethod
    def tracking_point_px(hand_landmarks, fw, fh):
        xs = [hand_landmarks.landmark[i].x for i in TRACK_LANDMARKS]
        ys = [hand_landmarks.landmark[i].y for i in TRACK_LANDMARKS]
        return (int(np.mean(xs) * fw), int(np.mean(ys) * fh))

    def update(self, point_px: tuple) -> bool:
        """Feed one point from the LEFT hand. Returns True on leftward swipe."""
        now = time.time()
        cur_x = point_px[0]

        # Cooldown — prevent rapid-fire
        if (now - self.last_fire_t) < SWIPE_COOLDOWN_SEC:
            self.prev_x = cur_x
            return False

        # First frame — just store position
        if self.prev_x is None:
            self.prev_x = cur_x
            return False

        dx = cur_x - self.prev_x  # negative = leftward
        self.prev_x = cur_x

        # Any leftward motion beyond threshold = backspace
        if dx <= -SWIPE_MIN_LEFT_PX:
            self._tracking = True
            self.last_fire_t = now
            return True

        self._tracking = False
        return False

    def reset(self):
        self.prev_x = None
        self._tracking = False

    @property
    def tracking(self):
        return self._tracking



class Word3Detector:
    """
    Real-time letter detection using MediaPipe Hands + keypoint classifier.
    Ported from word3.py — same logic, no display code.
    """

    def __init__(self, model_path: str, labels_path: str,
                 letter_hold_sec: float = 0.6, word_break_sec: float = 1.3):
        self.letter_hold_sec = letter_hold_sec
        self.word_break_sec = word_break_sec

        # State
        self.current_word: List[str] = []
        self.sentence: List[str] = []
        self.last_letter = ""
        self.last_letter_time = 0.0
        self.last_hand_time = time.time()

        # Swipe detector for backspace
        self.swiper = BackspaceSwipeDetector()
        self.did_backspace = False  # flag to send to client

        # Load TFLite keypoint classifier
        try:
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path, num_threads=1
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Word3 keypoint classifier loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load word3 model: {e}")
            raise

        # Load labels
        try:
            with open(labels_path, encoding="utf-8-sig") as f:
                self.labels = [row[0] for row in csv.reader(f)]
            logger.info(f"Word3 labels loaded: {self.labels}")
        except Exception as e:
            logger.error(f"Failed to load word3 labels: {e}")
            raise

        # Initialize MediaPipe Hands (optimized for speed)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,   # Lower = faster detection (0.5 is MediaPipe default)
            min_tracking_confidence=0.5,
            model_complexity=0,              # Lighter palm detection model
        )

        logger.info("Word3 detector initialized successfully")

    def classify_keypoints(self, landmark_list: List[float]) -> int:
        """Run keypoint classifier — exact logic from word3.py KeyPointClassifier"""
        input_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_index, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()
        output_index = self.output_details[0]["index"]
        result = self.interpreter.get_tensor(output_index)
        return int(np.argmax(np.squeeze(result)))

    @staticmethod
    def calc_landmark_list(image_w: int, image_h: int, landmarks) -> List[List[int]]:
        """Convert MediaPipe landmarks to pixel coords — from word3.py"""
        return [
            [min(int(lm.x * image_w), image_w - 1),
             min(int(lm.y * image_h), image_h - 1)]
            for lm in landmarks.landmark
        ]

    @staticmethod
    def pre_process_landmark(landmark_list: List[List[int]]) -> List[float]:
        """Normalise landmarks relative to wrist — from word3.py"""
        temp = copy.deepcopy(landmark_list)
        bx, by = temp[0]
        for pt in temp:
            pt[0] -= bx
            pt[1] -= by
        flat = list(itertools.chain.from_iterable(temp))
        max_val = max(map(abs, flat)) or 1
        return [v / max_val for v in flat]

    def process_frame(self, frame_data: str) -> Optional[Dict]:
        """
        Process a single base64-encoded video frame.
        Returns dict with letter, word, sentence, hand landmarks, and backspace status.
        """
        try:
            # Decode base64 frame
            raw = frame_data.split(',')[1] if ',' in frame_data else frame_data
            frame_bytes = base64.b64decode(raw)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            h, w = frame.shape[:2]

            # Resize to 320x240 if client sent a larger frame (safety net)
            if w > 320 or h > 240:
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
                h, w = 240, 320

            # NOTE: cv2.flip removed — client now sends pre-flipped frames

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            detected_letter = ""
            hand_landmarks_data = []  # For keypoint visualization on client
            self.did_backspace = False

            if results.multi_hand_landmarks is not None:
                self.last_hand_time = time.time()

                # Split detected hands into left / right (exact word3.py logic)
                left_lm = None
                right_lm = None
                for lm, hd in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
                    label = hd.classification[0].label
                    # Skip building landmark visualization data — client canvas is hidden
                    if label == LEFT_HAND_LABEL:
                        left_lm = lm
                    else:
                        right_lm = lm

                # ── LEFT HAND — swipe detector only ──
                if left_lm is not None:
                    track_pt = BackspaceSwipeDetector.tracking_point_px(left_lm, w, h)
                    did_swipe = self.swiper.update(track_pt)

                    if did_swipe:
                        self.backspace()
                        self.did_backspace = True
                else:
                    self.swiper.reset()

                # ── RIGHT HAND — letter classification only ──
                if right_lm is not None:
                    landmark_list = self.calc_landmark_list(w, h, right_lm)
                    pre_processed = self.pre_process_landmark(landmark_list)
                    hand_sign_id = self.classify_keypoints(pre_processed)
                    detected_letter = self.labels[hand_sign_id]

                    now = time.time()
                    # Freeze letter hold timer if left hand is mid-swipe
                    if not self.swiper.tracking:
                        if detected_letter == self.last_letter:
                            if (now - self.last_letter_time) >= self.letter_hold_sec:
                                if not self.current_word or self.current_word[-1] != detected_letter:
                                    self.current_word.append(detected_letter)
                                    self.last_letter_time = now
                        else:
                            self.last_letter = detected_letter
                            self.last_letter_time = now
                    else:
                        # Keep updating detected letter but don't advance timer
                        self.last_letter = detected_letter
                        self.last_letter_time = now
                else:
                    self.last_letter = ""
            else:
                # No hands → word break after timeout
                self.swiper.reset()
                if (time.time() - self.last_hand_time) >= self.word_break_sec:
                    if self.current_word:
                        self.sentence.append("".join(self.current_word))
                        self.current_word = []
                self.last_letter = ""

            # Calculate hold progress for the current letter
            hold_progress = 0.0
            if self.last_letter and self.last_letter_time > 0 and not self.swiper.tracking:
                held = min(time.time() - self.last_letter_time, self.letter_hold_sec)
                hold_progress = held / self.letter_hold_sec

            return {
                'letter': detected_letter,
                'hold_progress': hold_progress,
                'current_word': "".join(self.current_word),
                'sentence': " ".join(self.sentence),
                'hand_landmarks': hand_landmarks_data,
                'did_backspace': self.did_backspace,
                'is_swiping': self.swiper.tracking,
            }

        except Exception as e:
            logger.error(f"Word3 process_frame error: {e}")
            return None

    def backspace(self):
        """Remove last letter or last word"""
        if self.current_word:
            self.current_word.pop()
        elif self.sentence:
            self.current_word = list(self.sentence.pop())

    def clear(self):
        """Clear everything"""
        self.current_word = []
        self.sentence = []

    def reset_sequence(self):
        """Reset detector state"""
        self.current_word = []
        self.sentence = []
        self.last_letter = ""
        self.last_letter_time = 0
        self.last_hand_time = time.time()

    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()


class Word3DetectorPool:
    """Manages Word3Detector instances per user (like SignLanguageDetectorPool)"""

    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.detectors: Dict[str, Word3Detector] = {}

    def get_detector(self, user_id: str) -> Word3Detector:
        if user_id not in self.detectors:
            self.detectors[user_id] = Word3Detector(
                self.model_path, self.labels_path
            )
            logger.info(f"Created Word3 detector for user {user_id}")
        return self.detectors[user_id]

    def remove_detector(self, user_id: str):
        if user_id in self.detectors:
            del self.detectors[user_id]
            logger.info(f"Removed Word3 detector for user {user_id}")

    def reset_detector(self, user_id: str):
        if user_id in self.detectors:
            self.detectors[user_id].reset_sequence()
