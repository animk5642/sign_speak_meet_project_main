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
from collections import deque
from enum import Enum, auto
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# ── Swipe config — exact values from word3.py ────────────────────────────────
SWIPE_HISTORY_FRAMES    = 14
SWIPE_MIN_DISTANCE_PX   = 85
SWIPE_MAX_VERTICAL_PX   = 50
SWIPE_MIN_VELOCITY_PX   = 15
SWIPE_CONSISTENCY_RATIO = 0.72
SWIPE_MIN_TIME_SEC      = 0.08
SWIPE_MAX_TIME_SEC      = 1.20
SWIPE_MIN_R2            = 0.80
SWIPE_COOLDOWN_SEC      = 0.95

LEFT_HAND_LABEL  = "Left"
RIGHT_HAND_LABEL = "Right"
TRACK_LANDMARKS  = [0, 5, 9, 13, 17, 8]  # wrist + 4 knuckles + index tip


class SwipeState(Enum):
    IDLE     = auto()
    TRACKING = auto()
    COOLDOWN = auto()


class BackspaceSwipeDetector:
    """
    Watches ONLY the physical left hand.
    Fires True when a LEFTWARD swipe (negative dx) is confirmed.
    Ported directly from word3.py.
    """

    def __init__(self):
        self.history: deque = deque(maxlen=SWIPE_HISTORY_FRAMES)
        self.timestamps: deque = deque(maxlen=SWIPE_HISTORY_FRAMES)
        self.state: SwipeState = SwipeState.IDLE
        self.last_fire_t: float = 0.0
        self._start_t: float = None

    @staticmethod
    def tracking_point_px(hand_landmarks, fw, fh):
        xs = [hand_landmarks.landmark[i].x for i in TRACK_LANDMARKS]
        ys = [hand_landmarks.landmark[i].y for i in TRACK_LANDMARKS]
        return (int(np.mean(xs) * fw), int(np.mean(ys) * fh))

    @staticmethod
    def _median_smooth(history):
        pts = list(history)
        if len(pts) < 3:
            return pts
        result = [pts[0]]
        for i in range(1, len(pts) - 1):
            mx = int(np.median([pts[i-1][0], pts[i][0], pts[i+1][0]]))
            my = int(np.median([pts[i-1][1], pts[i][1], pts[i+1][1]]))
            result.append((mx, my))
        result.append(pts[-1])
        return result

    @staticmethod
    def _linear_r2(xs):
        n = len(xs)
        if n < 3:
            return 0.0
        t = np.arange(n, dtype=float)
        x = np.array(xs, dtype=float)
        coeffs = np.polyfit(t, x, 1)
        x_fit = np.polyval(coeffs, t)
        ss_res = np.sum((x - x_fit) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    def update(self, point_px: tuple) -> bool:
        """Feed one point from the LEFT hand. Returns True on confirmed left-swipe."""
        now = time.time()
        self.history.append(point_px)
        self.timestamps.append(now)

        if self.state == SwipeState.COOLDOWN:
            if (now - self.last_fire_t) >= SWIPE_COOLDOWN_SEC:
                self.state = SwipeState.IDLE
            return False

        if self.state == SwipeState.IDLE:
            if len(self.history) >= 3:
                recent_dx = self.history[-1][0] - self.history[-3][0]
                if recent_dx <= -SWIPE_MIN_VELOCITY_PX:
                    self.state = SwipeState.TRACKING
                    self._start_t = self.timestamps[-3]
            return False

        if self.state == SwipeState.TRACKING:
            if (now - self._start_t) > SWIPE_MAX_TIME_SEC:
                self._reset_to_idle()
                return False
            if self._evaluate():
                self.last_fire_t = now
                self.state = SwipeState.COOLDOWN
                self.history.clear()
                self.timestamps.clear()
                return True

        return False

    def _evaluate(self) -> bool:
        smoothed = self._median_smooth(self.history)
        if len(smoothed) < SWIPE_HISTORY_FRAMES:
            return False

        xs = [p[0] for p in smoothed]
        ys = [p[1] for p in smoothed]

        net_dx = xs[-1] - xs[0]
        net_dy = max(ys) - min(ys)

        if net_dx > -SWIPE_MIN_DISTANCE_PX:
            return False
        if net_dy > SWIPE_MAX_VERTICAL_PX:
            return False
        step_dx = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
        if max(abs(d) for d in step_dx) < SWIPE_MIN_VELOCITY_PX:
            return False
        same_dir = sum(1 for d in step_dx if d < 0)
        if same_dir / len(step_dx) < SWIPE_CONSISTENCY_RATIO:
            return False
        elapsed = self.timestamps[-1] - self._start_t
        if elapsed < SWIPE_MIN_TIME_SEC or elapsed > SWIPE_MAX_TIME_SEC:
            return False
        if self._linear_r2(xs) < SWIPE_MIN_R2:
            return False

        return True

    def _reset_to_idle(self):
        self.state = SwipeState.IDLE
        self._start_t = None

    def reset(self):
        self.history.clear()
        self.timestamps.clear()
        self._reset_to_idle()

    @property
    def tracking(self):
        return self.state == SwipeState.TRACKING


class Word3Detector:
    """
    Real-time letter detection using MediaPipe Hands + keypoint classifier.
    Ported from word3.py — same logic, no display code.
    """

    def __init__(self, model_path: str, labels_path: str,
                 letter_hold_sec: float = 1.0, word_break_sec: float = 2.0):
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

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
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

            # Flip horizontally (mirror) — word3.py does this
            frame = cv2.flip(frame, 1)

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
                    # Build landmark data for ALL hands (for visualization)
                    landmark_list = self.calc_landmark_list(w, h, lm)
                    hand_landmarks_data.append({
                        'label': label,
                        'landmarks': landmark_list,
                    })
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
