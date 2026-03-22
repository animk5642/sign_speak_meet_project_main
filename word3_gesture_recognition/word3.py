import os
import csv
import copy
import time
import argparse
import itertools
from collections import deque
from enum import Enum, auto

import cv2 as cv
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands
import mediapipe.python.solutions.drawing_utils

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


datasetdir = "model/dataset/dataset 1"

# ══════════════════════════════════════════════════════════════════════════════
#  SWIPE CONFIG  — left hand swipes LEFT → backspace
# ══════════════════════════════════════════════════════════════════════════════
SWIPE_HISTORY_FRAMES    = 14
SWIPE_MIN_DISTANCE_PX   = 85
SWIPE_MAX_VERTICAL_PX   = 50
SWIPE_MIN_VELOCITY_PX   = 15
SWIPE_CONSISTENCY_RATIO = 0.72
SWIPE_MIN_TIME_SEC      = 0.08
SWIPE_MAX_TIME_SEC      = 1.20
SWIPE_MIN_R2            = 0.80
SWIPE_COOLDOWN_SEC      = 0.95
# ══════════════════════════════════════════════════════════════════════════════

# From screenshot: MediaPipe labels physical left hand as "Left"
LEFT_HAND_LABEL  = "Left"
RIGHT_HAND_LABEL = "Right"

TRACK_LANDMARKS = [0, 5, 9, 13, 17, 8]   # wrist + 4 knuckles + index tip


class SwipeState(Enum):
    IDLE     = auto()
    TRACKING = auto()
    COOLDOWN = auto()


def tracking_point_px(hand_landmarks, fw, fh):
    xs = [hand_landmarks.landmark[i].x for i in TRACK_LANDMARKS]
    ys = [hand_landmarks.landmark[i].y for i in TRACK_LANDMARKS]
    return (int(np.mean(xs) * fw), int(np.mean(ys) * fh))


def median_smooth(history):
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


def linear_r2(xs):
    n = len(xs)
    if n < 3:
        return 0.0
    t      = np.arange(n, dtype=float)
    x      = np.array(xs, dtype=float)
    coeffs = np.polyfit(t, x, 1)
    x_fit  = np.polyval(coeffs, t)
    ss_res = np.sum((x - x_fit) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


class BackspaceSwipeDetector:
    """
    Watches ONLY the physical left hand.
    Fires True when a LEFTWARD swipe (negative dx) is confirmed.
    """

    def __init__(self):
        self.history    : deque      = deque(maxlen=SWIPE_HISTORY_FRAMES)
        self.timestamps : deque      = deque(maxlen=SWIPE_HISTORY_FRAMES)
        self.state      : SwipeState = SwipeState.IDLE
        self.last_fire_t: float      = 0.0
        self._start_t   : float | None = None

    def update(self, point_px: tuple) -> bool:
        """Feed one point from the LEFT hand. Returns True on confirmed left-swipe."""
        now = time.time()
        self.history.append(point_px)
        self.timestamps.append(now)

        # COOLDOWN
        if self.state == SwipeState.COOLDOWN:
            if (now - self.last_fire_t) >= SWIPE_COOLDOWN_SEC:
                self.state = SwipeState.IDLE
            return False

        # IDLE → detect leftward motion start (dx is NEGATIVE for left swipe)
        if self.state == SwipeState.IDLE:
            if len(self.history) >= 3:
                recent_dx = self.history[-1][0] - self.history[-3][0]
                if recent_dx <= -SWIPE_MIN_VELOCITY_PX:          # moving LEFT
                    self.state    = SwipeState.TRACKING
                    self._start_t = self.timestamps[-3]
            return False

        # TRACKING → accumulate & evaluate
        if self.state == SwipeState.TRACKING:
            if (now - self._start_t) > SWIPE_MAX_TIME_SEC:
                self._reset_to_idle()
                return False
            if self._evaluate():
                self.last_fire_t = now
                self.state       = SwipeState.COOLDOWN
                self.history.clear()
                self.timestamps.clear()
                return True

        return False

    def _evaluate(self) -> bool:
        smoothed = median_smooth(self.history)
        if len(smoothed) < SWIPE_HISTORY_FRAMES:
            return False

        xs = [p[0] for p in smoothed]
        ys = [p[1] for p in smoothed]

        net_dx = xs[-1] - xs[0]          # negative = leftward
        net_dy = max(ys) - min(ys)

        # Must be leftward
        if net_dx > -SWIPE_MIN_DISTANCE_PX:
            return False
        # Mostly horizontal
        if net_dy > SWIPE_MAX_VERTICAL_PX:
            return False
        # Peak velocity
        step_dx = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
        if max(abs(d) for d in step_dx) < SWIPE_MIN_VELOCITY_PX:
            return False
        # Consistency — all steps go left (negative)
        same_dir = sum(1 for d in step_dx if d < 0)
        if same_dir / len(step_dx) < SWIPE_CONSISTENCY_RATIO:
            return False
        # Time window
        elapsed = self.timestamps[-1] - self._start_t
        if elapsed < SWIPE_MIN_TIME_SEC or elapsed > SWIPE_MAX_TIME_SEC:
            return False
        # Path linearity
        if linear_r2(xs) < SWIPE_MIN_R2:
            return False

        return True

    def _reset_to_idle(self):
        self.state    = SwipeState.IDLE
        self._start_t = None

    def reset(self):
        self.history.clear()
        self.timestamps.clear()
        self._reset_to_idle()

    @property
    def tracking(self):
        return self.state == SwipeState.TRACKING


# ── Visual helpers ────────────────────────────────────────────────────────────

def draw_swipe_trail(image, history, active: bool):
    pts = list(history)
    if len(pts) < 2:
        return image
    for i in range(1, len(pts)):
        ratio = i / len(pts)
        # Orange-red trail for left-swipe backspace
        color = (0, int(120 * ratio), 255) if active else (50, 50, 50)
        cv.line(image, pts[i-1], pts[i], color, max(1, int(ratio*4)), cv.LINE_AA)
    if pts:
        cv.circle(image, pts[-1], 7, (0, 100, 255) if active else (80,80,80), -1)
        cv.circle(image, pts[-1], 7, (0,0,0), 1)
    return image


def draw_backspace_feedback(image, show_until):
    if time.time() < show_until:
        h, w = image.shape[:2]
        overlay = image.copy()
        cv.rectangle(overlay, (w//2-200, h//2-48), (w//2+200, h//2+48), (10,10,10), -1)
        cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        cv.rectangle(image, (w//2-200, h//2-48), (w//2+200, h//2+48), (40,120,255), 2)
        cv.putText(image, "◄ BACKSPACE",
                   (w//2-185, h//2+16),
                   cv.FONT_HERSHEY_SIMPLEX, 1.1, (40, 120, 255), 3, cv.LINE_AA)
    return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width",  type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",  type=int,   default=0.5)
    return parser.parse_args()


def main():
    args = get_args()

    cap_device = args.device
    cap_width  = args.width
    cap_height = args.height

    use_static_image_mode    = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence  = args.min_tracking_confidence
    use_brect                = True

    # ── Word / sentence state ─────────────────────────────────────────────────
    current_word     = []
    sentence         = []
    last_letter      = ""
    last_letter_time = 0
    last_hand_time   = time.time()
    LETTER_HOLD_SEC  = 1.0
    WORD_BREAK_SEC   = 2.0
    # ─────────────────────────────────────────────────────────────────────────

    swiper               = BackspaceSwipeDetector()
    backspace_show_until = 0.0

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    with open("model/keypoint_classifier/keypoint_classifier_label.csv",
              encoding="utf-8-sig") as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    mode = 0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)

        if key == 27:
            break
        number, mode = select_mode(key, mode)

        # Keyboard fallback
        if key in (ord('b'), ord('B')):
            if current_word:   current_word.pop()
            elif sentence:     current_word = list(sentence.pop())
        if key in (ord('c'), ord('C')):
            current_word = []
            sentence     = []

        ret, image = cap.read()
        if not ret:
            break
        image       = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # ── Dataset mode ──────────────────────────────────────────────────────
        if mode == 2:
            loading_img = cv.imread("./assets/om606.png", cv.IMREAD_COLOR)
            cv.putText(loading_img, "Loading...", (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 4, cv.LINE_AA)
            cv.imshow("Hand Gesture Recognition", loading_img)
            cv.waitKey(1000)
            imglabel = -1
            for imgclass in os.listdir(datasetdir):
                imglabel += 1
                numofimgs = 0
                for img_name in os.listdir(os.path.join(datasetdir, imgclass)):
                    numofimgs += 1
                    imgpath = os.path.join(datasetdir, imgclass, img_name)
                    try:
                        img = cv.imread(imgpath)
                        dbg = copy.deepcopy(img)
                        for _ in [1, 2]:
                            img.flags.writeable = False
                            res = hands.process(img)
                            img.flags.writeable = True
                            if res.multi_hand_landmarks:
                                for hl, _ in zip(res.multi_hand_landmarks,
                                                 res.multi_handedness):
                                    ll = calc_landmark_list(dbg, hl)
                                    pp = pre_process_landmark(ll)
                                    logging_csv(imglabel, mode, pp)
                            img = cv.flip(img, 0)
                    except Exception:
                        print(f"Issue with image {imgpath}")
                print(f"Num of image of class {imglabel}: {numofimgs}")
            mode = 1
            print("End of job!")
            break

        # ── Normal detection mode ─────────────────────────────────────────────
        else:
            if results.multi_hand_landmarks is not None:
                last_hand_time = time.time()

                # ── Split detected hands into left / right ────────────────────
                left_lm  = None;  left_hd  = None
                right_lm = None;  right_hd = None

                for lm, hd in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
                    label = hd.classification[0].label   # "Left" or "Right"
                    if label == LEFT_HAND_LABEL:
                        left_lm = lm;  left_hd = hd
                    else:
                        right_lm = lm; right_hd = hd

                # ── LEFT HAND — swipe detector only ──────────────────────────
                if left_lm is not None:
                    track_pt      = tracking_point_px(left_lm, cap_width, cap_height)
                    did_backspace = swiper.update(track_pt)

                    debug_image = draw_swipe_trail(debug_image, swiper.history,
                                                   swiper.tracking)

                    if did_backspace:
                        if current_word:   current_word.pop()
                        elif sentence:     current_word = list(sentence.pop())
                        backspace_show_until = time.time() + 0.75

                    # Draw left hand — show "<< " label while swiping
                    brect_l = calc_bounding_rect(debug_image, left_lm)
                    ll_list = calc_landmark_list(debug_image, left_lm)
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect_l)
                    debug_image = draw_landmarks(debug_image, ll_list)
                    swipe_lbl   = "<<" if swiper.tracking else ""
                    debug_image = draw_info_text(debug_image, brect_l, left_hd, swipe_lbl)
                else:
                    swiper.reset()   # left hand gone — reset swipe state

                # ── RIGHT HAND — letter classification only ───────────────────
                if right_lm is not None:
                    brect         = calc_bounding_rect(debug_image, right_lm)
                    landmark_list = calc_landmark_list(debug_image, right_lm)
                    pre_processed = pre_process_landmark(landmark_list)

                    logging_csv(number, mode, pre_processed)

                    hand_sign_id    = keypoint_classifier(pre_processed)
                    detected_letter = keypoint_classifier_labels[hand_sign_id]

                    now = time.time()
                    # Freeze letter hold timer if left hand is mid-swipe
                    if not swiper.tracking:
                        if detected_letter == last_letter:
                            if (now - last_letter_time) >= LETTER_HOLD_SEC:
                                if not current_word or current_word[-1] != detected_letter:
                                    current_word.append(detected_letter)
                                    last_letter_time = now
                        else:
                            last_letter      = detected_letter
                            last_letter_time = now
                    else:
                        # Keep updating detected letter but don't advance timer
                        last_letter      = detected_letter
                        last_letter_time = now

                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, right_hd, detected_letter)
                else:
                    last_letter = ""

            else:
                # ── No hand at all → word break = space ───────────────────────
                swiper.reset()
                if (time.time() - last_hand_time) >= WORD_BREAK_SEC:
                    if current_word:
                        sentence.append("".join(current_word))
                        current_word = []
                last_letter = ""

        # ── Overlays ──────────────────────────────────────────────────────────
        debug_image = draw_backspace_feedback(debug_image, backspace_show_until)

        h, w = debug_image.shape[:2]
        cv.rectangle(debug_image, (0, h-105), (w, h), (0,0,0), -1)

        cv.putText(debug_image,
                   "Left hand swipe LEFT = backspace  |  No hand 2s = space  |  C = clear  |  ESC = quit",
                   (10, h-83), cv.FONT_HERSHEY_SIMPLEX, 0.40, (130,130,130), 1, cv.LINE_AA)

        # Left-hand swipe state indicator
        if swiper.tracking:
            cv.putText(debug_image, "◄◄ SWIPING (LEFT HAND)",
                       (w-270, 60), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,140,255), 2, cv.LINE_AA)

        # Letter hold bar (right hand)
        if last_letter and last_letter_time > 0 and not swiper.tracking:
            held  = min(time.time() - last_letter_time, LETTER_HOLD_SEC)
            bar_w = int((held / LETTER_HOLD_SEC) * 200)
            cv.rectangle(debug_image, (10, h-70), (10+bar_w, h-58), (0,255,120), -1)
            cv.rectangle(debug_image, (10, h-70), (210,      h-58), (80,80,80), 1)
            cv.putText(debug_image, f"Holding: {last_letter}", (220, h-58),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,120), 1, cv.LINE_AA)

        current_str  = "".join(current_word)
        sentence_str = " ".join(sentence)

        cv.putText(debug_image, f"Word:     {current_str}",
                   (10, h-38), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,180), 2, cv.LINE_AA)
        cv.putText(debug_image, f"Sentence: {sentence_str}",
                   (10, h-12), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1, cv.LINE_AA)

        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


# ── Helper functions ──────────────────────────────────────────────────────────

def select_mode(key, mode):
    number = -1
    if 65 <= key <= 90:   number = key - 65
    if key == 110:        mode = 0
    if key == 107:        mode = 1
    if key == 100:        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    arr = np.array([
        [min(int(lm.x*iw), iw-1), min(int(lm.y*ih), ih-1)]
        for lm in landmarks.landmark
    ], dtype=int)
    x, y, w, h = cv.boundingRect(arr)
    return [x, y, x+w, y+h]


def calc_landmark_list(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    return [
        [min(int(lm.x*iw), iw-1), min(int(lm.y*ih), ih-1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    temp   = copy.deepcopy(landmark_list)
    bx, by = temp[0]
    for pt in temp:
        pt[0] -= bx
        pt[1] -= by
    flat    = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat))
    return [v/max_val for v in flat]


def logging_csv(number, mode, landmark_list):
    if mode in (1, 2) and 0 <= number <= 35:
        with open("model/keypoint_classifier/keypoint.csv", "a", newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])


def draw_landmarks(image, landmark_point):
    connections = [
        (2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
        (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
        (0,1),(1,2),(2,5),(5,9),(9,13),(13,17),(17,0),
    ]
    for a, b in connections:
        cv.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]), (0,0,0), 6)
        cv.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]), (255,255,255), 2)
    for i, pt in enumerate(landmark_point):
        r = 8 if i in (4,8,12,16,20) else 5
        cv.circle(image, tuple(pt), r, (255,255,255), -1)
        cv.circle(image, tuple(pt), r, (0,0,0), 1)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0],brect[1]), (brect[2],brect[3]), (0,0,0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0],brect[1]), (brect[2],brect[1]-22), (0,0,0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text += ": " + hand_sign_text
    cv.putText(image, info_text, (brect[0]+5, brect[1]-4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:"+str(fps), (10,30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:"+str(fps), (10,30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
    mode_string = ["Logging Key Point", "Capturing Landmarks From Dataset"]
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:"+mode_string[mode-1], (10,90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:"+str(number), (10,110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    return image


if __name__ == "__main__":
    main()