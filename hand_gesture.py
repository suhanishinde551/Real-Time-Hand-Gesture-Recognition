"""
Real-Time Hand Gesture Recognition
Python + OpenCV + MediaPipe + NumPy + Multithreading

Features
  - 21 landmarks per hand, up to 2 hands at once
  - Gestures: Thumbs Up/Down, Peace, Rock, Call Me, OK, Pinch, Fist,
              Open Palm, Pointing, Finger Heart
  - Live finger counter (0-10)
  - "Signal for Help" distress gesture detection
  - Smoothed landmarks + neon skeleton overlay

Run:   python hand_gesture.py        (press Q to quit)
"""

import threading
import time

import cv2
import mediapipe as mp
import numpy as np

# =====================================================================
# PART 1: THREADED CAMERA
# Reading frames in a separate thread keeps the video smooth, because
# the main thread never waits for the webcam.
# =====================================================================
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ok, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame

    def read(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


# =====================================================================
# PART 2: LANDMARK SMOOTHING
# Raw landmarks jitter a little. An exponential moving average (EMA)
# blends the new position with the old one:  new = a*raw + (1-a)*old
# =====================================================================
class Smoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = {}

    def apply(self, key, pts):
        if key in self.prev:
            pts = self.alpha * pts + (1 - self.alpha) * self.prev[key]
        self.prev[key] = pts
        return pts


# =====================================================================
# PART 2.5: GESTURE STABILIZER
# Majority voting across recent frames prevents single-frame flickers.
# Automatically prunes hands that have left the screen.
# =====================================================================
class GestureStabilizer:
    def __init__(self, history_len=4, threshold=3):
        self.history_len = history_len
        self.threshold = threshold
        self.history = {}
        self.current = {}
        self.last_seen = {}

    def update(self, key, gesture):
        now = time.time()
        self.last_seen[key] = now
        if key not in self.history:
            self.history[key] = []
        q = self.history[key]
        q.append(gesture)
        if len(q) > self.history_len:
            q.pop(0)

        counts = {}
        for g in q:
            counts[g] = counts.get(g, 0) + 1

        best_g, count = max(counts.items(), key=lambda item: item[1])
        if count >= self.threshold:
            self.current[key] = best_g
        return self.current.get(key, gesture)

    def prune(self, active_keys, max_idle=0.5):
        """Removes stale history for hands that have left the screen."""
        now = time.time()
        dead = [k for k in self.history if k not in active_keys and (now - self.last_seen.get(k, 0)) >= max_idle]
        for k in dead:
            self.history.pop(k, None)
            self.current.pop(k, None)
            self.last_seen.pop(k, None)


# =====================================================================
# PART 3: FINGER STATE
# MediaPipe landmark ids:
#   0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
#   (4, 8, 12, 16, 20 are the fingertips)
#
# A finger is "up" if its tip is farther from the wrist than its middle
# joint. This works at any hand rotation (not only upright hands).
# =====================================================================
def dist(a, b):
    return float(np.linalg.norm(a - b))


def finger_states(p):
    """Returns [thumb, index, middle, ring, pinky] as True (up) / False."""
    wrist = p[0]
    states = []
    # Thumb: compare distance from pinky base (17) to tip vs. to joint 3
    states.append(dist(p[4], p[17]) > dist(p[3], p[17]) * 1.15)
    # Other four fingers: tip vs. PIP joint, measured from the wrist
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        states.append(dist(p[tip], wrist) > dist(p[pip], wrist) * 1.1)
    return states


# =====================================================================
# PART 4: GESTURE CLASSIFIER
# Uses the finger states, plus a few distances for tricky gestures.
# =====================================================================
def is_finger_heart(p, fingers, palm):
    """Detects Korean Finger Heart: thumb crossed over index segment, others curled."""
    thumb, index, middle, ring, pinky = fingers
    # 1. Middle, ring, pinky must be curled down
    if middle or ring or pinky:
        return False

    wrist = p[0]
    v_idx = p[8] - p[6]
    v_idx_sq = float(np.dot(v_idx, v_idx))
    if v_idx_sq < 1e-6:
        return False

    # Projection of thumb tip (4) onto index segment [PIP(6), TIP(8)]
    # t=0 is PIP(6), t~0.5 is DIP(7), t=1.0 is TIP(8)
    t = float(np.dot(p[4] - p[6], v_idx) / v_idx_sq)
    proj = p[6] + t * v_idx
    perp_dist = dist(p[4], proj) / palm
    d_thumb_dip = dist(p[4], p[7]) / palm

    # A. Index must be extended outward, not curled in palm like a fist
    if dist(p[8], wrist) < dist(p[5], wrist) * 0.95:
        return False

    # B. Thumb crosses index body (t between 0.25 and 0.88; Pinch is ~1.0)
    crosses_segment = (0.25 <= t <= 0.88)
    # C. Thumb tip is close to index segment line
    close_to_segment = perp_dist < 0.22
    # D. Thumb tip is near index DIP joint 7
    near_dip = d_thumb_dip < 0.28
    # E. Index tip extends past thumb tip
    idx_past_thumb = dist(p[8], wrist) > dist(p[4], wrist) * 0.92

    return crosses_segment and close_to_segment and near_dip and idx_past_thumb


def classify(p, fingers):
    thumb, index, middle, ring, pinky = fingers
    palm = dist(p[0], p[9]) + 1e-6          # hand size, for scale-free tests
    pinch_d = dist(p[4], p[8]) / palm        # thumb tip <-> index tip

    # OK: thumb and index touch, other three fingers up
    if pinch_d < 0.30 and middle and ring and pinky:
        return "OK"

    # Finger Heart: thumb crossed over index segment, others curled
    if is_finger_heart(p, fingers, palm):
        return "Finger Heart"

    # Pinch: thumb and index touch, others not up (not a closed fist)
    if pinch_d < 0.30 and not (middle and ring and pinky) and any(fingers):
        return "Pinch"

    # Thumbs up / down: only thumb extended, check which way it points
    if thumb and not (index or middle or ring or pinky):
        if p[4][1] < p[2][1] - 0.4 * palm:
            return "Thumbs Up"
        if p[4][1] > p[2][1] + 0.4 * palm:
            return "Thumbs Down"

    if index and middle and not ring and not pinky and not thumb:
        return "Peace"
    if index and pinky and not middle and not ring:
        return "Rock"
    if thumb and pinky and not (index or middle or ring):
        return "Call Me"
    if index and not (middle or ring or pinky):
        return "Pointing"
    if all(fingers):
        return "Open Palm"
    if not any(fingers):
        return "Fist"
    return ""


# =====================================================================
# PART 5: SIGNAL FOR HELP  (the domestic-violence distress gesture)
# Real signal: 1) show open palm  2) tuck thumb into the palm
#              3) fold the four fingers down over the thumb.
# We detect it as a 2-step sequence within a few seconds:
#   Step A: thumb tucked in palm + four fingers up
#   Step B: thumb tucked in palm + four fingers folded (thumb trapped)
# =====================================================================
class HelpSignalDetector:
    def __init__(self, window=4.0, alert_time=4.0):
        self.window = window            # max seconds between step A and B
        self.alert_time = alert_time    # how long the alert stays on screen
        self.armed_at = 0.0
        self.alert_until = 0.0

    @staticmethod
    def thumb_tucked(p):
        palm = dist(p[0], p[9]) + 1e-6
        # thumb tip sits close to the base of middle/ring finger = in palm
        d = min(dist(p[4], p[9]), dist(p[4], p[13])) / palm
        return d < 0.55

    def update(self, p, fingers):
        now = time.time()
        four_up = all(fingers[1:])
        four_down = not any(fingers[1:])
        tucked = self.thumb_tucked(p)

        if tucked and four_up:
            self.armed_at = now                       # Step A
        elif tucked and four_down and now - self.armed_at < self.window:
            self.alert_until = now + self.alert_time  # Step B -> ALERT
            self.armed_at = 0.0

    @property
    def active(self):
        return time.time() < self.alert_until


# =====================================================================
# PART 6: DRAWING - NEON SKELETON
# Trick for a glow: draw thick lines on a blank layer, blur it,
# add it to the frame, then draw thin bright lines on top.
# =====================================================================
CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
NEON = [(255, 255, 0), (255, 0, 255)]   # BGR: cyan, magenta (one per hand)


def draw_neon(frame, hands_pts):
    glow = np.zeros_like(frame)
    for i, pts in enumerate(hands_pts):
        color = NEON[i % 2]
        for a, b in CONNECTIONS:
            cv2.line(glow, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 9)
    glow = cv2.GaussianBlur(glow, (0, 0), 9)
    frame = cv2.addWeighted(frame, 1.0, glow, 1.2, 0)

    for i, pts in enumerate(hands_pts):
        color = NEON[i % 2]
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2)
        for pt in pts:
            cv2.circle(frame, tuple(pt.astype(int)), 4, (255, 255, 255), -1)
    return frame


def put_text(img, text, org, scale=1.0, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_heart(img, center, size=10, color=(0, 0, 255)):
    """Draws a clean filled heart symbol (♥)."""
    x, y = int(center[0]), int(center[1])
    r = int(size * 0.5)
    cv2.circle(img, (x - r, y - r // 2), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + r, y - r // 2), r, color, -1, cv2.LINE_AA)
    pts = np.array([[x - 2 * r, y - r // 4], [x + 2 * r, y - r // 4], [x, y + int(size * 1.1)]], dtype=np.int32)
    cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)


# =====================================================================
# PART 7: MAIN LOOP
# =====================================================================
def main():
    cam = CameraStream()
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    smoother = Smoother(alpha=0.5)
    stabilizer = GestureStabilizer(history_len=4, threshold=3)
    help_detector = HelpSignalDetector()
    prev_time = time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)                 # mirror view
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB
        result = hands.process(rgb)

        hands_pts, labels, total = [], [], 0
        active_keys = set()
        if result.multi_hand_landmarks:
            for idx, (lms, hd) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                side = hd.classification[0].label   # "Left" / "Right"
                key = f"{side}{idx}"
                active_keys.add(key)
                raw = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark])
                pts = smoother.apply(key, raw)

                fingers = finger_states(pts)
                total += sum(fingers)
                raw_gesture = classify(pts, fingers)
                gesture = stabilizer.update(key, raw_gesture)
                help_detector.update(pts, fingers)

                hands_pts.append(pts)
                labels.append((side, gesture, pts))

        stabilizer.prune(active_keys)

        if hands_pts:
            frame = draw_neon(frame, hands_pts)
            for side, gesture, pts in labels:
                x, y = pts[:, 0].min(), pts[:, 1].min()
                text = f"{side}: {gesture}" if gesture else side
                put_text(frame, text, (int(x), max(30, int(y) - 15)), 0.8, (0, 255, 255))

                if gesture == "Finger Heart":
                    # Draw a heart symbol (♥) right next to the gesture label
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    heart_x = int(x) + tw + 16
                    heart_y = max(30, int(y) - 15) - th // 2
                    draw_heart(frame, (heart_x, heart_y), size=11, color=(50, 50, 255))
                    # Draw a floating mini heart above the crossed fingers
                    finger_heart_pos = ((pts[4] + pts[8]) / 2).astype(int)
                    finger_heart_pos[1] -= 25
                    draw_heart(frame, finger_heart_pos, size=13, color=(180, 50, 255))

        # HUD
        put_text(frame, f"Fingers: {total}", (20, 50), 1.4, (0, 255, 0), 3)
        fps = 1.0 / max(time.time() - prev_time, 1e-6)
        prev_time = time.time()
        put_text(frame, f"FPS: {fps:.0f}", (20, 90), 0.7)

        if help_detector.active:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 12)
            put_text(frame, "SIGNAL FOR HELP DETECTED!", (w // 2 - 330, h - 40),
                     1.4, (0, 0, 255), 3)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cam.stop()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
