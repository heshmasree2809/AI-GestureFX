"""
app.py
------
GestureFX main application.

Pipeline per frame:
  1. Read a frame from the webcam, flip it (mirror) for natural interaction.
  2. Run MediaPipe Hands -> 21 landmarks (if a hand is visible).
  3. Feed the normalized landmarks into our trained Keras model
     (via predict.py) -> (gesture_name, confidence).
  4. STABILITY CHECK: only "confirm" a gesture if the model has predicted
     the same gesture for ~0.5 seconds straight (prevents flicker from
     triggering effects on every noisy single-frame misclassification).
  5. On a newly confirmed gesture, ask EffectsManager to trigger its
     animation. EffectsManager enforces its own 2-second cooldown per
     gesture internally, so holding a pose doesn't spam-retrigger it.
  6. Draw landmarks, the predicted gesture + confidence, and any active
     effect animations, then show the frame.

Controls: press 'q' to quit.
"""

import collections
import time

import cv2
import mediapipe as mp

from effects import EffectsManager
from predict import GesturePredictor
from utils import GESTURE_DISPLAY, landmarks_to_pixel_coords

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75      # ignore predictions the model isn't sure about
STABILITY_WINDOW_SECONDS = 0.5    # how long a gesture must be held to "confirm"
EFFECT_COOLDOWN_SECONDS = 2.0     # min seconds between re-triggers of same effect
HISTORY_MAXLEN = 30                # ring buffer size for prediction history


class GestureStabilizer:
    """
    Tracks recent (timestamp, gesture_name) predictions and reports the
    "confirmed" gesture only once the same gesture has consistently been
    the top prediction for at least STABILITY_WINDOW_SECONDS.

    This directly implements the "avoid flickering" requirement: a single
    misclassified frame won't trigger an effect, but ~0.5s of consistent
    agreement will.
    """

    def __init__(self, window_seconds=STABILITY_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self.history = collections.deque(maxlen=HISTORY_MAXLEN)
        self.last_confirmed = None

    def update(self, gesture_name, timestamp):
        self.history.append((timestamp, gesture_name))

        # keep only entries within the stability window
        cutoff = timestamp - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        if not self.history:
            return None

        # require the window to actually span ~window_seconds of real time
        # (otherwise a burst of frames at start-up could false-trigger)
        span = self.history[-1][0] - self.history[0][0]
        if span < self.window_seconds * 0.8:
            return None

        # all predictions in the window must agree
        names = {g for _, g in self.history}
        if len(names) == 1:
            confirmed = names.pop()
            if confirmed != self.last_confirmed:
                self.last_confirmed = confirmed
                return confirmed  # newly confirmed gesture
        return None


def draw_header(frame, gesture_name, confidence, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    if gesture_name:
        label = GESTURE_DISPLAY.get(gesture_name, gesture_name)
        text = f"{label}  ({confidence*100:.0f}%)"
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 200, 255)
    else:
        text = "No hand detected"
        color = (0, 0, 255)

    cv2.putText(frame, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "GestureFX  |  press 'q' to quit", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    predictor = GesturePredictor()
    effects_manager = EffectsManager(cooldown_seconds=EFFECT_COOLDOWN_SECONDS)
    stabilizer = GestureStabilizer(window_seconds=STABILITY_WINDOW_SECONDS)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    prev_time = time.time()
    fps = 0.0

    print("GestureFX running. Press 'q' in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_name, confidence = None, 0.0
        now = time.time()

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            gesture_name, confidence = predictor.predict(hand_landmarks)

            if confidence >= CONFIDENCE_THRESHOLD:
                confirmed = stabilizer.update(gesture_name, now)
                if confirmed is not None:
                    landmarks_px = landmarks_to_pixel_coords(hand_landmarks, w, h)
                    effects_manager.trigger(confirmed, landmarks_px, w, h)
            # if confidence is low, we simply don't feed it into the
            # stabilizer -- an uncertain frame shouldn't count as evidence
            # either for or against the currently-held gesture.

        # ---- render ----
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

        effects_manager.update_and_draw(frame, dt)
        draw_header(frame, gesture_name, confidence, fps)

        cv2.imshow("GestureFX - AI Gesture Effects Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
