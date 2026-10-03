"""
collect_data.py
----------------
Interactive tool to build the gesture-landmark dataset used by train.py.

HOW IT WORKS
------------
1. Opens your webcam and runs MediaPipe Hands to find 21 hand landmarks.
2. You select a gesture (press 0-9, mapped to the list printed on screen).
3. You capture samples of that gesture:
     - press and HOLD SPACE to auto-capture ~15 samples/second
     - or press 'c' once to capture a single sample
4. Each sample is the 63-number normalized feature vector (see utils.py)
   plus the gesture label, appended to dataset/gestures.csv.
5. Press 's' any time to save progress, 'q' to save and quit.

TIPS FOR A GOOD DATASET
------------------------
- Aim for at least 150-300 samples per gesture.
- Vary hand distance from camera, slight rotation, and position in frame
  (the normalization in utils.py removes translation/scale differences,
  but varying them anyway makes MediaPipe's own landmark detection noise
  part of your training distribution, which makes the model more robust).
- Try both hands if you want the app to work for left- and right-handed
  users.
"""

import csv
import os
import time

import cv2
import mediapipe as mp

from utils import GESTURES, GESTURE_DISPLAY, landmarks_to_feature_vector

DATASET_PATH = os.path.join("dataset", "gestures.csv")
AUTO_CAPTURE_INTERVAL = 1.0 / 15.0  # seconds between auto-captures while holding SPACE


def ensure_csv_header():
    """Create the CSV with a header row if it doesn't exist yet."""
    if not os.path.exists(DATASET_PATH):
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"f{i}" for i in range(63)] + ["label"]
            writer.writerow(header)


def count_existing_samples():
    """Read the CSV (if any) and return a dict {gesture_name: count}."""
    counts = {g: 0 for g in GESTURES}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label")
                if label in counts:
                    counts[label] += 1
    return counts


def main():
    ensure_csv_header()
    counts = count_existing_samples()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    selected_idx = 0
    last_auto_capture = 0.0

    print("=" * 60)
    print("GestureFX - Data Collection")
    print("=" * 60)
    for i, g in enumerate(GESTURES):
        print(f"  [{i}] {GESTURE_DISPLAY[g]}")
    print("Controls: 0-9 select gesture | SPACE hold=auto-capture | "
          "c=single capture | s=save | q=save & quit")
    print("=" * 60)

    csv_file = open(DATASET_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        feature_vec = None
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            feature_vec = landmarks_to_feature_vector(hand_landmarks)

        # ---- capture logic ----
        now = time.time()
        captured_this_frame = False
        key = cv2.waitKey(1) & 0xFF

        # NOTE: a plain OpenCV waitKey loop can't detect key-up events
        # reliably across platforms, so we treat SPACE as "capture on every
        # frame the key is polled as pressed" and just rate-limit it. This
        # is intentionally simple for a beginner project.
        if key == ord(" ") and feature_vec is not None:
            if now - last_auto_capture >= AUTO_CAPTURE_INTERVAL:
                writer.writerow(list(feature_vec) + [GESTURES[selected_idx]])
                counts[GESTURES[selected_idx]] += 1
                last_auto_capture = now
                captured_this_frame = True

        if key == ord("c") and feature_vec is not None:
            writer.writerow(list(feature_vec) + [GESTURES[selected_idx]])
            counts[GESTURES[selected_idx]] += 1
            captured_this_frame = True

        if key == ord("s"):
            csv_file.flush()
            print("Progress saved to", DATASET_PATH)

        if key == ord("q"):
            break

        if ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < len(GESTURES):
                selected_idx = idx

        # ---- on-screen UI ----
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        current_gesture = GESTURES[selected_idx]
        cv2.putText(
            frame,
            f"Selected: {GESTURE_DISPLAY[current_gesture]} ({selected_idx})",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.putText(
            frame,
            f"Samples for this gesture: {counts[current_gesture]}",
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        total = sum(counts.values())
        cv2.putText(
            frame,
            f"Total dataset size: {total}   [SPACE=capture  c=single  s=save  q=quit]",
            (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )

        if captured_this_frame:
            cv2.circle(frame, (w - 40, 40), 15, (0, 255, 0), -1)

        if feature_vec is None:
            cv2.putText(
                frame, "No hand detected", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        cv2.imshow("GestureFX - Data Collection", frame)

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal sample counts:")
    for g in GESTURES:
        print(f"  {GESTURE_DISPLAY[g]:15s}: {counts[g]}")
    print(f"\nDataset saved at: {DATASET_PATH}")


if __name__ == "__main__":
    main()
