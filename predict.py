"""
predict.py
----------
Wraps the trained Keras model in a small GesturePredictor class so app.py
(and anything else) can do:

    predictor = GesturePredictor()
    gesture_name, confidence = predictor.predict(hand_landmarks)

Running this file directly (`python predict.py`) opens the webcam and shows
raw predictions with NO effects -- useful for sanity-checking a freshly
trained model before wiring up the full app.
"""

import json
import os

import numpy as np
import tensorflow as tf

from utils import landmarks_to_feature_vector

MODEL_PATH = os.path.join("models", "gesture_model.h5")
LABELS_PATH = os.path.join("models", "labels.json")


class GesturePredictor:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run train.py first."
            )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"No labels file found at {labels_path}. Run train.py first."
            )

        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r") as f:
            self.labels = json.load(f)  # index -> gesture name, in training order

    def predict(self, hand_landmarks):
        """
        hand_landmarks: a MediaPipe hand_landmarks object (21 points).

        Returns
        -------
        (gesture_name: str, confidence: float)
        """
        feature_vec = landmarks_to_feature_vector(hand_landmarks)
        feature_vec = np.expand_dims(feature_vec, axis=0)  # batch dim -> (1, 63)
        probs = self.model.predict(feature_vec, verbose=0)[0]  # shape (10,)
        best_idx = int(np.argmax(probs))
        return self.labels[best_idx], float(probs[best_idx])


# ---------------------------------------------------------------------------
# Standalone test mode: webcam + raw predictions, no effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    import mediapipe as mp

    predictor = GesturePredictor()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    print("Standalone prediction test. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture, conf = predictor.predict(hand_landmarks)
            cv2.putText(
                frame, f"{gesture} ({conf*100:.1f}%)", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )

        cv2.imshow("GestureFX - Predict Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
