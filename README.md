# GestureFX – AI Gesture Effects Camera ✋✨

A beginner-friendly computer vision project that opens your webcam, recognizes
10 hand gestures **using a custom-trained TensorFlow/Keras model** (not
MediaPipe's built-in gesture recognizer), and plays a fun animated effect for
each one — floating balloons, falling hearts, confetti, fire, lightning, and
more.

| Gesture | Effect |
|---|---|
| ✌️ Peace | 🎈 Floating balloons |
| ❤️ Heart Hands | ❤️ Falling hearts |
| 👍 Thumbs Up | 🎉 Confetti burst |
| ✋ Open Palm | ✨ Sparkles around the hand |
| 👋 Wave | 🌈 Rainbow trail |
| 👌 OK | ⭐ Star burst |
| 🤟 Love Sign | 💖 Pink glowing particles |
| ✊ Fist | 💥 Comic-style explosion |
| ☝️ Point Up | ⚡ Lightning from fingertip |
| 🤘 Rock Sign | 🔥 Fire effect around the hand |

---

## How MediaPipe, TensorFlow, and OpenCV work together

It's easy to assume "MediaPipe does the gesture recognition," but in this
project each library has one specific job:

1. **MediaPipe Hands** — pure hand *detection*. Every frame, it finds the
   hand in the image and returns **21 (x, y, z) landmark points** (fingertips,
   knuckles, wrist). It knows nothing about "peace sign" or "thumbs up" — it
   just reports geometry.
2. **Our TensorFlow/Keras model** — the actual *classifier*. We convert the
   21 landmarks into a normalized 63-number vector (`utils.py`) and feed it
   into a small Dense neural network we trained ourselves on our own
   recorded examples. This is the piece that actually knows what "peace
   sign" looks like.
3. **OpenCV** — the *display layer*. It captures webcam frames, draws the
   hand skeleton, renders text overlays, and draws every particle-based
   animation (`effects.py`) directly onto the frame with simple shapes
   (circles, lines, polygons) — no external image assets needed.

So the pipeline is:

```
Webcam frame (OpenCV)
      │
      ▼
21 hand landmarks (MediaPipe Hands)
      │
      ▼
63-number normalized feature vector (utils.py)
      │
      ▼
Gesture name + confidence (our trained Keras model, predict.py)
      │
      ▼
0.5s stability check + 2s cooldown (app.py)
      │
      ▼
Particle animation (effects.py) drawn back onto the frame (OpenCV)
```

---

## Project structure

```
GestureFX/
│── dataset/            # gestures.csv (landmark samples) lives here
│── models/              # trained model, labels.json, training plots
│── assets/               # (reserved — effects are drawn with OpenCV, no assets required)
│── utils.py                # shared constants + landmark normalization (used by every script)
│── collect_data.py          # webcam tool to record labeled gesture samples
│── train.py                  # trains the Keras model on dataset/gestures.csv
│── predict.py                  # GesturePredictor class + standalone test mode
│── effects.py                    # particle animation engine (10 effects + EffectsManager)
│── app.py                          # main application — run this to use GestureFX
│── requirements.txt
│── README.md
```

---

## 1. Setup

```bash
# (recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Requires Python 3.9–3.11 (MediaPipe does not yet support every Python
version — check MediaPipe's docs if `pip install mediapipe` fails).

---

## 2. Collect the dataset

```bash
python collect_data.py
```

- A window opens showing your webcam feed with hand landmarks drawn on it.
- Press a **number key 0–9** to select which gesture you're about to record
  (the mapping is printed in the terminal and shown on screen).
- **Hold `SPACE`** to auto-capture ~15 samples/second while you hold the
  pose — move your hand slightly (angle, distance, position) between bursts
  for a more robust dataset.
- Press `c` to capture a single sample instead.
- Press `s` any time to save progress, `q` to save and quit.

**Aim for at least 150–300 samples per gesture.** Everything is appended to
`dataset/gestures.csv` as rows of `f0..f62,label`.

> No dataset yet? That's expected — this script *is* the dataset creator.
> Just run it and start recording.

---

## 3. Train the model

```bash
python train.py
```

This will:
- Load `dataset/gestures.csv`
- Split it 80/20 into train/validation (stratified by gesture)
- Train a small Keras `Sequential` model (Dense 128 → 64 → 32 → softmax(10))
  with dropout and early stopping
- Print final validation accuracy
- Save:
  - `models/gesture_model.h5` — the trained model
  - `models/labels.json` — maps model output index → gesture name
  - `models/training_history.png` — accuracy & loss curves

If accuracy is low, the usual fix is: collect more samples, especially for
the gestures the model confuses with each other (similar hand shapes like
OK vs. Love Sign benefit from extra, varied examples).

---

## 4. Run the app

```bash
python app.py
```

- Show any of the 10 gestures to the camera.
- The predicted gesture and confidence appear at the top of the frame.
- Hold the gesture steady for about **half a second** — this "stability
  window" avoids triggering effects from single flickery misclassified
  frames.
- Once confirmed, the matching animation plays. Each gesture has its own
  **2-second cooldown** before it can re-trigger, so holding the pose
  doesn't spam the animation nonstop.
- Press `q` to quit.

Want to sanity-check the model without any animations first?

```bash
python predict.py
```

This opens a bare webcam window showing raw predicted gesture + confidence,
no effects — useful for confirming the model works before wiring up the
full app.

---

## Tuning

All the important thresholds live at the top of `app.py`:

```python
CONFIDENCE_THRESHOLD = 0.75      # minimum model confidence to consider a prediction
STABILITY_WINDOW_SECONDS = 0.5    # how long a gesture must be held to "confirm"
EFFECT_COOLDOWN_SECONDS = 2.0     # cooldown before the same effect can fire again
```

---

## Troubleshooting

- **`ModuleNotFoundError: mediapipe`** — MediaPipe wheels aren't published
  for every Python version/OS combo; check you're on Python 3.9–3.11 (64-bit).
- **Webcam won't open** — try changing `cv2.VideoCapture(0)` to `1` in
  `app.py` / `collect_data.py` / `predict.py` if you have multiple cameras.
- **Model confuses two gestures** — record more samples for those two
  gestures specifically, including edge cases (hand slightly rotated,
  farther from camera, etc.), then re-run `train.py`.
- **Effects feel laggy** — MediaPipe + TensorFlow inference is the
  bottleneck, not OpenCV drawing; lowering webcam resolution
  (`cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)`) usually helps most on modest
  hardware.

---


---

## How it all fits together (code-level)

- `utils.py` defines the **single source of truth** for the gesture list and
  the landmark-normalization math, so training and inference can never
  drift out of sync with each other.
- `collect_data.py` and `predict.py`/`app.py` both call
  `landmarks_to_feature_vector()` from `utils.py` — same function, same math,
  every time.
- `effects.py` is fully decoupled from MediaPipe/TensorFlow — it only needs
  pixel coordinates, so you could reuse it in a totally different gesture
  app.
- `app.py` is the only file that ties detection → classification → effects
  together; each other module can be tested independently (see
  `python predict.py` for a MediaPipe+model-only test).
