# AI-GestureFX — Real-Time Hand Gesture Recognition & Interactive Vision Effects

AI-GestureFX is a real-time computer vision application that recognizes **10 hand gestures** from webcam input using a **custom-trained TensorFlow/Keras neural network** and maps confirmed gestures to interactive visual effects.

The system combines **MediaPipe Hands for hand landmark detection, TensorFlow/Keras for gesture classification, and OpenCV for real-time video processing and effect rendering**.

Unlike applications that rely on a pre-built gesture recognizer, AI-GestureFX trains its own classifier using collected hand-landmark data.

---

## Overview

The application follows an end-to-end machine learning pipeline:

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
63-D Normalized Feature Vector
   ↓
Custom TensorFlow/Keras Classifier
   ↓
Gesture + Confidence
   ↓
Confidence Filtering
   ↓
Temporal Stability Validation
   ↓
Gesture → Effect Mapping
   ↓
OpenCV Real-Time Rendering
```

This separation allows the data collection, model training, prediction, and visualization components to operate independently.

---

## Key Features

* Custom-trained TensorFlow/Keras gesture classification model
* Real-time webcam inference
* 10 gesture classes
* 21-point MediaPipe hand landmark extraction
* 63-dimensional normalized feature representation
* Translation and scale normalization
* Stratified train/validation split
* Dropout regularization
* Early stopping during training
* Confidence-based prediction filtering
* Temporal gesture stability validation
* Per-gesture effect cooldown
* Real-time FPS monitoring
* Modular OpenCV effect engine
* Standalone prediction testing
* Webcam-based dataset collection

---

## Supported Gestures

| Gesture     | Interactive Effect     |
| ----------- | ---------------------- |
| Peace       | Floating balloons      |
| Heart Hands | Falling hearts         |
| Thumbs Up   | Confetti burst         |
| Open Palm   | Sparkles               |
| Wave        | Rainbow trail          |
| OK          | Star burst             |
| Love Sign   | Pink glowing particles |
| Fist        | Comic-style explosion  |
| Point Up    | Lightning              |
| Rock Sign   | Fire effect            |

The gesture list is maintained centrally in `utils.py`, allowing the classifier, data collection pipeline, and inference pipeline to use the same class definitions.

---

## Technology Stack

| Technology      | Role                                     |
| --------------- | ---------------------------------------- |
| Python          | Application and ML development           |
| TensorFlow      | Machine learning framework               |
| Keras           | Neural network construction and training |
| MediaPipe Hands | Hand landmark detection                  |
| OpenCV          | Webcam processing and visualization      |
| NumPy           | Numerical feature processing             |
| Pandas          | Dataset loading and management           |
| Scikit-learn    | Label encoding and dataset splitting     |
| Matplotlib      | Training visualization                   |

---

# How the System Works

## 1. Hand Detection

For every webcam frame, MediaPipe Hands detects the visible hand and extracts **21 landmarks**.

Each landmark contains:

```text
x
y
z
```

This produces:

```text
21 × 3 = 63 features
```

MediaPipe is responsible for **landmark detection**, not gesture classification.

---

## 2. Feature Normalization

Raw landmark coordinates depend on where the hand appears in the camera frame and how close it is to the camera.

AI-GestureFX normalizes the landmarks before training and inference.

### Translation normalization

The wrist landmark is used as the reference point:

```text
landmark = landmark - wrist
```

This makes the representation less dependent on the hand's position in the frame.

### Scale normalization

The coordinates are then normalized using the distance between the wrist and middle-finger MCP landmark.

This reduces the effect of hand size and camera distance.

The final representation is a fixed:

```text
63-dimensional float32 vector
```

The same normalization function is used during both dataset collection and real-time prediction.

---

# Machine Learning Model

The classifier uses a fully connected TensorFlow/Keras neural network.

```text
63 Input Features
       │
       ▼
Dense(128, ReLU)
       │
       ▼
Dropout(0.3)
       │
       ▼
Dense(64, ReLU)
       │
       ▼
Dropout(0.3)
       │
       ▼
Dense(32, ReLU)
       │
       ▼
Softmax Output
```

A fully connected architecture is appropriate for this implementation because the model receives an engineered landmark feature vector rather than raw image pixels.

The output layer produces a probability distribution across the gesture classes.

---

# Dataset Collection

AI-GestureFX includes its own webcam-based dataset collection pipeline.

Run:

```bash
python collect_data.py
```

The application detects hand landmarks and converts them into normalized feature vectors.

Samples are stored in:

```text
dataset/gestures.csv
```

Each dataset row contains:

```text
f0 ... f62 + label
```

The collector supports:

* Gesture selection
* Continuous sample capture
* Individual sample capture
* Dataset progress tracking
* Saving collected samples
* Resume-friendly data collection

For better generalization, samples can be collected with variations in:

* Hand position
* Hand orientation
* Distance from camera
* Lighting
* Background
* Left/right hand usage

---

# Model Training

After collecting the dataset:

```bash
python train.py
```

The training pipeline:

1. Loads the landmark dataset.
2. Separates features and labels.
3. Encodes gesture labels.
4. Performs an 80/20 stratified train/validation split.
5. Builds the Keras neural network.
6. Trains for up to 150 epochs.
7. Uses early stopping based on validation loss.
8. Evaluates validation performance.
9. Saves the trained model.
10. Saves the class-label mapping.
11. Generates training curves.

Generated artifacts:

```text
models/
├── gesture_model.h5
├── labels.json
└── training_history.png
```

The actual validation accuracy is printed by the training script, rather than being hard-coded in the README.

---

# Real-Time Inference

To test the trained classifier independently:

```bash
python predict.py
```

The prediction pipeline:

```text
Camera Frame
     ↓
MediaPipe Hands
     ↓
Landmark Normalization
     ↓
Keras Model
     ↓
Gesture Label
     ↓
Confidence Score
```

The prediction window displays the detected gesture and model confidence.

This makes it possible to validate the ML component separately from the visual-effects application.

---

# Real-Time Application

Run:

```bash
python app.py
```

The application performs the complete pipeline:

```text
Camera
  ↓
Hand Detection
  ↓
Landmark Extraction
  ↓
Feature Normalization
  ↓
Gesture Classification
  ↓
Confidence Check
  ↓
Stability Check
  ↓
Effect Trigger
  ↓
Frame Rendering
```

The interface displays:

* Current gesture
* Prediction confidence
* FPS
* Hand landmarks
* Active visual effects

Press:

```text
q
```

to exit.

---

# Prediction Reliability

Real-time predictions can fluctuate between frames because of camera noise, hand movement, lighting changes, or similar gesture geometries.

AI-GestureFX uses three mechanisms to reduce unwanted triggers.

### Confidence threshold

```python
CONFIDENCE_THRESHOLD = 0.75
```

Predictions below this confidence level are ignored by the gesture stabilizer.

### Stability window

```python
STABILITY_WINDOW_SECONDS = 0.5
```

A gesture must remain consistent for approximately half a second before it is confirmed.

### Effect cooldown

```python
EFFECT_COOLDOWN_SECONDS = 2.0
```

After triggering an effect, the same gesture cannot immediately retrigger it.

Together:

```text
Low-confidence prediction
        ↓
       Ignore

Consistent high-confidence prediction
        ↓
   Confirm gesture
        ↓
   Trigger effect
        ↓
   Cooldown period
```

This prevents isolated misclassifications from repeatedly triggering animations.

---

# Project Architecture

```text
AI-GestureFX/
│
├── app.py
│   └── Main real-time application
│
├── collect_data.py
│   └── Webcam dataset collection
│
├── train.py
│   └── Model training pipeline
│
├── predict.py
│   └── Standalone inference
│
├── effects.py
│   └── Visual effects engine
│
├── utils.py
│   └── Shared gesture configuration
│       and landmark processing
│
├── dataset/
│   └── gestures.csv
│
├── models/
│   ├── gesture_model.h5
│   ├── labels.json
│   └── training_history.png
│
├── assets/
│
├── requirements.txt
└── README.md
```

---

# Module Responsibilities

### `utils.py`

Centralizes:

* Gesture definitions
* Display labels
* Landmark constants
* Feature dimensions
* Landmark normalization
* Pixel coordinate conversion

Keeping normalization in one shared function ensures that the feature representation used during training matches the representation used during inference.

### `collect_data.py`

Provides the webcam-based data collection pipeline.

### `train.py`

Handles:

* Dataset loading
* Label encoding
* Train/validation splitting
* Neural network construction
* Model training
* Early stopping
* Validation evaluation
* Model serialization
* Training visualization

### `predict.py`

Provides standalone real-time model inference.

### `effects.py`

Contains the visual effects engine and `EffectsManager`.

The effects layer is independent of the TensorFlow classifier and receives gesture information and landmark coordinates.

### `app.py`

Acts as the orchestration layer connecting:

```text
Detection
    ↓
Feature Processing
    ↓
Classification
    ↓
Prediction Validation
    ↓
Effect Triggering
    ↓
Rendering
```

---

# Installation

## Requirements

Python 3.9–3.11 is recommended for compatibility with the installed computer-vision dependencies.

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

### Step 1 — Collect training data

```bash
python collect_data.py
```

### Step 2 — Train the model

```bash
python train.py
```

### Step 3 — Test predictions

```bash
python predict.py
```

### Step 4 — Run the complete application

```bash
python app.py
```

---

# Configuration

Important runtime parameters are defined in `app.py`:

```python
CONFIDENCE_THRESHOLD = 0.75
STABILITY_WINDOW_SECONDS = 0.5
EFFECT_COOLDOWN_SECONDS = 2.0
HISTORY_MAXLEN = 30
```

These parameters allow the real-time inference behavior to be adjusted without changing the core model.

---

# Engineering Highlights

### Consistent Feature Engineering

The same landmark normalization function is used during data collection and inference, preventing inconsistencies between the training and production feature pipelines.

### Modular ML Pipeline

Dataset collection, model training, inference, and visualization are separated into independent modules.

### Real-Time Processing

The application continuously processes webcam frames and maintains an FPS estimate while simultaneously performing landmark detection, model inference, and effect rendering.

### Temporal Prediction Filtering

Instead of reacting to individual predictions, the application requires consistent high-confidence predictions across a time window before triggering an effect.

### Decoupled Effect Engine

The visual-effects implementation is separated from the machine-learning classifier, allowing the rendering layer to be reused independently.

---

# Future Improvements

Potential extensions include:

* Increasing the number of gesture classes
* Multi-hand gesture recognition
* Automated data augmentation
* Expanded model architectures
* Confusion-matrix based error analysis
* More comprehensive test datasets
* Model quantization
* TensorFlow Lite deployment
* GPU-optimized inference
* Gesture sequence recognition
* User-configurable gesture mappings
* Model experiment tracking
* Containerized deployment

---

# Learning Outcomes

This project demonstrates practical experience with:

* Computer vision
* Hand landmark detection
* Feature engineering
* Neural network classification
* Supervised learning
* Dataset creation
* Model training and validation
* Real-time inference
* Prediction confidence handling
* Temporal filtering
* Modular Python application design
* Real-time video processing

---

## License

This project is intended for educational and experimental purposes.
