"""
train.py
--------
Trains a TensorFlow/Keras model to classify hand gestures from the
normalized 63-number landmark feature vectors collected by collect_data.py.

Pipeline
--------
1. Load dataset/gestures.csv into a pandas DataFrame.
2. Split features (X) and labels (y). Encode string labels -> integers
   -> one-hot vectors with sklearn's LabelEncoder + Keras' to_categorical.
3. Train/validation split (80/20, stratified so each gesture is represented
   proportionally in both sets).
4. Build a small fully-connected (Dense) Sequential network. We don't need
   anything fancier than Dense layers because the input is already a clean,
   normalized geometric feature vector -- not raw pixels.
5. Train with EarlyStopping so we don't overfit a small dataset.
6. Plot accuracy & loss curves to models/training_history.png.
7. Save the trained model to models/gesture_model.h5 and the label order
   to models/labels.json (predict.py needs this to map model output index
   -> gesture name).

Run:  python train.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # so this also works on machines with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks, utils as keras_utils

from utils import FEATURE_LENGTH, GESTURES

DATASET_PATH = os.path.join("dataset", "gestures.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"No dataset found at {DATASET_PATH}. Run collect_data.py first."
        )
    df = pd.read_csv(DATASET_PATH)
    if len(df) == 0:
        raise ValueError("Dataset is empty. Collect some samples first.")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    print(f"Loaded {len(df)} samples across {df['label'].nunique()} gestures.")
    print(df["label"].value_counts())

    return X, y_raw


def build_model(num_classes):
    """
    A simple, beginner-friendly fully-connected classifier.
    Input: 63-dim normalized landmark vector.
    Output: probability distribution over the 10 gestures (softmax).
    """
    model = models.Sequential([
        layers.Input(shape=(FEATURE_LENGTH,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=120)
    print(f"Saved training curves to {HISTORY_PLOT_PATH}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y_raw = load_dataset()

    # Warn (but don't crash) if some gestures from GESTURES are missing
    missing = set(GESTURES) - set(np.unique(y_raw))
    if missing:
        print(f"WARNING: no samples collected yet for: {sorted(missing)}. "
              f"The model will not be able to recognize them.")

    encoder = LabelEncoder()
    encoder.fit(sorted(np.unique(y_raw)))  # deterministic class order
    y_int = encoder.transform(y_raw)
    y_onehot = keras_utils.to_categorical(y_int, num_classes=len(encoder.classes_))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42,
        stratify=y_int,
    )
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")

    model = build_model(num_classes=len(encoder.classes_))
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc*100:.2f}%  (loss={val_loss:.4f})")

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    with open(LABELS_PATH, "w") as f:
        json.dump(list(encoder.classes_), f, indent=2)
    print(f"Saved label mapping to {LABELS_PATH}")

    plot_history(history)


if __name__ == "__main__":
    main()
