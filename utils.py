"""
utils.py
--------
Shared constants and helper functions used across collect_data.py, train.py,
predict.py and app.py.

Keeping this logic in ONE place is important: the way we turn raw MediaPipe
landmarks into a feature vector during data collection MUST exactly match
the way we do it during live prediction, or the trained model will see
inputs that don't match what it learned on.
"""

import numpy as np

# ---------------------------------------------------------------------------
# The 10 gestures GestureFX recognizes, in a fixed order. The INDEX of a
# gesture in this list is the label used everywhere (dataset, model output,
# etc). Keep this list identical across the whole project.
# ---------------------------------------------------------------------------
GESTURES = [
    "peace",        # -> Floating balloons
    "heart_hands",  # -> Falling hearts
    "thumbs_up",    # -> Confetti burst
    "open_palm",    # -> Sparkles around the hand
    "wave",         # -> Rainbow trail
    "ok",           # -> Star burst
    "love_sign",    # -> Pink glowing particles
    "fist",         # -> Comic-style explosion
    "point_up",     # -> Lightning from fingertip
    "rock_sign",    # -> Fire effect around the hand
]

# Human-friendly labels + emoji for on-screen display
GESTURE_DISPLAY = {
    "peace": "Peace",
    "heart_hands": "Heart Hands",
    "thumbs_up": "Thumbs Up",
    "open_palm": "Open Palm",
    "wave": "Wave",
    "ok": "OK",
    "love_sign": "Love Sign",
    "fist": "Fist",
    "point_up": "Point Up",
    "rock_sign": "Rock Sign",
}

NUM_LANDMARKS = 21          # MediaPipe Hands always returns 21 landmarks
NUM_COORDS = 3               # x, y, z per landmark
FEATURE_LENGTH = NUM_LANDMARKS * NUM_COORDS  # 63 -> input size of the model

WRIST_IDX = 0                # landmark index of the wrist
MIDDLE_MCP_IDX = 9            # landmark index of the middle-finger knuckle


def landmarks_to_feature_vector(hand_landmarks):
    """
    Convert a MediaPipe `hand_landmarks` object (21 landmarks with
    .x, .y, .z in [0, 1] normalized image coordinates) into a fixed-length,
    translation- and scale-invariant feature vector.

    Why normalize like this?
    - Translation invariance (subtract the wrist): the model shouldn't care
      WHERE in the frame your hand is, only its shape.
    - Scale invariance (divide by palm size): the model shouldn't care how
      close your hand is to the camera.

    Returns
    -------
    np.ndarray of shape (63,), dtype float32
    """
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )  # shape (21, 3)

    wrist = pts[WRIST_IDX].copy()
    pts -= wrist  # translation invariance

    # Scale invariance: normalize by distance from wrist to middle-finger MCP
    # (a stable reference distance that barely changes with hand pose).
    scale = np.linalg.norm(pts[MIDDLE_MCP_IDX])
    if scale < 1e-6:
        scale = 1e-6  # avoid divide-by-zero on degenerate detections
    pts /= scale

    return pts.flatten().astype(np.float32)  # shape (63,)


def landmarks_to_pixel_coords(hand_landmarks, frame_width, frame_height):
    """
    Convert normalized MediaPipe landmarks into pixel (x, y) integer
    coordinates for the given frame size. Used by effects.py to know WHERE
    on screen to draw animations (e.g. at the fingertip).

    Returns
    -------
    list of 21 (x, y) integer tuples
    """
    return [
        (int(lm.x * frame_width), int(lm.y * frame_height))
        for lm in hand_landmarks.landmark
    ]


# Handy landmark index names (MediaPipe Hands topology) used by effects.py
# to find specific points like the index fingertip.
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_MCP = 5
LM_INDEX_TIP = 8
LM_MIDDLE_TIP = 12
LM_RING_TIP = 16
LM_PINKY_TIP = 20
