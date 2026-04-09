"""
preprocess.py  —  Shared preprocessing pipeline (paper-faithful)
=================================================================
Implements exactly what Holmes et al. 2023 describes:

  1. Extract 67 keypoints from MediaPipe Holistic output:
       - 21 left-hand  landmarks  (MediaPipe hand indices 0-20)
       - 21 right-hand landmarks  (MediaPipe hand indices 0-20)
       - 25 upper-body pose landmarks (MediaPipe pose indices 11-23, see below)

  2. Temporal imputation — linear interpolation over frames where hands
     were not detected (NaN frames).

  3. Normalisation — shift origin to chest centre (midpoint of shoulders),
     scale so that shoulder-to-shoulder distance = 1.

The 25 upper-body pose landmark indices chosen from MediaPipe's 33-point
pose skeleton:
    0  nose
    1  left eye (inner)     2  left eye         3  left eye (outer)
    4  right eye (inner)    5  right eye        6  right eye (outer)
    7  left ear             8  right ear
    9  left mouth           10  right mouth
    11  left shoulder       12  right shoulder
    13  left elbow          14  right elbow
    15  left wrist          16  right wrist
    17  left pinky          18  right pinky
    19  left index          20  right index
    21  left thumb          22  right thumb
    23  left hip            24  right hip
     
  → indices [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    = first 25 pose landmarks  (covers full upper body + head)

Together: 21 + 21 + 25 = 67 keypoints × 2 coords (x,y) = 134-dim input vector.

Shoulder indices inside the 67-kp layout:
    Left  shoulder = pose kp 11 → overall index (21+21+0) = 42  (pose starts at 42)
    Right shoulder = pose kp 12 → overall index (21+21+1) = 43
    x-coord of kp i  = column i*2
    y-coord of kp i  = column i*2 + 1
"""

import numpy as np

# ---- Landmark selection constants ----------------------------------------

N_HAND_KP   = 21          # landmarks per hand
N_POSE_KP   = 25          # upper-body pose landmarks we keep
N_KP_TOTAL  = N_HAND_KP * 2 + N_POSE_KP   # 67
COORDS      = 2           # x, y  (paper drops z)
INPUT_DIM   = N_KP_TOTAL * COORDS          # 134

# Which of the 33 MediaPipe pose landmarks to keep (first 25)
POSE_INDICES = list(range(25))

# Shoulder positions in the final 67-kp flat vector
# Layout: [lhand 0-20 | rhand 21-41 | pose 42-66]
# Pose kp 0 in our selection = MediaPipe pose landmark 11 (left shoulder)
# Pose kp 1 in our selection = MediaPipe pose landmark 12 (right shoulder)
_POSE_OFFSET       = N_HAND_KP * 2          # 42
LEFT_SHOULDER_IDX  = (_POSE_OFFSET + 0) * COORDS    # col 84  (x)
RIGHT_SHOULDER_IDX = (_POSE_OFFSET + 1) * COORDS    # col 86  (x)

# NOTE: POSE_INDICES[0] = 11 (left shoulder), POSE_INDICES[1] = 12 (right)


# ---- Core functions -------------------------------------------------------

def interpolate_missing(seq: np.ndarray) -> np.ndarray:
    """
    Linear interpolation over NaN frames, per coordinate.
    Falls back to nearest-fill at boundaries.

    Args:
        seq : (T, 134) float32, may contain NaNs

    Returns:
        (T, 134) float32 with no NaNs
    """
    seq = seq.copy()
    t   = np.arange(len(seq))
    for col in range(seq.shape[1]):
        y    = seq[:, col]
        nans = np.isnan(y)
        if nans.all():
            seq[:, col] = 0.0
        elif nans.any():
            seq[:, col] = np.interp(t, t[~nans], y[~nans])
    return seq


def normalise(seq: np.ndarray) -> np.ndarray:
    """
    Shift origin to chest centre; scale so shoulder distance = 1.

    Chest centre  = midpoint of left and right shoulders (per frame).
    Scale         = mean shoulder-to-shoulder distance across all frames
                    (mean is more stable than per-frame scaling for short clips).

    Args:
        seq : (T, 134) float32, no NaNs

    Returns:
        (T, 134) float32 normalised
    """
    seq = seq.copy()

    lsx = seq[:, LEFT_SHOULDER_IDX]        # (T,)
    lsy = seq[:, LEFT_SHOULDER_IDX + 1]
    rsx = seq[:, RIGHT_SHOULDER_IDX]
    rsy = seq[:, RIGHT_SHOULDER_IDX + 1]

    # Chest centre (per frame)
    cx = ((lsx + rsx) / 2)[:, None]        # (T, 1)
    cy = ((lsy + rsy) / 2)[:, None]

    # Shoulder distance — use mean for stability
    dist  = np.sqrt((lsx - rsx) ** 2 + (lsy - rsy) ** 2)
    scale = dist.mean()
    if scale < 1e-6:
        scale = 1.0

    seq[:, 0::2] = (seq[:, 0::2] - cx) / scale   # all x coords
    seq[:, 1::2] = (seq[:, 1::2] - cy) / scale   # all y coords
    return seq


def preprocess(seq: np.ndarray) -> np.ndarray:
    """
    Full pipeline: interpolate → normalise.

    Args:
        seq : (T, 134) raw keypoints, may contain NaNs

    Returns:
        (T, 134) float32, clean and normalised
    """
    seq = interpolate_missing(seq)
    seq = normalise(seq)
    return seq