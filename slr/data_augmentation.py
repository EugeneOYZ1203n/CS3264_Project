import numpy as np
from scipy.ndimage import gaussian_filter1d

def temporal_mask(x, ratio_range=(0.7, 1.3)):
    """
    Randomly scales the sequence length.
    - Ratio < 1.0: Randomly drops frames (downsampling).
    - Ratio > 1.0: Randomly duplicates frames (upsampling).
    Ensures the structural integrity by keeping indices sorted.
    """
    T = x.shape[0]
    # Pick a random ratio from the provided range
    ratio = np.random.uniform(ratio_range[0], ratio_range[1])
    n_target = int(T * ratio)
    
    # Ensure we don't end up with 0 frames
    n_target = max(1, n_target)

    if n_target < T:
        # Downsampling: Pick unique indices to keep
        indices = np.sort(np.random.choice(np.arange(T), n_target, replace=False))
    elif n_target > T:
        # Upsampling: Pick indices with replacement to create duplicates
        # We start with all indices to ensure every original frame exists at least once,
        # then add extra random ones.
        extra_indices = np.random.choice(np.arange(T), n_target - T, replace=True)
        indices = np.sort(np.concatenate([np.arange(T), extra_indices]))
    else:
        return x
        
    return x[indices]

def horizontal_flip(x, prob=0.5):
    """
    Flips the X-axis and swaps left/right hand landmark blocks.
    Layout: [0:21 LH | 21:42 RH | 42:67 Pose]
    """
    if np.random.rand() > prob:
        return x
        
    x = x.copy()
    # Flip X coordinates (assuming 0-1 normalization)
    x[:, :, 0] = 1.0 - x[:, :, 0]
    
    # Swap hand data
    lh_part = x[:, 0:21, :].copy()
    rh_part = x[:, 21:42, :].copy()
    x[:, 0:21, :] = rh_part
    x[:, 21:42, :] = lh_part

    # 3. Swap Pose Keypoints (Symmetry)
    # Based on MediaPipe Holistic indices + your 42-point offset:
    # Left Shoulder: 53 | Right Shoulder: 54
    # Left Elbow: 55    | Right Elbow: 56
    # Left Wrist: 57    | Right Wrist: 58
    # Left Pinky: 59    | Right Pinky: 60
    # Left Index: 61    | Right Index: 62
    # Left Thumb: 63    | Right Thumb: 64
    # Left Hip: 65      | Right Hip: 66
    
    pairs = [
        (53, 54), (55, 56), (57, 58), # Shoulders, Elbows, Wrists
        (59, 60), (61, 62), (63, 64), # Fingers (Pose version)
        (65, 66)                      # Hips
    ]
    
    for left, right in pairs:
        temp = x[:, left, :].copy()
        x[:, left, :] = x[:, right, :]
        x[:, right, :] = temp
    
    return x

def rotate_points(points, center, angle):
    """Helper to rotate points around a pivot."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    shifted = points - center
    rx = shifted[..., 0] * cos_a - shifted[..., 1] * sin_a
    ry = shifted[..., 0] * sin_a + shifted[..., 1] * cos_a
    return np.stack([rx, ry], axis=-1) + center

def rotate_hierarchical(x, rotate_std=0.05, smoothness=5.0):
    x = x.copy()
    T = x.shape[0]

    def get_smooth_noise():
        raw = np.random.normal(0, rotate_std, size=T)
        return gaussian_filter1d(raw, sigma=smoothness)

    # 4 independent smooth paths
    angles = {k: get_smooth_noise() for k in ['l_sh', 'r_sh', 'l_el', 'r_el']}

    for t in range(T):
        # --- LEFT SIDE ---
        # 1. Shoulder Pivot (53)
        if not np.isnan(x[t, 53]).any():
            # Move EVERYTHING from elbow down
            l_arm_chain = [55, 57] + list(range(0, 21))
            x[t, l_arm_chain] = rotate_points(x[t, l_arm_chain], x[t, 53], angles['l_sh'][t])
        
        # 2. Elbow Pivot (55) - IMPORTANT: This index 55 was JUST updated by the shoulder move
        if not np.isnan(x[t, 55]).any():
            # Move only wrist and hand
            l_forearm_chain = [57] + list(range(0, 21))
            x[t, l_forearm_chain] = rotate_points(x[t, l_forearm_chain], x[t, 55], angles['l_el'][t])

        # --- RIGHT SIDE ---
        # 1. Shoulder Pivot (54)
        if not np.isnan(x[t, 54]).any():
            r_arm_chain = [56, 58] + list(range(21, 42))
            x[t, r_arm_chain] = rotate_points(x[t, r_arm_chain], x[t, 54], angles['r_sh'][t])
            
        # 2. Elbow Pivot (56)
        if not np.isnan(x[t, 56]).any():
            r_forearm_chain = [58] + list(range(21, 42))
            x[t, r_forearm_chain] = rotate_points(x[t, r_forearm_chain], x[t, 56], angles['r_el'][t])
            
    return x

class SignAugmentor:
    def __init__(self, flip_prob=0.5, rotate_std=0.05, temporal_ratio=(0.7,1.3), ik_std=0.03):
        self.flip_prob = flip_prob
        self.rotate_std = rotate_std
        self.temporal_ratio = temporal_ratio
        self.ik_std = ik_std

    def __call__(self, x):
        # x: (T, 134) -> (T, 67, 2)
        T, D = x.shape
        x = x.reshape(T, 67, 2)

        # 1. Temporal Masking (Stitching frames)
        x = temporal_mask(x, self.temporal_ratio)
        
        # 2. Horizontal Flip
        x = horizontal_flip(x, self.flip_prob)
        
        # 3. Hierarchical Rotation
        x = rotate_hierarchical(x, self.rotate_std)

        # Clean up NaNs and reshape back
        x = np.nan_to_num(x)
        return x.reshape(-1, D).astype(np.float32)