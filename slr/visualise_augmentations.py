"""
visualise_pose.py  —  Animate a full .npy pose sequence as a skeleton video
============================================================================
Shows the original sequence and N augmented versions side by side,
animating every frame so you can see the full temporal motion.

Usage:
    python visualise_pose.py --npy_path ./asl-signs/train_landmarks_npy/85282.npy
    python visualise_pose.py --npy_path ./asl-signs/train_landmarks_npy/85282.npy --variations 3 --interval 80
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from data_augmentation import SignAugmentor


# ---------------------------------------------------------------------------
# Skeleton connectivity
# ---------------------------------------------------------------------------

# Hand finger chains — indices within the 21-kp hand block
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
    [5, 9, 13, 17],        # knuckle bar
]

# Pose upper-body connections — absolute indices into the 67-kp array
# Pose block starts at 42; slot = MediaPipe pose landmark index
# slot 0=nose, 7=left_ear, 8=right_ear, 11=L_shoulder, 12=R_shoulder,
# 13=L_elbow, 14=R_elbow, 15=L_wrist, 16=R_wrist, 23=L_hip, 24=R_hip
POSE_CONNECTIONS = [
    (42+0,  42+7),   # nose — left ear
    (42+0,  42+8),   # nose — right ear
    (42+11, 42+12),  # L_shoulder — R_shoulder
    (42+11, 42+13),  # L_shoulder — L_elbow
    (42+13, 42+15),  # L_elbow    — L_wrist
    (42+12, 42+14),  # R_shoulder — R_elbow
    (42+14, 42+16),  # R_elbow    — R_wrist
    (42+11, 42+23),  # L_shoulder — L_hip
    (42+12, 42+24),  # R_shoulder — R_hip
    (42+23, 42+24),  # L_hip      — R_hip
]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_hand(ax, hand, color):
    """hand: (21, 2)"""
    ax.scatter(hand[:, 0], hand[:, 1], c=color, s=12, zorder=3)
    for chain in FINGER_CHAINS:
        for a, b in zip(chain, chain[1:]):
            ax.plot([hand[a, 0], hand[b, 0]],
                    [hand[a, 1], hand[b, 1]],
                    c=color, alpha=0.5, linewidth=0.9)


def _draw_pose(ax, points):
    """points: (67, 2) — full keypoint array"""
    pose = points[42:67]
    ax.scatter(pose[:, 0], pose[:, 1], c='green', s=12, zorder=3)
    for a, b in POSE_CONNECTIONS:
        ax.plot([points[a, 0], points[b, 0]],
                [points[a, 1], points[b, 1]],
                c='black', alpha=0.55, linewidth=1.1)


def compute_axis_limits(sequences: list, margin: float = 0.15):
    """
    Compute fixed axis limits from all frames and sequences.
    Both x and y use the same coordinate range so the skeleton
    is scaled uniformly — no stretching in either dimension.
    """
    all_points = np.concatenate([s.reshape(-1, 67, 2) for s in sequences], axis=0)
    flat  = all_points.reshape(-1, 2)
    valid = flat[~np.all(np.abs(flat) < 1e-6, axis=1)]
    if len(valid) < 2:
        return (-1.5, 1.5), (1.5, -1.5)

    x_min = valid[:, 0].min() - margin
    x_max = valid[:, 0].max() + margin
    y_min = valid[:, 1].min() - margin
    y_max = valid[:, 1].max() + margin

    # Use the same span for both axes so 1 unit = 1 unit
    span  = max(x_max - x_min, y_max - y_min)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    xlim = (x_mid - span / 2, x_mid + span / 2)
    ylim = (y_mid + span / 2, y_mid - span / 2)  # inverted Y
    return xlim, ylim


def draw_frame(ax, points, xlim, ylim, title=None):
    """
    Draw one frame onto ax using fixed square coordinate limits.
    points : (67, 2)
    xlim, ylim : pre-computed from compute_axis_limits()
    """
    ax.clear()
    _draw_hand(ax, points[0:21],  color='red')
    _draw_hand(ax, points[21:42], color='blue')
    _draw_pose(ax, points)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)   # inverted Y

    if title:
        ax.set_title(title, fontsize=9)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Main animation function
# ---------------------------------------------------------------------------

def visualise_sequence(npy_path: str, num_variations: int = 3, interval: int = 60):
    """
    Animate the full pose sequence and N augmented versions side by side.

    Args:
        npy_path      : path to a .npy file of shape (T, 134)
        num_variations: number of augmented copies to show alongside original
        interval      : milliseconds between frames
    """
    data = np.load(npy_path)   # (T, 134)
    print(f"Loaded '{Path(npy_path).name}': {data.shape[0]} frames, {data.shape[1]} features")

    augmentor = SignAugmentor(flip_prob=0.5, rotate_std=0.5, keep_ratio=0.7)

    # Generate augmented sequences up front
    sequences = [data]
    labels    = ["Original"]
    for i in range(num_variations):
        aug = augmentor(data)   # (T', 134)
        sequences.append(aug)
        labels.append(f"Aug {i+1}")

    n_cols  = len(sequences)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(Path(npy_path).stem, fontsize=11)

    # Total animation frames = longest sequence
    max_frames = max(len(s) for s in sequences)

    # Add a frame counter text
    frame_text = fig.text(0.5, 0.01, '', ha='center', fontsize=9, color='grey')

    xlim, ylim = compute_axis_limits(sequences)

    def update(frame_idx):
        for ax, seq, label in zip(axes, sequences, labels):
            # Loop shorter sequences so all panels stay in sync
            t = min(frame_idx, len(seq) - 1)
            points = seq[t].reshape(67, 2)
            draw_frame(ax, points, xlim, ylim,
                       title=f"{label}\n(frame {t+1}/{len(seq)})")
        frame_text.set_text(f"t = {frame_idx + 1} / {max_frames}")
        return axes

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=interval,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()
    return ani   # keep reference alive so garbage collector doesn't kill it


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate a .npy pose sequence")
    parser.add_argument("--npy_path",   required=True,       help="Path to .npy file")
    parser.add_argument("--variations", type=int, default=3, help="Number of augmented views")
    parser.add_argument("--interval",   type=int, default=60, help="ms between frames")
    args = parser.parse_args()

    path = Path(args.npy_path)
    if not path.exists():
        print(f"File not found: {path}")
    else:
        ani = visualise_sequence(str(path), args.variations, args.interval)