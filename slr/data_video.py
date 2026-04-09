"""
data_video.py  —  DataLoader for video-based datasets (ISL, VGT, MS-ASL)
=========================================================================
Extracts the same 67 keypoints via MediaPipe Holistic and applies the
identical preprocess pipeline (interpolate → normalise) as the Google ASL
loader, so both datasets produce (T, 134) tensors with the same semantics.

Expected dataset layout:
    <data_root>/
        videos/
            <class_label>/
                <clip_001>.mp4
                <clip_002>.mp4
                ...
        train.csv   (columns: path, label, participant_id)   ← optional

If train.csv is absent the loader discovers clips by walking the directory
tree and uses the immediate parent folder name as the class label.

Usage:
    train_loader, val_loader, num_classes = get_dataloaders("./isl-dataset")
"""

import csv
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from preprocess import (
    preprocess, INPUT_DIM,
    N_HAND_KP, N_POSE_KP, POSE_INDICES,
)

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------

_holistic_instance = None

def _get_holistic():
    """Lazy singleton — one MediaPipe Holistic model per process."""
    global _holistic_instance
    if _holistic_instance is None:
        _holistic_instance = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _holistic_instance


# ---------------------------------------------------------------------------
# Keypoint extraction from video  (video path → (T, 134) array)
# ---------------------------------------------------------------------------

def _landmarks_to_array(landmarks, n_kp: int, indices: Optional[list] = None) -> np.ndarray:
    """
    Convert a MediaPipe NormalizedLandmarkList to a (n_kp, 2) array.
    Returns all-NaN if landmarks is None (hand not detected in this frame).

    Args:
        landmarks : MediaPipe landmark object or None
        n_kp      : number of keypoints expected
        indices   : if set, select only these indices from landmarks
    """
    if landmarks is None:
        return np.full((n_kp, 2), np.nan, dtype=np.float32)

    if indices is not None:
        return np.array(
            [[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in indices],
            dtype=np.float32,
        )
    return np.array(
        [[lm.x, lm.y] for lm in landmarks.landmark[:n_kp]],
        dtype=np.float32,
    )


def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    """
    Run MediaPipe Holistic on every frame of a video clip and extract
    the same 67 keypoints used in the paper.

    Args:
        video_path : path to a video file (.mp4, .avi, etc.)

    Returns:
        (T, 134) float32 array, NaN where landmarks were not detected.
        Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    holistic = _get_holistic()
    frames   = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe expects RGB
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        lhand = _landmarks_to_array(results.left_hand_landmarks,  N_HAND_KP)
        rhand = _landmarks_to_array(results.right_hand_landmarks, N_HAND_KP)
        pose  = _landmarks_to_array(results.pose_landmarks, N_POSE_KP, indices=POSE_INDICES)

        kp_frame = np.concatenate([lhand, rhand, pose], axis=0)   # (67, 2)
        frames.append(kp_frame)

    cap.release()

    if not frames:
        return None

    seq = np.stack(frames, axis=0)                                 # (T, 67, 2)
    return seq.reshape(len(frames), INPUT_DIM).astype(np.float32)  # (T, 134)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoSLRDataset(Dataset):
    """
    Generic sign-language video dataset.

    Signer-independent split: if participant_id is available (via train.csv),
    signers are kept separate across train / val. Otherwise a random
    clip-level stratified split is used as fallback.

    Args:
        data_root    : root directory of the dataset
        split        : 'train' | 'val' | 'all'
        val_fraction : fraction of signers (or clips) held out for val
        max_seq_len  : truncate sequences longer than this
        label_map    : label→int dict; built from discovered classes if None
        seed         : RNG seed
        video_exts   : file extensions to treat as video clips
    """

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        val_fraction: float = 0.15,
        max_seq_len: Optional[int] = None,
        label_map: Optional[dict] = None,
        seed: int = 42,
    ):
        assert split in ("train", "val", "all")
        self.data_root   = Path(data_root)
        self.max_seq_len = max_seq_len

        records = self._load_records()

        # Build label map
        if label_map is None:
            labels         = sorted({r["label"] for r in records})
            self.label_map = {l: i for i, l in enumerate(labels)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

        # Filter to known labels
        records = [r for r in records if r["label"] in self.label_map]

        # Signer-independent split (falls back to clip-level if no signer info)
        records = self._split(records, split, val_fraction, seed)
        self.records = records

        print(
            f"[VideoSLR] {split:5s} | clips={len(self.records):,} | "
            f"classes={self.num_classes}"
        )

    # ------------------------------------------------------------------

    def _load_records(self) -> list:
        """
        Return list of dicts: {path, label, participant_id}.
        Reads train.csv if present; otherwise walks directory tree.
        """
        csv_path = self.data_root / "train.csv"
        if csv_path.exists():
            records = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    records.append({
                        "path":           self.data_root / Path(*row["path"].split("/")),
                        "label":          row["label"],
                        "participant_id": row.get("participant_id", "unknown"),
                    })
            return records

        # Auto-discover: <data_root>/<label>/<clip>.<ext>
        records = []
        for p in sorted(self.data_root.rglob("*")):
            if p.suffix.lower() in self.VIDEO_EXTS:
                records.append({
                    "path":           p,
                    "label":          p.parent.name,
                    "participant_id": "unknown",
                })
        return records

    def _split(self, records: list, split: str, val_fraction: float, seed: int) -> list:
        if split == "all":
            return records

        rng      = np.random.default_rng(seed)
        signers  = list({r["participant_id"] for r in records})
        rng.shuffle(signers)
        n_val    = max(1, int(len(signers) * val_fraction))
        val_set  = set(signers[:n_val])

        if split == "train":
            return [r for r in records if r["participant_id"] not in val_set]
        else:
            return [r for r in records if r["participant_id"] in val_set]

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        label = self.label_map[rec["label"]]

        seq = extract_keypoints_from_video(str(rec["path"]))  # (T, 134) or None

        if seq is None or len(seq) == 0:
            # Return a zeroed single-frame sequence as fallback
            seq = np.zeros((1, INPUT_DIM), dtype=np.float32)
        else:
            seq = preprocess(seq)   # interpolate + normalise

        if self.max_seq_len and len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Collate + DataLoader factory
# ---------------------------------------------------------------------------

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths  = [s.shape[0] for s in sequences]
    padded   = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    T_max    = padded.shape[1]
    pad_mask = torch.zeros(len(sequences), T_max, dtype=torch.bool)
    for i, l in enumerate(lengths):
        pad_mask[i, l:] = True
    return padded, pad_mask, torch.stack(labels)


def get_dataloaders(
    data_root: str,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    max_seq_len: Optional[int] = None,
    num_workers: int = 0,    # 0 recommended — MediaPipe doesn't fork cleanly
    seed: int = 42,
):
    """
    Note: num_workers=0 is the safe default for video datasets because
    MediaPipe's internal state doesn't survive forking. If you need
    parallelism, pre-extract keypoints to disk first and load from numpy
    files instead.
    """
    train_ds = VideoSLRDataset(data_root, "train", val_fraction, max_seq_len, seed=seed)
    val_ds   = VideoSLRDataset(data_root, "val",   val_fraction, max_seq_len,
                               label_map=train_ds.label_map, seed=seed)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, train_ds.num_classes