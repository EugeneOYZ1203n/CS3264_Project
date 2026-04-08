"""
Google ASL Data Pipeline
========================
Loads and preprocesses the Google ASL Kaggle dataset for the SLR model.

Expected dataset layout (after Kaggle download):
    <data_root>/
        train_landmark_files/       # pre-extracted MediaPipe keypoints
            <participant_id>/
                <sample_id>.parquet
        train.csv                   # columns: path, participant_id, sequence_id, sign
        sign_to_prediction_index_map.json

Usage:
    dataset = GoogleASLDataset(data_root="path/to/asl-signs")
    train_loader, val_loader = dataset.get_dataloaders(batch_size=64)
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------

# MediaPipe Holistic landmark indices used by the paper (67 keypoints total):
#   Pose:       landmarks 11-16 (shoulders, elbows, wrists) + 23-24 (hips) = 8
#               and face/nose landmarks 0, 2, 5, 11-16 depending on version.
#   The Google ASL dataset provides face (468), pose (33), and hand (21*2)
#   landmarks. We select the same 67 used in the paper.
#
# For simplicity and reproducibility we select:
#   - Left  hand : 21 keypoints  (indices 0-20  in left_hand  array)
#   - Right hand : 21 keypoints  (indices 0-20  in right_hand array)
#   - Pose       : 25 upper-body keypoints (indices 0-24 in pose array,
#                  which covers nose, eyes, ears, shoulders, elbows,
#                  wrists, hips, knees — we take first 25)
# Total: 21 + 21 + 25 = 67  ✓

N_HAND_KP  = 21
N_POSE_KP  = 25
N_KP_TOTAL = N_HAND_KP * 2 + N_POSE_KP   # 67
COORDS     = 2                             # x, y only (paper drops z)
INPUT_DIM  = N_KP_TOTAL * COORDS          # 134


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def extract_keypoints(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the 67 x 2 keypoints from a single sample's parquet DataFrame.

    The Google ASL parquet files use LONG format:
        columns: frame | row_id | type | landmark_index | x | y | z
        rows:    one row per landmark per frame

    where `type` is one of: 'face', 'left_hand', 'right_hand', 'pose'

    We select:
        left_hand  indices 0-20  (21 kp)
        right_hand indices 0-20  (21 kp)
        pose       indices 0-24  (25 kp)
    and keep only x, y (drop z).

    Returns:
        Array of shape (T, 134) — float32, NaN where landmarks were missing.
    """
    def _pivot(landmark_type: str, n_kp: int) -> np.ndarray:
        """Returns (T, n_kp, 2) with NaN where the landmark was absent."""
        sub = df[(df["type"] == landmark_type) & (df["landmark_index"] < n_kp)]
        frames = sorted(df["frame"].unique())
        if sub.empty:
            return np.full((len(frames), n_kp, 2), np.nan, dtype=np.float32)

        px = sub.pivot(index="frame", columns="landmark_index", values="x")
        py = sub.pivot(index="frame", columns="landmark_index", values="y")

        kp_idx = list(range(n_kp))
        px = px.reindex(index=frames, columns=kp_idx)
        py = py.reindex(index=frames, columns=kp_idx)

        return np.stack([px.values, py.values], axis=-1).astype(np.float32)

    lhand = _pivot("left_hand",  N_HAND_KP)   # (T, 21, 2)
    rhand = _pivot("right_hand", N_HAND_KP)   # (T, 21, 2)
    pose  = _pivot("pose",       N_POSE_KP)   # (T, 25, 2)

    T = lhand.shape[0]
    kps = np.concatenate([lhand, rhand, pose], axis=1)  # (T, 67, 2)
    return kps.reshape(T, INPUT_DIM).astype(np.float32)  # (T, 134)


def interpolate_missing(seq: np.ndarray) -> np.ndarray:
    """
    Linear interpolation over frames where hand keypoints are NaN.
    Falls back to forward-fill / back-fill at the boundaries.

    Args:
        seq: (T, 134) float32, may contain NaNs.

    Returns:
        (T, 134) float32 with NaNs replaced.
    """
    seq = seq.copy()
    for col in range(seq.shape[1]):
        y = seq[:, col]
        nans = np.isnan(y)
        if nans.all():
            seq[:, col] = 0.0
            continue
        if nans.any():
            x = np.arange(len(y))
            seq[:, col] = np.interp(x, x[~nans], y[~nans])
    return seq


def normalise(seq: np.ndarray) -> np.ndarray:
    """
    Normalise keypoints so that:
      - Origin is shifted to the chest centre (midpoint of the two shoulders).
      - Scale is set so the shoulder-to-shoulder distance = 1.

    Shoulder keypoints in our 67-kp layout:
      Left  shoulder = pose kp index 11 → overall index (21+21+11) = 53
      Right shoulder = pose kp index 12 → overall index 54

    x coords for kp i are at column  i*2
    y coords for kp i are at column  i*2 + 1
    """
    LEFT_SHOULDER_IDX  = (N_HAND_KP * 2 + 11) * COORDS   # col 106
    RIGHT_SHOULDER_IDX = (N_HAND_KP * 2 + 12) * COORDS   # col 108

    lsx = seq[:, LEFT_SHOULDER_IDX]
    lsy = seq[:, LEFT_SHOULDER_IDX + 1]
    rsx = seq[:, RIGHT_SHOULDER_IDX]
    rsy = seq[:, RIGHT_SHOULDER_IDX + 1]

    # Chest centre
    cx = ((lsx + rsx) / 2)[:, None]   # (T, 1)
    cy = ((lsy + rsy) / 2)[:, None]

    # Shoulder distance (per frame, then mean for stability)
    dist = np.sqrt((lsx - rsx) ** 2 + (lsy - rsy) ** 2)
    scale = dist.mean()
    if scale < 1e-6:
        scale = 1.0

    # Shift x and y channels separately
    seq = seq.copy()
    seq[:, 0::2] = (seq[:, 0::2] - cx) / scale   # all x coords
    seq[:, 1::2] = (seq[:, 1::2] - cy) / scale   # all y coords
    return seq


def preprocess_sequence(df: pd.DataFrame) -> np.ndarray:
    """Full preprocessing pipeline for one sample: extract → interpolate → normalise."""
    kps = extract_keypoints(df)
    kps = interpolate_missing(kps)
    kps = normalise(kps)
    return kps   # (T, 134)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GoogleASLDataset(Dataset):
    """
    PyTorch Dataset for the Google ASL Kaggle competition data.

    The dataset is signer-independent: signers are split across
    train / val subsets so no signer appears in both.

    Args:
        data_root   : path to the unzipped Kaggle dataset directory.
        split       : 'train' | 'val' | 'all'
        val_fraction: fraction of signers held out for validation.
        max_seq_len : truncate sequences longer than this (None = no limit).
        label_map   : optional dict mapping sign string → int index.
                      If None, built automatically from train.csv.
        seed        : random seed for the signer split.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        val_fraction: float = 0.15,
        max_seq_len: Optional[int] = None,
        label_map: Optional[dict] = None,
        seed: int = 42,
    ):
        assert split in ("train", "val", "all"), "split must be 'train', 'val', or 'all'"
        self.data_root   = Path(data_root)
        self.max_seq_len = max_seq_len

        # ---- Load metadata ------------------------------------------------
        csv_path = self.data_root / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"train.csv not found at {csv_path}.\n"
                "Download the dataset from: "
                "https://www.kaggle.com/competitions/asl-signs/data"
            )
        meta = pd.read_csv(csv_path)

        # ---- Label map ----------------------------------------------------
        if label_map is None:
            sign_map_path = self.data_root / "sign_to_prediction_index_map.json"
            if sign_map_path.exists():
                with open(sign_map_path) as f:
                    self.label_map = json.load(f)
            else:
                # Build from the CSV if the JSON isn't present
                signs = sorted(meta["sign"].unique())
                self.label_map = {s: i for i, s in enumerate(signs)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

        # ---- Signer-independent split -------------------------------------
        rng = np.random.default_rng(seed)
        all_signers = meta["participant_id"].unique()
        rng.shuffle(all_signers)
        n_val = max(1, int(len(all_signers) * val_fraction))
        val_signers   = set(all_signers[:n_val])
        train_signers = set(all_signers[n_val:])

        if split == "train":
            mask = meta["participant_id"].isin(train_signers)
        elif split == "val":
            mask = meta["participant_id"].isin(val_signers)
        else:
            mask = pd.Series([True] * len(meta))

        self.meta = meta[mask].reset_index(drop=True)

        print(
            f"[GoogleASLDataset] split={split} | "
            f"samples={len(self.meta):,} | "
            f"signers={self.meta['participant_id'].nunique()} | "
            f"classes={self.num_classes}"
        )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row   = self.meta.iloc[idx]
        label = self.label_map[row["sign"]]

        # Load parquet — path column is relative to data_root.
        # train.csv always uses forward slashes; Path() handles both OSes.
        parquet_path = self.data_root / Path(*row["path"].split("/"))
        df = pd.read_parquet(parquet_path)

        # Preprocess
        seq = preprocess_sequence(df)  # (T, 134)

        # Truncate long sequences
        if self.max_seq_len and len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Collate function — pads variable-length sequences within a batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Pads sequences to the length of the longest sequence in the batch.

    Returns:
        sequences    : (B, T_max, 134)
        padding_mask : (B, T_max) bool — True where frames are padding
        labels       : (B,) long
    """
    sequences, labels = zip(*batch)
    lengths = [s.shape[0] for s in sequences]

    # pad_sequence stacks along dim=0 after padding to max length
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)   # (B, T_max, 134)

    T_max = padded.shape[1]
    padding_mask = torch.zeros(len(sequences), T_max, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, length:] = True   # True = padding (ignored in attention)

    labels = torch.stack(labels)
    return padded, padding_mask, labels


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    max_seq_len: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Build train and validation DataLoaders for the Google ASL dataset.

    Args:
        data_root    : path to the Kaggle dataset root directory.
        batch_size   : mini-batch size (paper uses 64).
        val_fraction : fraction of signers held out for validation.
        max_seq_len  : optional cap on sequence length.
        num_workers  : DataLoader worker processes.
        seed         : reproducibility seed.

    Returns:
        train_loader, val_loader, num_classes
    """
    train_ds = GoogleASLDataset(
        data_root, split="train",
        val_fraction=val_fraction,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    val_ds = GoogleASLDataset(
        data_root, split="val",
        val_fraction=val_fraction,
        max_seq_len=max_seq_len,
        label_map=train_ds.label_map,   # share the same label map
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.num_classes


# ---------------------------------------------------------------------------
# Quick test (no real data required — uses synthetic parquet files)
# ---------------------------------------------------------------------------

def _make_fake_parquet(n_frames: int) -> pd.DataFrame:
    """
    Generate a synthetic parquet in the real Google ASL long format:
        frame | row_id | type | landmark_index | x | y | z
    543 landmarks per frame: 468 face + 33 pose + 21 left_hand + 21 right_hand
    """
    rows = []
    lm_types = [("face", 468), ("pose", 33), ("left_hand", 21), ("right_hand", 21)]
    row_id = 0
    for frame in range(n_frames):
        for lm_type, count in lm_types:
            for idx in range(count):
                # Randomly NaN ~15% of hand landmarks to test interpolation
                nan = (lm_type in ("left_hand", "right_hand") and np.random.rand() < 0.15)
                rows.append({
                    "frame": frame,
                    "row_id": row_id,
                    "type": lm_type,
                    "landmark_index": idx,
                    "x": np.nan if nan else np.random.randn(),
                    "y": np.nan if nan else np.random.randn(),
                    "z": np.nan if nan else np.random.randn(),
                })
                row_id += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import tempfile

    print("Running synthetic pipeline test (long-format parquet)...")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        signs = [f"sign_{i}" for i in range(10)]
        sign_map = {s: i for i, s in enumerate(signs)}
        with open(root / "sign_to_prediction_index_map.json", "w") as f:
            json.dump(sign_map, f)

        records = []
        for participant in range(5):
            p_dir = root / "train_landmark_files" / str(participant)
            p_dir.mkdir(parents=True)
            for seq_id in range(4):
                T = np.random.randint(10, 30)
                df = _make_fake_parquet(T)
                rel_path = f"train_landmark_files/{participant}/{seq_id}.parquet"
                df.to_parquet(root / rel_path, index=False)
                records.append({
                    "path": rel_path,
                    "participant_id": participant,
                    "sequence_id": seq_id,
                    "sign": signs[seq_id % len(signs)],
                })

        pd.DataFrame(records).to_csv(root / "train.csv", index=False)

        # Test the dataset and loader
        train_loader, val_loader, n_cls = get_dataloaders(
            str(root), batch_size=4, num_workers=0
        )
        batch_seq, batch_mask, batch_labels = next(iter(train_loader))
        print(f"Batch sequences  : {batch_seq.shape}")    # (B, T_max, 134)
        print(f"Padding mask     : {batch_mask.shape}")   # (B, T_max)
        print(f"Labels           : {batch_labels}")
        print(f"Num classes      : {n_cls}")
        print(f"NaNs in batch    : {batch_seq.isnan().sum().item()}")
        print("Pipeline test passed ✓")