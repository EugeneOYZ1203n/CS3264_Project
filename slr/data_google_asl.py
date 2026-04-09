"""
data_google_asl.py  —  DataLoader for the Google ASL Kaggle dataset
====================================================================
Parquet schema (long format):
    frame | row_id | type | landmark_index | x | y | z

One row per landmark per frame.
`type` ∈ {'face', 'pose', 'left_hand', 'right_hand'}

We select the same 67 keypoints described in the paper and apply
the shared preprocess pipeline (interpolate → normalise).

Usage:
    train_loader, val_loader, num_classes = get_dataloaders("./asl-signs")
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from preprocess import (
    preprocess, INPUT_DIM,
    N_HAND_KP, N_POSE_KP, POSE_INDICES,
)


# ---------------------------------------------------------------------------
# Keypoint extraction  (long-format parquet → (T, 134) array)
# ---------------------------------------------------------------------------

def extract_keypoints(df: pd.DataFrame) -> np.ndarray:
    """
    Extract (T, 134) from a long-format Google ASL parquet DataFrame.

    Selects:
        left_hand  landmarks 0-20   → 21 kp
        right_hand landmarks 0-20   → 21 kp
        pose       landmarks 11-35  → first 25 of POSE_INDICES
    Keeps x, y only (drops z).
    Returns NaN where a landmark was absent in a frame.
    """
    frames = sorted(df["frame"].unique())

    def _pivot(lm_type: str, lm_indices: list) -> np.ndarray:
        """Returns (T, len(lm_indices), 2), NaN where absent."""
        sub = df[(df["type"] == lm_type) & (df["landmark_index"].isin(lm_indices))]
        if sub.empty:
            return np.full((len(frames), len(lm_indices), 2), np.nan, dtype=np.float32)

        # Map original landmark_index → position in our selection
        idx_map = {orig: pos for pos, orig in enumerate(lm_indices)}
        sub = sub.copy()
        sub["sel_idx"] = sub["landmark_index"].map(idx_map)

        px = sub.pivot(index="frame", columns="sel_idx", values="x")
        py = sub.pivot(index="frame", columns="sel_idx", values="y")

        cols = list(range(len(lm_indices)))
        px = px.reindex(index=frames, columns=cols)
        py = py.reindex(index=frames, columns=cols)

        return np.stack([px.values, py.values], axis=-1).astype(np.float32)

    lhand = _pivot("left_hand",  list(range(N_HAND_KP)))   # (T, 21, 2)
    rhand = _pivot("right_hand", list(range(N_HAND_KP)))   # (T, 21, 2)
    pose  = _pivot("pose",       POSE_INDICES)              # (T, 25, 2)

    T   = len(frames)
    kps = np.concatenate([lhand, rhand, pose], axis=1)     # (T, 67, 2)
    return kps.reshape(T, INPUT_DIM).astype(np.float32)    # (T, 134)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GoogleASLDataset(Dataset):
    """
    Signer-independent split: no signer appears in both train and val.

    Args:
        data_root    : path to the unzipped Kaggle asl-signs directory
        split        : 'train' | 'val' | 'all'
        val_fraction : fraction of signers held out for val
        max_seq_len  : truncate sequences longer than this (None = no limit)
        label_map    : sign→int dict; built from sign_to_prediction_index_map.json
                       if not provided
        seed         : RNG seed for the signer split
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
        assert split in ("train", "val", "all")
        self.data_root   = Path(data_root)
        self.max_seq_len = max_seq_len

        meta = pd.read_csv(self.data_root / "train.csv")

        # Label map
        if label_map is None:
            sign_map_path = self.data_root / "sign_to_prediction_index_map.json"
            if sign_map_path.exists():
                with open(sign_map_path) as f:
                    self.label_map = json.load(f)
            else:
                signs = sorted(meta["sign"].unique())
                self.label_map = {s: i for i, s in enumerate(signs)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

        # Signer-independent split
        rng        = np.random.default_rng(seed)
        signers    = meta["participant_id"].unique()
        rng.shuffle(signers)
        n_val      = max(1, int(len(signers) * val_fraction))
        val_signers = set(signers[:n_val])

        if split == "train":
            meta = meta[~meta["participant_id"].isin(val_signers)]
        elif split == "val":
            meta = meta[meta["participant_id"].isin(val_signers)]

        self.meta = meta.reset_index(drop=True)
        print(
            f"[GoogleASL] {split:5s} | samples={len(self.meta):,} | "
            f"signers={self.meta['participant_id'].nunique()} | "
            f"classes={self.num_classes}"
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row   = self.meta.iloc[idx]
        label = self.label_map[row["sign"]]

        # Windows-safe path join (train.csv uses forward slashes on all OSes)
        parquet_path = self.data_root / Path(*row["path"].split("/"))
        df  = pd.read_parquet(parquet_path)
        seq = extract_keypoints(df)          # (T, 134), may have NaNs
        seq = preprocess(seq)                # interpolate + normalise

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
        pad_mask[i, l:] = True      # True = padding (ignored in attention)
    return padded, pad_mask, torch.stack(labels)


def get_dataloaders(
    data_root: str,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    max_seq_len: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
):
    train_ds = GoogleASLDataset(data_root, "train", val_fraction, max_seq_len, seed=seed)
    val_ds   = GoogleASLDataset(data_root, "val",   val_fraction, max_seq_len,
                                label_map=train_ds.label_map, seed=seed)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.num_classes