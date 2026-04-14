"""
data_sgsl.py  —  Augmentation-based pseudo-val split for the SGSL one-shot dataset
====================================================================================
All original samples train. A held-out subset has augmented copies used for
validation — giving a proxy loss curve for early stopping without removing any
class from training.

How it works:
  1. All N original .npy files go into the training set.
  2. A random subset (val_fraction of N) is selected as "pseudo-val seeds".
  3. For each pseudo-val seed, `n_aug_per_sample` augmented copies are generated
     on the fly (using a stronger augmentation than training, to ensure they're
     distinct enough to be a useful signal).
  4. Val set = these augmented copies only. No original appears in val.

Caveats (understood tradeoffs):
  - Val loss will be lower than a truly held-out set because augmented copies
    share the same underlying motion pattern as their source.
  - It is still useful for detecting training divergence and guiding early stopping.
  - Not suitable for reporting final generalisation performance — use K-Fold for that.

Layout expected:
    ./sgsl/
        pose_npy/
            hello.npy
            thank_you.npy
            ...
        label_map.json

Usage:
    train_loader, val_loader, num_classes = get_dataloaders(augmentor=my_augmentor)
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path("./sgsl")
NPY_DIR   = DATA_ROOT / "pose_npy"
LABEL_MAP = DATA_ROOT / "label_map.json"


# ---------------------------------------------------------------------------
# Training dataset — all original samples, augmented each epoch
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """
    All original .npy files. Augmentor is applied on every __getitem__ call
    so each epoch sees differently augmented versions.
    """

    def __init__(
        self,
        files: list,
        label_map: dict,
        max_seq_len: Optional[int] = None,
        augmentor=None,
    ):
        self.files       = files
        self.label_map   = label_map
        self.num_classes = len(label_map)
        self.max_seq_len = max_seq_len
        self.augmentor   = augmentor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        label = self.label_map[path.stem]
        seq   = np.load(path)   # (T, 134)

        if self.augmentor is not None:
            seq = self.augmentor(seq)

        if self.max_seq_len and len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        return (
            torch.tensor(seq,   dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Pseudo-val dataset — pre-generated augmented copies, fixed across epochs
# ---------------------------------------------------------------------------

class PseudoValDataset(Dataset):
    """
    Pre-generates `n_aug_per_sample` augmented copies of the val seed files
    at construction time. The copies are fixed for the lifetime of the dataset
    so val loss is consistent across epochs (not randomly re-augmented).

    Using a stronger augmentation than training creates more distinct copies
    and makes the val loss a more useful divergence signal.
    """

    def __init__(
        self,
        seed_files: list,
        label_map: dict,
        augmentor,                  # required — val set is augmentation-only
        n_aug_per_sample: int = 3,
        max_seq_len: Optional[int] = None,
        rng_seed: int = 99,
    ):
        if augmentor is None:
            raise ValueError("PseudoValDataset requires an augmentor — "
                             "val set is made entirely of augmented copies.")

        self.label_map   = label_map
        self.num_classes = len(label_map)
        self.max_seq_len = max_seq_len

        np.random.seed(rng_seed)

        # Pre-generate all augmented copies upfront and store in memory
        self.samples = []   # list of (seq_tensor, label_tensor)
        for path in seed_files:
            label = label_map[path.stem]
            orig  = np.load(path)   # (T, 134)
            for _ in range(n_aug_per_sample):
                aug = augmentor(orig.copy())
                if max_seq_len and len(aug) > max_seq_len:
                    aug = aug[:max_seq_len]
                self.samples.append((
                    torch.tensor(aug,   dtype=torch.float32),
                    torch.tensor(label, dtype=torch.long),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths  = [s.shape[0] for s in sequences]
    padded   = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    T_max    = padded.shape[1]
    pad_mask = torch.zeros(len(sequences), T_max, dtype=torch.bool)
    for i, length in enumerate(lengths):
        pad_mask[i, length:] = True
    return padded, pad_mask, torch.stack(labels)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(
    batch_size: int        = 64,
    val_fraction: float    = 0.2,
    max_seq_len: int       = 128,
    num_workers: int       = 4,
    augmentor              = None,
    n_aug_per_sample: int  = 3,
    seed: int              = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build train and pseudo-val DataLoaders.

    Args:
        batch_size       : mini-batch size
        val_fraction     : fraction of files used as pseudo-val seeds
        max_seq_len      : truncate sequences longer than this
        num_workers      : DataLoader worker count
        augmentor        : augmentation callable (T,134) → (T',134).
                           Applied to training samples each epoch, and used
                           to pre-generate the fixed pseudo-val copies.
        n_aug_per_sample : number of augmented copies per pseudo-val seed
        seed             : RNG seed for reproducible val seed selection

    Returns:
        train_loader, val_loader, num_classes
    """
    with open(LABEL_MAP) as f:
        label_map = json.load(f)

    all_files = sorted(NPY_DIR.glob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"No .npy files found in {NPY_DIR}")

    # Select pseudo-val seeds — these files ALSO stay in training
    rng        = np.random.default_rng(seed)
    n_val      = max(1, int(len(all_files) * val_fraction))
    val_indices = rng.choice(len(all_files), size=n_val, replace=False)
    val_seeds  = [all_files[i] for i in sorted(val_indices)]

    # Training = all original files (including the val seeds)
    train_ds = TrainDataset(
        all_files, label_map,
        max_seq_len=max_seq_len,
        augmentor=augmentor,
    )

    # Val = fixed augmented copies of the seed subset
    val_ds = PseudoValDataset(
        val_seeds, label_map,
        augmentor=augmentor,
        n_aug_per_sample=n_aug_per_sample,
        max_seq_len=max_seq_len,
        rng_seed=seed + 1,
    )

    print(
        f"[SGSL] train={len(train_ds)} (all originals) | "
        f"pseudo-val={len(val_ds)} ({n_aug_per_sample} aug × {n_val} seeds) | "
        f"classes={len(label_map)}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, len(label_map)