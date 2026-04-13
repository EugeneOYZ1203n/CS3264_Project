import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# --- CONFIGURATION ---
# Hardcoded paths as requested
DATA_ROOT = Path("./sgsl")
NPY_DIR   = DATA_ROOT / "pose_npy"
LABEL_MAP = DATA_ROOT / "label_map.json"

class PoseNpyDataset(Dataset):
    """
    Dataset that loads pre-extracted .npy files and a label map.
    """
    def __init__(
        self,
        split: str = "train",
        val_fraction: float = 0.2,
        max_seq_len: Optional[int] = None,
        seed: int = 42,
        augmentor=None,
    ):
        self.max_seq_len = max_seq_len
        self.augmentor = augmentor
        
        # 1. Load the label map
        with open(LABEL_MAP, "r") as f:
            self.label_map = json.load(f)
        self.num_classes = len(self.label_map)

        # 2. Collect all .npy files
        all_files = sorted(list(NPY_DIR.glob("*.npy")))
        
        # 3. Perform Split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        
        n_val = max(1, int(len(all_files) * val_fraction))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        if split == "train":
            self.files = [all_files[i] for i in train_idx]
        elif split == "val":
            self.files = [all_files[i] for i in val_idx]
        else:
            self.files = all_files

        print(f"[PoseNpy] {split:5s} | samples={len(self.files):<6} | classes={self.num_classes}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npy_path = self.files[idx]
        
        # Label is the filename (stem) looked up in label_map
        label_name = npy_path.stem
        label = self.label_map[label_name]

        # Load pre-processed numpy array (T, 134)
        seq = np.load(npy_path)

        if self.augmentor is not None:
            seq = self.augmentor(seq)

        if self.max_seq_len and len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- Collate Function ---
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths  = [s.shape[0] for s in sequences]
    padded   = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Attention Mask: True for padding, False for real data
    t_max    = padded.shape[1]
    pad_mask = torch.zeros(len(sequences), t_max, dtype=torch.bool)
    for i, length in enumerate(lengths):
        pad_mask[i, length:] = True
        
    return padded, pad_mask, torch.stack(labels)

# --- DataLoader Factory ---
def get_dataloaders(
    batch_size: int = 64,
    val_fraction: float = 0.15,
    max_seq_len: Optional[int] = 128,
    num_workers: int = 4,
    augmentor = None,
    seed: int = 42,
):
    train_ds = PoseNpyDataset("train", val_fraction, max_seq_len, seed, augmentor=augmentor)
    val_ds   = PoseNpyDataset("val",   val_fraction, max_seq_len, seed, augmentor=augmentor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_ds.num_classes