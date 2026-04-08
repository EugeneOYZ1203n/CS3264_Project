"""
Training Loop
=============
Ties together the SLR model, Google ASL data pipeline, and checkpoint
manager into a complete train / validate / resume workflow.

Quick start:
    python train.py --data_root path/to/asl-signs --save_dir checkpoints/google_asl

Resume interrupted training:
    python train.py --data_root path/to/asl-signs --save_dir checkpoints/google_asl --resume
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from slr_model        import SLRModel, build_optimizer, build_schedulers
from data_google_asl    import get_dataloaders
from checkpoint_manager import CheckpointManager, EarlyStopping


# ---------------------------------------------------------------------------
# F1 helper
# ---------------------------------------------------------------------------

def compute_f1(all_preds, all_labels):
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # Configure the loading bar
    # ncols=100 keeps the bar a consistent width
    pbar = tqdm(loader, desc="🚀 Training", unit="batch", ncols=100)

    for sequences, padding_mask, labels in pbar:
        sequences    = sequences.to(device)
        padding_mask = padding_mask.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences, padding_mask=padding_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping is vital for the Attention/Transformer stage 
        # you described to prevent exploding gradients.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        batch_loss = loss.item()
        total_loss += batch_loss * len(labels)
        
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # Update the loading bar postfix with the current loss
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    f1       = compute_f1(all_preds, all_labels)
    
    # Optional: Print final epoch stats after the bar finishes
    print(f" -> Epoch Complete | Loss: {avg_loss:.4f} | F1: {f1:.4f}")
    
    return avg_loss, f1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for sequences, padding_mask, labels in loader:
        sequences    = sequences.to(device)
        padding_mask = padding_mask.to(device)
        labels       = labels.to(device)

        logits = model(sequences, padding_mask=padding_mask)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1       = compute_f1(all_preds, all_labels)
    return avg_loss, f1


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_root:     str,
    save_dir:      str,
    resume:        bool  = False,
    max_epochs:    int   = 50,
    batch_size:    int   = 64,
    lr:            float = 3e-4,
    max_seq_len:   int   = None,
    num_workers:   int   = 4,
    embedding_dim: int   = 192,
    num_attn_layers: int = 4,
    num_attn_heads:  int = 8,
    dropout:       float = 0.1,
):
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Data root   : {data_root}")
    print(f"  Save dir    : {save_dir}")
    print(f"  Resume      : {resume}")
    print(f"{'='*60}\n")

    # ---- Data ------------------------------------------------------------
    train_loader, val_loader, num_classes = get_dataloaders(
        data_root,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
    )

    # ---- Model -----------------------------------------------------------
    model = SLRModel(
        num_keypoints=67,
        coords_per_kp=2,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        num_attn_layers=num_attn_layers,
        num_attn_heads=num_attn_heads,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}\n")

    # ---- Optimisation ----------------------------------------------------
    optimizer  = build_optimizer(model, lr=lr)
    scheduler  = build_schedulers(optimizer)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- Checkpointing & early stopping ----------------------------------
    ckpt = CheckpointManager(save_dir, metric_name="val_f1")
    es   = EarlyStopping(patience=20, mode="max")   # paper: patience=20 on val loss

    start_epoch = 0
    best_val_f1 = 0.0

    if resume and ckpt.has_checkpoint("latest"):
        start_epoch, best_val_f1 = ckpt.load_latest(model, optimizer, scheduler, device)
        es_val = ckpt.best_metric() or 0.0
        # Restore early stopping state to the best known metric
        es.best = es_val
        print(f"  Resuming from epoch {start_epoch}, best val_f1={best_val_f1:.4f}\n")
    elif resume:
        print("  No checkpoint found — starting fresh.\n")

    # ---- Training loop ---------------------------------------------------
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_f1   = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        # LR scheduler (monitors val F1 — higher is better)
        scheduler.step(val_f1)

        # Checkpoint
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1

        ckpt.save(
            epoch, model, optimizer, scheduler, val_f1,
            is_best=is_best,
            extra={"train_loss": train_loss, "train_f1": train_f1,
                   "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]},
        )

        print(
            f"  Epoch {epoch:3d}/{max_epochs-1} | "
            f"train loss={train_loss:.4f} f1={train_f1:.4f} | "
            f"val loss={val_loss:.4f} f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping (monitored on val F1)
        if es.step(val_f1):
            print(f"\n  Early stopping triggered after {epoch+1} epochs (patience={es.patience}).")
            break

    print(f"\n  Training complete. Best val_f1 = {best_val_f1:.4f}")
    print(f"  Best checkpoint saved to: {Path(save_dir) / 'best.pt'}\n")
    return model, best_val_f1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SLR model on Google ASL")
    parser.add_argument("--data_root",       required=True,        help="Path to the Kaggle dataset root")
    parser.add_argument("--save_dir",        default="checkpoints/google_asl")
    parser.add_argument("--resume",          action="store_true",  help="Resume from latest checkpoint")
    parser.add_argument("--max_epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--max_seq_len",     type=int,   default=None)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--embedding_dim",   type=int,   default=192)
    parser.add_argument("--num_attn_layers", type=int,   default=4)
    parser.add_argument("--num_attn_heads",  type=int,   default=8)
    parser.add_argument("--dropout",         type=float, default=0.1)
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        save_dir=args.save_dir,
        resume=args.resume,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        embedding_dim=args.embedding_dim,
        num_attn_layers=args.num_attn_layers,
        num_attn_heads=args.num_attn_heads,
        dropout=args.dropout,
    )