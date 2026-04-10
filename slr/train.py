"""
train.py  —  Simple training script for SLRModel

Usage:
    python train.py --data_root ./asl-signs
    python train.py --data_root ./asl-signs --resume
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics    import f1_score
from tqdm import tqdm

from model              import SLRModel
from data_google_asl    import get_dataloaders
from checkpoint_manager import CheckpointManager, EarlyStopping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_f1(preds, labels):
    return f1_score(labels, preds, average="macro", zero_division=0)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Single pass over loader. Pass optimizer=None for validation."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for sequences, padding_mask, labels in loader:
            sequences    = sequences.to(device)
            padding_mask = padding_mask.to(device)
            labels       = labels.to(device)

            logits = model(sequences, padding_mask=padding_mask)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader.dataset), compute_f1(all_preds, all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Device    : {device}")
    print(f"  Data root : {args.data_root}")
    print(f"  Save dir  : {args.save_dir}")
    print(f"{'='*55}\n")

    # Data
    train_loader, val_loader, num_classes = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=128,
        npy_dir= f"{args.data_root}/train_landmarks_npy"
    )

    # Model
    start_prob = 1.0
    end_prob = 1.0
    if args.stoch_drop:
        start_prob = 0.95
        end_prob = 0.2
    model = SLRModel(num_classes=num_classes, stochastic_drop_start_prob=start_prob, stochastic_drop_end_prob=end_prob).to(device)
    print(f"  Params    : {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimiser + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.scheduler == "onecyclelr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=0.001, 
            total_steps=total_steps
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5
        )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Checkpointing + early stopping
    ckpt = CheckpointManager(args.save_dir, metric_name="val_f1", verbose=False)
    es = EarlyStopping(patience=20, mode="min")

    log_path = Path(args.save_dir) / "history.csv"
    
    if not args.resume or not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_f1,val_loss,val_f1,lr\n")

    start_epoch, best_f1 = 0, 0.0
    if args.resume and ckpt.has_checkpoint():
        start_epoch, best_f1 = ckpt.load_latest(model, optimizer, scheduler, device)
        es.best = ckpt.best_metric() or 0.0
        print(f"  Resumed from epoch {start_epoch},  best val_f1={best_f1:.4f}\n")

    # ---- Training loop ---------------------------------------------------
    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc="  Epochs",
        unit="ep",
        ncols=100,
    )

    for epoch in epoch_bar:
        t0 = time.time()

        # --- Train ---
        train_bar = tqdm(train_loader, desc="  train", unit="batch",
                         ncols=100, leave=False)
        model.train()
        t_loss, t_preds, t_labels = 0.0, [], []
        for seqs, masks, lbls in train_bar:
            seqs, masks, lbls = seqs.to(device), masks.to(device), lbls.to(device)
            logits = model(seqs, padding_mask=masks)
            loss   = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.scheduler == "onecyclelr":
                scheduler.step()
            t_loss  += loss.item() * len(lbls)
            t_preds.extend(logits.argmax(1).cpu().tolist())
            t_labels.extend(lbls.cpu().tolist())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = t_loss / len(train_loader.dataset)
        train_f1   = compute_f1(t_preds, t_labels)

        # --- Validate ---
        val_bar = tqdm(val_loader, desc="  val  ", unit="batch",
                       ncols=100, leave=False)
        model.eval()
        v_loss, v_preds, v_labels = 0.0, [], []
        with torch.no_grad():
            for seqs, masks, lbls in val_bar:
                seqs, masks, lbls = seqs.to(device), masks.to(device), lbls.to(device)
                logits = model(seqs, padding_mask=masks)
                loss   = criterion(logits, lbls)
                v_loss  += loss.item() * len(lbls)
                v_preds.extend(logits.argmax(1).cpu().tolist())
                v_labels.extend(lbls.cpu().tolist())
                val_bar.set_postfix(loss=f"{loss.item():.4f}")
        val_loss = v_loss / len(val_loader.dataset)
        val_f1   = compute_f1(v_preds, v_labels)

        if args.scheduler == "reduceonplateau":
            scheduler.step(val_f1)

        # Checkpoint
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
        ckpt.save(epoch, model, optimizer, scheduler, val_f1, is_best=is_best,
                  extra={"train_loss": train_loss, "val_loss": val_loss})
        
        current_lr = optimizer.param_groups[0]['lr']
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_f1:.6f},{val_loss:.6f},{val_f1:.6f},{current_lr:.2e}\n")

        # Update outer bar + persistent log line
        epoch_bar.set_postfix(
            tr_f1=f"{train_f1:.3f}", val_f1=f"{val_f1:.3f}",
            best=f"{best_f1:.3f}", t=f"{time.time()-t0:.0f}s"
        )
        tqdm.write(
            f"  ep {epoch:3d} | "
            f"train {train_loss:.4f}/{train_f1:.4f} | "
            f"val {val_loss:.4f}/{val_f1:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | "
            f"{time.time()-t0:.1f}s" + (" ★" if is_best else "")
        )

        if es.step(val_loss):
            tqdm.write(f"\n  Early stopping at epoch {epoch+1}.")
            break

    tqdm.write(f"\n  Done.  Best val_f1={best_f1:.4f}")
    tqdm.write(f"  Best checkpoint → {Path(args.save_dir) / 'best.pt'}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--save_dir",    default="checkpoints/google_asl")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--scheduler",   type=str,   default="reduceonplateau", choices=["onecyclelr", "reduceonplateau"])
    p.add_argument("--stoch_drop",  action="store_true")
    train(p.parse_args())