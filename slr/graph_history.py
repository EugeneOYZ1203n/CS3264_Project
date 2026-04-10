import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_history(csv_paths):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    colors = plt.cm.tab10.colors 
    
    for i, path_str in enumerate(csv_paths):
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        df = pd.read_csv(path)
        
        # --- FIX 1: Use parent folder name as label ---
        # e.g., 'experiments/v1_baseline/log.csv' -> 'v1_baseline'
        label_prefix = path.parent.name if path.parent.name else path.stem
        color = colors[i % len(colors)]

        # --- Subplot 1: Loss ---
        ax1.plot(df['epoch'], df['train_loss'], '--', color=color, alpha=0.3)
        ax1.plot(df['epoch'], df['val_loss'], '-', color=color, label=f'{label_prefix}')
        
        # --- FIX 2: Highlight Lowest Loss ---
        min_loss_idx = df['val_loss'].idxmin()
        min_loss_val = df.loc[min_loss_idx, 'val_loss']
        min_loss_ep  = df.loc[min_loss_idx, 'epoch']
        ax1.scatter(min_loss_ep, min_loss_val, color=color, s=80, edgecolors='black', zorder=5)
        ax1.annotate(f'Ep {int(min_loss_ep)}: {min_loss_val:.3f}', 
                     (min_loss_ep, min_loss_val), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8, color=color, weight='bold')

        # --- Subplot 2: F1 Score ---
        ax2.plot(df['epoch'], df['train_f1'], '--', color=color, alpha=0.3)
        ax2.plot(df['epoch'], df['val_f1'], '-', color=color, label=f'{label_prefix}')

        # --- FIX 3: Highlight Highest F1 ---
        max_f1_idx = df['val_f1'].idxmax()
        max_f1_val = df.loc[max_f1_idx, 'val_f1']
        max_f1_ep  = df.loc[max_f1_idx, 'epoch']
        ax2.scatter(max_f1_ep, max_f1_val, color=color, s=80, edgecolors='black', zorder=5)
        ax2.annotate(f'Ep {int(max_f1_ep)}: {max_f1_val:.3f}', 
                     (max_f1_ep, max_f1_val), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8, color=color, weight='bold')

        # --- Subplot 3: Learning Rate ---
        ax3.plot(df['epoch'], df['lr'], '-', color=color, label=f'{label_prefix}')

    # Formatting Loss Plot
    ax1.set_title('Validation Loss (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize='small', loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Formatting F1 Plot
    ax2.set_title('Macro F1 Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend(fontsize='small', loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Formatting LR Plot
    ax3.set_title('Learning Rate (Log Scale)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('LR')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which="both")

    plt.suptitle('SLR Training Performance Comparison', fontsize=18, y=1.02)
    plt.tight_layout()
    
    save_name = "training_comparison.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as {save_name}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs='+')
    args = parser.parse_args()
    plot_history(args.csv_files)