import matplotlib.pyplot as plt
import numpy as np
from data_google_asl    import get_dataloaders
from tqdm import tqdm

def plot_sequence_lengths(dataset, title="Sequence Length Distribution"):
    lengths = []
    
    # Wrap the range in tqdm for a progress bar
    # We use a custom desc so you know exactly what's happening
    print(f"Analyzing {len(dataset)} samples...")
    
    for i in tqdm(range(len(dataset)), desc="Reading Parquet Lengths", unit="file"):
        # We access the meta directly to get the path without 
        # running the full preprocess pipeline (interpolation/norm)
        # to make this check 10x faster.
        row = dataset.meta.iloc[i]
        parquet_path = dataset.data_root / row["path"]
        
        # Fast way: read just the 'frame' column to get the count
        try:
            df = pd.read_parquet(parquet_path, columns=["frame"])
            lengths.append(df["frame"].nunique())
        except Exception:
            # Fallback to standard getitem if path logic differs
            seq, _ = dataset[i]
            lengths.append(len(seq))

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
    
    stats = {
        'Mean': np.mean(lengths),
        'Median': np.median(lengths),
        '95th %': np.percentile(lengths, 95),
        'Max': np.max(lengths)
    }
    
    colors = ['red', 'blue', 'green', 'orange']
    for (label, value), color in zip(stats.items(), colors):
        plt.axvline(value, color=color, linestyle='--', label=f'{label}: {value:.1f}')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Number of Frames', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.show()

# Usage
train_loader, val_loader, _ = get_dataloaders("./asl-signs")
plot_sequence_lengths(train_loader.dataset)