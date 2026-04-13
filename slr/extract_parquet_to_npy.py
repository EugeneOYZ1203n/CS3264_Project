# extract_to_npy.py  —  run once before training
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from preprocess import preprocess
from data_google_asl import extract_keypoints

data_root = Path("./asl-signs")
out_dir   = data_root / "train_landmarks_npy"
out_dir.mkdir(exist_ok=True)

meta = pd.read_csv(data_root / "train.csv")

for _, row in tqdm(meta.iterrows(), total=len(meta)):
    out_path = out_dir / f"{row['sequence_id']}.npy"
    if out_path.exists():
        continue
    parquet_path = data_root / Path(*row["path"].split("/"))
    df  = pd.read_parquet(parquet_path)
    seq = extract_keypoints(df)   # (T, 134), may have NaNs
    seq = preprocess(seq)         # interpolate missing
    np.save(out_path, seq)        # save as float32