import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from preprocess import preprocess, N_HAND_KP, N_POSE_KP, POSE_INDICES, INPUT_DIM

# --- CONFIGURATION ---
data_root = Path("../output/poses/sgsl")
out_dir   = Path("./sgsl") # Main output directory
npy_dir   = out_dir / "pose_npy"     # Subfolder for arrays

# Ensure directories exist
out_dir.mkdir(parents=True, exist_ok=True)
npy_dir.mkdir(parents=True, exist_ok=True)

# MediaPipe Holistic component names
_LHAND_COMPONENT = "LEFT_HAND_LANDMARKS"
_RHAND_COMPONENT = "RIGHT_HAND_LANDMARKS"
_POSE_COMPONENT  = "POSE_LANDMARKS"

def load_pose_file(path: Path) -> np.ndarray:
    """Read a .pose file and extract (T, 134) keypoints."""
    with open(path, "rb") as f:
        pose = Pose.read(f.read(), NumPyPoseBody)

    component_names = [c.name for c in pose.header.components]

    def _extract(component_name: str, n_kp: int, kp_indices=None) -> np.ndarray:
        if component_name not in component_names:
            T = pose.body.data.shape[0]
            return np.full((T, n_kp, 2), np.nan, dtype=np.float32)

        sub  = pose.get_components([component_name])
        # Extract (T, person=0, points, dims=x,y)
        data = sub.body.data.data[:, 0, :, :2].astype(np.float32)

        if kp_indices is not None:
            data = data[:, kp_indices, :]
        return data

    lhand    = _extract(_LHAND_COMPONENT, N_HAND_KP)
    rhand    = _extract(_RHAND_COMPONENT, N_HAND_KP)
    pose_kps = _extract(_POSE_COMPONENT, N_POSE_KP, kp_indices=POSE_INDICES)

    T   = lhand.shape[0]
    seq = np.concatenate([lhand, rhand, pose_kps], axis=1) # (T, 67, 2)
    return seq.reshape(T, INPUT_DIM).astype(np.float32)     # (T, 134)

# --- EXECUTION ---
pose_files = sorted(list(data_root.glob("*.pose")))

# 1. Generate and save label map in the immediate out_dir
glosses   = sorted({f.stem for f in pose_files})
label_map = {g: i for i, g in enumerate(glosses)}
with open(out_dir / "label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print(f"Found {len(pose_files)} files. Starting extraction...")

# 2. Extract and save .npy files in the subfolder
for pose_path in tqdm(pose_files):
    out_path = npy_dir / f"{pose_path.stem}.npy"
    
    if out_path.exists():
        continue
        
    try:
        seq = load_pose_file(pose_path) # Extract (T, 134)
        seq = preprocess(seq)           # Interpolate NaNs
        np.save(out_path, seq)          # Save as float32
    except Exception as e:
        print(f"\nError processing {pose_path.name}: {e}")

print(f"\nExtraction complete.")
print(f"Label map -> {out_dir / 'label_map.json'}")
print(f"NPY files -> {npy_dir}/")