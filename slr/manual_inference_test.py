import datetime

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import json
import os
from pathlib import Path
from tqdm import tqdm
from model import SLRModel
#from model_alternative import KerasStyleSLRModel as SLRModel
#from daniel_model.model.slr_model_v3 import SLRModel
from preprocess import POSE_INDICES, interpolate_missing, normalise
#from daniel_model.scripts.processing.process_google_asl import interpolate_missing, normalize as normalise

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/sgsl_fton_google_asl_dataaug_onecycle_awp_multiphase1/best.pt"
LABEL_MAP_PATH = "sgsl/label_map.json"
#LABEL_MAP_PATH = "daniel_model/weights/sgsl_sign_to_idx_map.json"
DATA_ROOT = "manual" # Change this
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rearrange_to_pose_first(seq):
    """
    Input:  seq with shape (T, 67, 2) ordered [Lhand, Rhand, Pose]
    Output: seq with shape (T, 67, 2) ordered [Pose, Lhand, Rhand]
    """
    # Slice the components based on your indices
    lhand = seq[:, 0:21, :]
    rhand = seq[:, 21:42, :]
    pose  = seq[:, 42:67, :]

    # Re-concatenate in the desired order
    new_seq = np.concatenate([pose, lhand, rhand], axis=1)
    
    return new_seq

def get_inference(video_path, model, holistic, rev_label_map):
    cap = cv2.VideoCapture(str(video_path))
    sequence_data = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        def get_coords(res_attr, count):
            return [[lm.x, lm.y] for lm in res_attr.landmark] if res_attr else [[np.nan, np.nan]] * count

        lhand = get_coords(results.left_hand_landmarks, 21)
        rhand = get_coords(results.right_hand_landmarks, 21)
        pose_kps = [[results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y] 
                    for i in POSE_INDICES] if results.pose_landmarks else [[np.nan, np.nan]] * 25
        
        sequence_data.append(np.array(lhand + rhand + pose_kps).flatten())
    cap.release()

    if len(sequence_data) < 5: return None 

    processed_seq = interpolate_missing(np.array(sequence_data, dtype=np.float32))
    processed_seq = normalise(processed_seq)
    
    input_tensor = torch.tensor(processed_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        idx = torch.argmax(F.softmax(logits, dim=1), dim=1).item()
    
    # Use rev_label_map to return the string name
    return rev_label_map[idx]

def run_evaluation():
    # 1. Setup
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    rev_label_map = {v: k for k, v in label_map.items()}

    model = SLRModel(input_dim=134, num_classes=len(label_map))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    mp_holistic = mp.solutions.holistic
    
    # 2. Trackers
    results_log = [] # List of (true_label, pred_label, signer_name, sign_name)
    
    # 3. Iterate Folders
    root_path = Path(DATA_ROOT)
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.WMV', '*.flv']
    video_files = []
    for ext in extensions:
        # rglob handles recursive search if your structure is nested
        video_files.extend(list(root_path.rglob(ext)))

    print(f"Starting evaluation on {len(video_files)} videos...")

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        for v_path in tqdm(video_files):
            sign_name = v_path.parent.name
            true_label = label_map[sign_name]
            
            # Extract Signer Name (Assumes <SignerName><Number>.mp4)
            # Regex or simple digit stripping
            filename = v_path.stem
            signer_name = ''.join([i for i in filename if not i.isdigit()])

            pred_label = get_inference(v_path, model, holistic, rev_label_map)
            
            if pred_label is not None:
                results_log.append({
                    "true": sign_name,
                    "pred": pred_label,
                    "signer": signer_name,
                })

    # --- SAVE RESULTS ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("evaluation_logs")
    log_dir.mkdir(exist_ok=True)

    # 1. Save Raw JSON Log
    json_path = log_dir / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_log, f, indent=4)

    # 2. Generate and Save Summary Report
    summary_path = log_dir / f"summary_{timestamp}.txt"
    
    # Calculate Stats
    correct_total = sum(1 for r in results_log if r["true"] == r["pred"])
    overall_acc = correct_total / len(results_log) if results_log else 0

    with open(summary_path, "w") as f:
        f.write("SgSL MODEL EVALUATION SUMMARY\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write("="*40 + "\n")
        f.write(f"OVERALL ACCURACY: {overall_acc:.4f} ({correct_total}/{len(results_log)})\n")
        f.write("="*40 + "\n\n")

        # Per-Signer Block
        f.write("PER-SIGNER ACCURACY:\n")
        signer_names = sorted(list(set(r["signer"] for r in results_log)))
        for sn in signer_names:
            s_results = [r for r in results_log if r["signer"] == sn]
            s_corr = sum(1 for r in s_results if r["true"] == r["pred"])
            s_acc = s_corr / len(s_results)
            f.write(f"{sn:15}: {s_acc:.4f} ({s_corr}/{len(s_results)})\n")
        
        # --- Per-Sign Block with Error Analysis ---
        f.write("\nPER-SIGN ACCURACY & COMMON MISTAKES:\n")
        f.write(f"{'Sign Name':<20} | {'Acc':<7} | {'Commonly Mistaken For':<30}\n")
        f.write("-" * 70 + "\n")
        
        sign_names = sorted(list(set(r["true"] for r in results_log)))
        for s in sign_names:
            sig_results = [r for r in results_log if r["true"] == s]
            sig_corr = sum(1 for r in sig_results if r["true"] == r["pred"])
            sig_acc = sig_corr / len(sig_results)
            
            # Identify mistakes
            mistakes = [r["pred"] for r in sig_results if r["true"] != r["pred"]]
            if mistakes:
                # Count occurrences of each mistake and sort by frequency
                mistake_counts = {}
                for m in mistakes:
                    mistake_counts[m] = mistake_counts.get(m, 0) + 1
                
                # Format the top 3 mistakes: "Label (count)"
                sorted_mistakes = sorted(mistake_counts.items(), key=lambda x: x[1], reverse=True)
                mistake_str = ", ".join([f"{m} ({c})" for m, c in sorted_mistakes[:3]])
            else:
                mistake_str = "None (Perfect)"

            f.write(f"{s:<20} | {sig_acc:.4f} | {mistake_str}\n")

    print(f"\nEvaluation Complete!")
    print(f"Raw results: {json_path}")
    print(f"Summary report: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    CHECKPOINT_PATH = args.path
    run_evaluation()