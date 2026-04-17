import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import json
from collections import deque # Added for sliding window
from pathlib import Path
from model import SLRModel
from preprocess import interpolate_missing, normalise, POSE_INDICES
from mediapipe.framework.formats import landmark_pb2

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/sgsl_fton_google_asl_dataaug_onecycle_awp_multiphase2/best.pt"
LABEL_MAP_PATH = "sgsl/label_map.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.15  # Increased for continuous recognition to avoid noise
WINDOW_SIZE = 128            # Previous 128 frames

# Load Label Map
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
rev_label_map = {v: k for k, v in label_map.items()}

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 1. Load Model
model = SLRModel(input_dim=134, num_classes=len(label_map), feature_extract=True)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def create_landmark_list(coords):
    landmark_list = landmark_pb2.NormalizedLandmarkList()

    for x, y in coords:
        lm = landmark_list.landmark.add()
        lm.x = float(x) if not np.isnan(x) else 0.0
        lm.y = float(y) if not np.isnan(y) else 0.0
        lm.z = 0.0

    return landmark_list

def filter_connections(coords, connections):
    valid = ~np.isnan(coords).any(axis=1)
    return [(i, j) for (i, j) in connections if valid[i] and valid[j]]

def draw_normalized_mediapipe(width, height, processed_feat):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    kps = processed_feat.reshape(-1, 2)

    L_SHOULDER_IDX = 42 + 0 
    R_SHOULDER_IDX = 42 + 1  

    valid_shoulders = ~np.isnan(kps[[L_SHOULDER_IDX, R_SHOULDER_IDX]]).any(axis=1)
    
    if valid_shoulders.all():
        ZOOM_FACTOR = 6.0  # Scaled down towards center
        center = (kps[L_SHOULDER_IDX] + kps[R_SHOULDER_IDX]) / 2
        shoulder_dist = np.linalg.norm(kps[L_SHOULDER_IDX] - kps[R_SHOULDER_IDX])
        scale = shoulder_dist * ZOOM_FACTOR
        
        kps = (kps - center) / (scale + 1e-6)
        kps = kps + np.array([0.5, 0.4]) 
    else:
        valid = kps[~np.isnan(kps).any(axis=1)]
        if len(valid) > 0:
            min_xy = valid.min(axis=0)
            max_xy = valid.max(axis=0)
            kps = (kps - min_xy) / (np.maximum(max_xy - min_xy, 1e-6))

    # Splitting for drawing
    lhand, rhand, pose25 = kps[0:21], kps[21:42], kps[42:67]
    
    # [Rest of drawing logic: lhand_lms, rhand_lms, etc. - keep your existing code here]
    # Rebuild full_pose and draw via mp_drawing...
    # (Abbreviated for brevity)
    return canvas

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # SLIDING WINDOW: This will only keep the last 128 frames
    sequence_data = deque(maxlen=WINDOW_SIZE)
    
    top_5_list = [] 
    frame_count = 0
    panel_w = 800 
    last_display = None

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame_count += 1
            h_orig, w_orig = frame.shape[:2]
            panel_h = int(panel_w * (h_orig / w_orig))
            frame = cv2.resize(frame, (panel_w, panel_h))
            
            original_img = frame.copy()
            overlay_img = frame.copy()
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Extraction
            def get_coords(res_attr, count):
                return [[lm.x, lm.y] for lm in res_attr.landmark] if res_attr else [[np.nan, np.nan]] * count

            lhand = get_coords(results.left_hand_landmarks, 21)
            rhand = get_coords(results.right_hand_landmarks, 21)
            pose_kps = [[results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y] 
                        for i in POSE_INDICES] if results.pose_landmarks else [[np.nan, np.nan]] * 25
            
            flat_feat = np.array(lhand + rhand + pose_kps).flatten()
            sequence_data.append(flat_feat) # Deque handles the sliding window logic automatically

            norm_view = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            
            # INFERENCE LOGIC
            current_window = np.array(list(sequence_data), dtype=np.float32)
            
            # Preprocess the 128 frame window
            processed_seq = interpolate_missing(current_window)
            processed_seq = normalise(processed_seq)
            
            # Visualize the latest frame in the buffer
            norm_view = draw_normalized_mediapipe(panel_w, panel_h, processed_seq[-1])

            input_tensor = torch.tensor(processed_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)
                top_probs, top_idxs = torch.topk(probs, 5)
                top_5_list = [(rev_label_map[idx.item()], prob.item()) for prob, idx in zip(top_probs[0], top_idxs[0])]

            # --- Drawing Overlay & UI ---
            mp_drawing.draw_landmarks(overlay_img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(overlay_img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(overlay_img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            inf_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            
            # Update prediction display
            main_display = "WAITING..."
            if top_5_list:
                if top_5_list[0][1] < CONFIDENCE_THRESHOLD:
                    main_display = "LOW CONFIDENCE"
                else:
                    main_display = top_5_list[0][0].upper()

            cv2.putText(inf_panel, f"PRED: {main_display}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 2)

            for i, (label, conf) in enumerate(top_5_list):
                y = 120 + i * 35
                cv2.putText(inf_panel, f"{label}: {conf:.2%}", (20, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Assemble and Show
            top_row = np.hstack((cv2.resize(original_img, (panel_w, panel_h)), overlay_img))
            bottom_row = np.hstack((norm_view, inf_panel))
            display = np.vstack((top_row, bottom_row))
            
            cv2.imshow("SgSL Real-Time Inference", display)
            if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    process_video(args.input)