import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from model import SLRModel
#from model_alternative import KerasStyleSLRModel as SLRModel
#from daniel_model.model.slr_model_v3 import SLRModel
from preprocess import interpolate_missing, normalise, POSE_INDICES
from mediapipe.framework.formats import landmark_pb2

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/sgsl_fton_google_asl_dataaug_onecycle_awp_multiphase2/best.pt"
LABEL_MAP_PATH = "sgsl/label_map.json"
#LABEL_MAP_PATH = "daniel_model/weights/sgsl_sign_to_idx_map.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.10  # Threshold for "Don't know"

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

# Constants
FINGER_CHAINS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], 
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
]

POSE_CONNECTIONS = [
    (42+0, 42+7), (42+0, 42+8), (42+11, 42+12), (42+11, 42+13),
    (42+13, 42+15), (42+12, 42+14), (42+14, 42+16), (42+11, 42+23),
    (42+12, 42+24), (42+23, 42+24)
]

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

    # 1. Identify indices for stabilization
    # In POSE_INDICES, typically 11 is Left Shoulder, 12 is Right Shoulder
    # We need to find where they sit in your 'kps' array (indices 42:67)
    L_SHOULDER_IDX = 42 + 11  # Adjust based on your POSE_INDICES mapping
    R_SHOULDER_IDX = 42 + 12  

    # 2. Stable Normalization (Anchor to shoulders)
    valid_shoulders = ~np.isnan(kps[[L_SHOULDER_IDX, R_SHOULDER_IDX]]).any(axis=1)
    
    if valid_shoulders.all():
        # Center point between shoulders
        ZOOM_FACTOR = 4.0  # Changed from 4.0 to 6.0 to scale it down
        
        # Center point between shoulders
        center = (kps[L_SHOULDER_IDX] + kps[R_SHOULDER_IDX]) / 2
        
        # Use shoulder distance as the base unit of measurement
        shoulder_dist = np.linalg.norm(kps[L_SHOULDER_IDX] - kps[R_SHOULDER_IDX])
        
        # Apply the scale factor
        scale = shoulder_dist * ZOOM_FACTOR
        
        # Transform: Center the person and scale
        # We map the 'center' to (0.5, 0.4) on the canvas
        kps = (kps - center) / (scale + 1e-6)
        kps = kps + np.array([0.5, 0.4]) 
    else:
        # Fallback to your old min-max logic if shoulders aren't visible
        valid = kps[~np.isnan(kps).any(axis=1)]
        if len(valid) > 0:
            min_xy = valid.min(axis=0)
            max_xy = valid.max(axis=0)
            kps = (kps - min_xy) / (np.maximum(max_xy - min_xy, 1e-6))

    # --- Rest of your drawing code (splitting, landmark lists, etc.) remains same ---
    lhand = kps[0:21]
    rhand = kps[21:42]
    pose25 = kps[42:67]
    corners = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    corner_connections = []

    # --- rebuild FULL 33 pose ---
    full_pose = np.full((33, 2), np.nan)

    for i, idx in enumerate(POSE_INDICES):
        if i < len(pose25):
            full_pose[idx] = pose25[i]

    # --- convert to mediapipe format ---
    lhand_lms = create_landmark_list(lhand)
    rhand_lms = create_landmark_list(rhand)
    pose_lms  = create_landmark_list(full_pose)
    corner_lms = create_landmark_list(corners)

    # --- draw ---
    pose_connections = filter_connections(full_pose, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(canvas, pose_lms, pose_connections)

    lhand_connections = filter_connections(lhand, mp_holistic.HAND_CONNECTIONS)
    rhand_connections = filter_connections(rhand, mp_holistic.HAND_CONNECTIONS)

    mp_drawing.draw_landmarks(canvas, lhand_lms, lhand_connections)
    mp_drawing.draw_landmarks(canvas, rhand_lms, rhand_connections)

    mp_drawing.draw_landmarks(
        canvas,
        corner_lms,
        corner_connections
    )

    return canvas

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    top_5_list = [] 
    panel_w = 800 
    last_display = None

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success: 
                if last_display is not None:
                    linger_time_ms = 1000

                    cv2.imshow("SgSL Real-Time Inference", last_display)
                    cv2.waitKey(linger_time_ms)
                    break

            h_orig, w_orig = frame.shape[:2]
            panel_h = int(panel_w * (h_orig / w_orig))
            frame = cv2.resize(frame, (panel_w, panel_h))
            
            original_img = frame.copy()
            overlay_img = frame.copy()
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            def get_coords(res_attr, count):
                return [[lm.x, lm.y] for lm in res_attr.landmark] if res_attr else [[np.nan, np.nan]] * count

            lhand = get_coords(results.left_hand_landmarks, 21)
            rhand = get_coords(results.right_hand_landmarks, 21)
            pose_kps = [[results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y] 
                        for i in POSE_INDICES] if results.pose_landmarks else [[np.nan, np.nan]] * 25
            
            flat_feat = np.array(lhand + rhand + pose_kps).flatten()
            sequence_data.append(flat_feat)

            norm_view = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            
            if len(sequence_data) > 2:
                current_window = np.array(sequence_data, dtype=np.float32)
                processed_seq = interpolate_missing(current_window)
                processed_seq = normalise(processed_seq)
                norm_view = draw_normalized_mediapipe(panel_w, panel_h, processed_seq[-1])

                input_tensor = torch.tensor(processed_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)
                    top_probs, top_idxs = torch.topk(probs, 10)
                    top_5_list = [(rev_label_map[idx.item()], prob.item()) for prob, idx in zip(top_probs[0], top_idxs[0])]


            mp_drawing.draw_landmarks(overlay_img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(overlay_img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(overlay_img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # --- ASSEMBLE DASHBOARD ---
            h, w = overlay_img.shape[:2]

            #norm_view = cv2.resize(norm_view, (w, h))
            original_img = cv2.resize(original_img, (w, h))
            
            # Prediction Logic with Uncertainty Threshold
            # --- CONFIG ---
            TOP_K = 10  # number of top predictions to display

            # --- INFO PANEL ---
            inf_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

            # Prediction logic
            if not top_5_list:
                main_display = "WAITING..."
                top_k_display = []
            elif top_5_list[0][1] < CONFIDENCE_THRESHOLD:
                main_display = "DON'T KNOW"
                top_k_display = top_5_list[:TOP_K]
            else:
                main_display = top_5_list[0][0].upper()
                top_k_display = top_5_list[:TOP_K]

            # --- DRAW TEXT ---
            # Title
            cv2.putText(
                inf_panel,
                f"PRED SO FAR: {main_display}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 200),
                2
            )

            # Stacked top-k predictions
            start_y = 80
            line_height = 30

            for i, (label, conf) in enumerate(top_k_display):
                text = f"{i+1}. {label}: {conf:.2f}"
                y = start_y + i * line_height

                # Highlight top prediction
                color = (0, 255, 0) if i == 0 else (200, 200, 200)

                cv2.putText(
                    inf_panel,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1
                )
            
            top_row = np.hstack((original_img, overlay_img))
            bottom_row = np.hstack((norm_view, inf_panel))

            display = np.vstack((top_row, bottom_row))
            last_display = display.copy()
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