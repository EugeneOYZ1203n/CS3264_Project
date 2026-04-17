"""
gif_to_pose.py  –  Batch GIF → .pose converter for sign-vq
------------------------------------------------------------
Processes every .gif in an input folder, runs MediaPipe Holistic
(with the same settings and hand-interpolation logic as debug_extract.py),
and writes a .pose file compatible with the sign-vq / pose-format ecosystem.

Install dependencies:
    pip install mediapipe Pillow numpy pose-format

Usage:
    # Basic – output beside the GIFs
    python gif_to_pose.py --input_dir output/gifs

    # Custom output directory, 8 parallel workers
    python gif_to_pose.py --input_dir output/gifs --output_dir poses/ --workers 8

    # Skip files that already have a .pose (handy for resuming a crashed run)
    python gif_to_pose.py --input_dir output/gifs --output_dir poses/ --skip_existing

    # Overwrite confidence thresholds (mediapipe defaults shown)
    python gif_to_pose.py --input_dir output/gifs --detection_conf 0.5 --tracking_conf 0.5

After creating the .pose files, feed them to sign-vq:
    python -m sign_vq.data.zip_dataset --dir=poses/ --out=normalized.zip
    python -m sign_vq.train --data-path=normalized.zip
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ── MediaPipe Holistic landmark counts ────────────────────────────────────────
#   Pose: 33 keypoints  (x, y, z, visibility)
#   Hands: 21 keypoints each
#   Face: 468 keypoints

POSE_LEFT_WRIST  = 15
POSE_RIGHT_WRIST = 16


# ── GIF loading ───────────────────────────────────────────────────────────────

def load_gif(path: str):
    """Return (frames: list[np.ndarray[H,W,3]], fps: float)."""
    gif = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(np.array(gif.copy().convert("RGB")))
            durations.append(gif.info.get("duration", 67))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    avg_ms = sum(durations) / max(len(durations), 1)
    return frames, 1000.0 / avg_ms


# ── MediaPipe extraction ──────────────────────────────────────────────────────

def extract_all(frames, detection_conf=0.5, tracking_conf=0.5):
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    results_list = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    ) as holistic:
        for frame in frames:
            results_list.append(holistic.process(frame))
    return results_list


# ── Hand interpolation (identical to debug_extract.py) ────────────────────────

def interpolate_hands(results_list):
    from mediapipe.framework.formats import landmark_pb2

    T = len(results_list)

    def lm_to_array(landmarks):
        if landmarks is None:
            return None
        return np.array([[l.x, l.y, l.z] for l in landmarks.landmark], dtype=np.float32)

    def pose_wrist(result, side):
        if result.pose_landmarks is None:
            return None
        idx = POSE_LEFT_WRIST if side == "left" else POSE_RIGHT_WRIST
        lm = result.pose_landmarks.landmark[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    def array_to_proto(arr):
        proto = landmark_pb2.NormalizedLandmarkList()
        for row in arr:
            lm = proto.landmark.add()
            lm.x, lm.y, lm.z = float(row[0]), float(row[1]), float(row[2])
        return proto

    def interpolate_one_hand(hand_arrays, side):
        valid_idx = [i for i, a in enumerate(hand_arrays) if a is not None]
        if not valid_idx:
            return hand_arrays, 0
        filled = list(hand_arrays)
        count = 0

        def fill_frame(t, s, e):
            if e is None:
                alpha = 0.0
                hand_s = hand_e = hand_arrays[s]
            else:
                alpha = (t - s) / (e - s)
                hand_s = hand_arrays[s]
                hand_e = hand_arrays[e]
            w_s = pose_wrist(results_list[s], side)
            w_e = pose_wrist(results_list[e if e is not None else s], side)
            if w_s is not None and w_e is not None:
                rel_s = hand_s - w_s
                rel_e = hand_e - (w_e if e is not None else w_s)
                rel_t = (1 - alpha) * rel_s + alpha * rel_e
                w_t = pose_wrist(results_list[t], side)
                if w_t is not None:
                    return rel_t + w_t
                w_interp = (1 - alpha) * w_s + alpha * (w_e if e is not None else w_s)
                return rel_t + w_interp
            else:
                return (1 - alpha) * hand_s + alpha * hand_e

        for t in range(valid_idx[0]):
            filled[t] = fill_frame(t, valid_idx[0], None)
            count += 1
        for t in range(valid_idx[-1] + 1, T):
            filled[t] = fill_frame(t, valid_idx[-1], None)
            count += 1
        for k in range(len(valid_idx) - 1):
            s, e = valid_idx[k], valid_idx[k + 1]
            for t in range(s + 1, e):
                filled[t] = fill_frame(t, s, e)
                count += 1
        return filled, count

    lhand_arrays = [lm_to_array(r.left_hand_landmarks) for r in results_list]
    rhand_arrays = [lm_to_array(r.right_hand_landmarks) for r in results_list]
    lhand_filled, _ = interpolate_one_hand(lhand_arrays, "left")
    rhand_filled, _ = interpolate_one_hand(rhand_arrays, "right")

    from mediapipe.framework.formats import landmark_pb2  # noqa (already imported above)
    for i, r in enumerate(results_list):
        if lhand_arrays[i] is None and lhand_filled[i] is not None:
            r.left_hand_landmarks = array_to_proto(lhand_filled[i])
        if rhand_arrays[i] is None and rhand_filled[i] is not None:
            r.right_hand_landmarks = array_to_proto(rhand_filled[i])

    return results_list


# ── Build pose-format Pose object ─────────────────────────────────────────────

def build_pose(frames, results_list, fps):
    """
    Convert MediaPipe Holistic results to a pose_format.Pose object.

    Component layout (matches what sign-vq / pose-format expects for Holistic):
      POSE_LANDMARKS        – 33 keypoints, (x, y, z, visibility)
      FACE_LANDMARKS        – 468 keypoints, (x, y, z)
      LEFT_HAND_LANDMARKS   – 21 keypoints, (x, y, z)
      RIGHT_HAND_LANDMARKS  – 21 keypoints, (x, y, z)

    All coordinates are in normalised image space (0..1 for x,y).
    The confidence channel is 1.0 when the landmark was detected or
    interpolated, and 0.0 when the component was absent for that frame.
    """
    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody
    from pose_format.pose_header import (
        PoseHeader,
        PoseHeaderComponent,
        PoseHeaderDimensions,
    )

    T = len(frames)
    H, W = frames[0].shape[:2]

    # ── Define components ──────────────────────────────────────────────────────
    # pose-format stores points as (x, y, z) or (x, y, z, conf) inside
    # each component.  We use the same component names as the official
    # mediapipe pose_format pipeline so sign-vq can treat them identically.

    def _make_component(name, point_names):
        return PoseHeaderComponent(
            name=name,
            points=point_names,
            limbs=[],
            # colors must have the same length as limbs; empty limbs → one
            # placeholder colour so the list is never completely absent
            colors=[(128, 128, 128)], 
            # 'XYZC' → num_dims() = len("XYZC") - 1 = 3, which matches our
            # (T, 1, N, 3) data arrays.  'XYZ' would give 2 and cause the
            # write() sanity-check to raise, producing a corrupt file.
            point_format="XYZC",
        )

    # Official MediaPipe Holistic point names (abbreviated for clarity)
    POSE_POINTS = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR",
        "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX",
        "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    ]
    assert len(POSE_POINTS) == 33

    HAND_POINTS = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]
    assert len(HAND_POINTS) == 21

    FACE_POINTS = [f"f{i}" for i in range(468)]

    components = [
        _make_component("POSE_LANDMARKS",       POSE_POINTS),
        _make_component("FACE_LANDMARKS",       FACE_POINTS),
        _make_component("LEFT_HAND_LANDMARKS",  HAND_POINTS),
        _make_component("RIGHT_HAND_LANDMARKS", HAND_POINTS),
    ]

    header = PoseHeader(
        version=0.1,
        dimensions=PoseHeaderDimensions(width=W, height=H, depth=1),
        components=components,
    )

    # ── Populate data arrays ───────────────────────────────────────────────────
    # pose-format body shape: (T, people, points, dims)
    # We have 1 person, dims=3 (X,Y,Z), plus a separate confidence array.

    n_pose  = 33
    n_face  = 468
    n_hand  = 21
    n_total = n_pose + n_face + n_hand + n_hand   # 543

    data  = np.zeros((T, 1, n_total, 3), dtype=np.float32)
    conf  = np.zeros((T, 1, n_total),    dtype=np.float32)

    for t, r in enumerate(results_list):
        col = 0

        # ── Pose (33 pts) ──────────────────────────────────────────────────────
        if r.pose_landmarks:
            for i, lm in enumerate(r.pose_landmarks.landmark):
                data[t, 0, col + i] = [lm.x, lm.y, lm.z]
                conf[t, 0, col + i] = float(lm.visibility)
        col += n_pose

        # ── Face (468 pts) ─────────────────────────────────────────────────────
        if r.face_landmarks:
            for i, lm in enumerate(r.face_landmarks.landmark):
                data[t, 0, col + i] = [lm.x, lm.y, lm.z]
                conf[t, 0, col + i] = 1.0
        col += n_face

        # ── Left hand (21 pts) ─────────────────────────────────────────────────
        if r.left_hand_landmarks:
            for i, lm in enumerate(r.left_hand_landmarks.landmark):
                data[t, 0, col + i] = [lm.x, lm.y, lm.z]
                conf[t, 0, col + i] = 1.0
        col += n_hand

        # ── Right hand (21 pts) ────────────────────────────────────────────────
        if r.right_hand_landmarks:
            for i, lm in enumerate(r.right_hand_landmarks.landmark):
                data[t, 0, col + i] = [lm.x, lm.y, lm.z]
                conf[t, 0, col + i] = 1.0
        col += n_hand

    # Mask: True = data is missing (masked arrays convention)
    mask = (conf == 0.0)

    import numpy.ma as ma
    masked_data = ma.MaskedArray(data, mask=np.stack([mask] * 3, axis=-1))

    body = NumPyPoseBody(fps=float(fps), data=masked_data, confidence=conf)
    return Pose(header=header, body=body)


# ── Single-file pipeline ───────────────────────────────────────────────────────

def process_gif(gif_path: str, out_path: str, detection_conf=0.5, tracking_conf=0.5):
    """Full pipeline for one GIF.  Returns (gif_path, ok, message)."""
    try:
        frames, fps = load_gif(gif_path)
        results = extract_all(frames, detection_conf, tracking_conf)
        results = interpolate_hands(results)
        pose = build_pose(frames, results, fps)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            pose.write(f)

        # Quick stats
        T = len(results)
        body_ok  = sum(1 for r in results if r.pose_landmarks)
        lhand_ok = sum(1 for r in results if r.left_hand_landmarks)
        rhand_ok = sum(1 for r in results if r.right_hand_landmarks)
        msg = (f"{T}fr  body={body_ok}/{T}  "
               f"L={lhand_ok}/{T}  R={rhand_ok}/{T}")
        return gif_path, True, msg

    except Exception:
        return gif_path, False, traceback.format_exc()


# ── Worker entry point (for multiprocessing) ──────────────────────────────────

def _worker(args):
    gif_path, out_path, det_conf, trk_conf = args
    return process_gif(gif_path, out_path, det_conf, trk_conf)


# ── Main batch loop ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert a folder of GIFs to .pose files for sign-vq"
    )
    parser.add_argument("--input_dir",   required=True,
                        help="Directory containing .gif files (searched recursively)")
    parser.add_argument("--output_dir",  default=None,
                        help="Where to write .pose files. "
                             "Defaults to <input_dir>/poses/")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Number of parallel processes (default 1). "
                             "MediaPipe loads a model per process; keep ≤ CPU cores.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip GIFs that already have a .pose output")
    parser.add_argument("--detection_conf", type=float, default=0.5,
                        help="MediaPipe min_detection_confidence (default 0.5)")
    parser.add_argument("--tracking_conf",  type=float, default=0.5,
                        help="MediaPipe min_tracking_confidence (default 0.5)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "poses"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all .gif files
    gif_files = sorted(input_dir.rglob("*.gif"))
    if not gif_files:
        print(f"No .gif files found under {input_dir}")
        sys.exit(0)

    # Build (gif_path, out_path) pairs
    tasks = []
    for gif in gif_files:
        # Preserve subdirectory structure relative to input_dir
        rel = gif.relative_to(input_dir)
        out = (output_dir / rel).with_suffix(".pose")
        if args.skip_existing and out.exists():
            continue
        tasks.append((str(gif), str(out), args.detection_conf, args.tracking_conf))

    skipped = len(gif_files) - len(tasks)
    print(f"Found {len(gif_files)} GIF(s)  |  "
          f"{skipped} skipped (already exist)  |  "
          f"{len(tasks)} to process")
    print(f"Output dir : {output_dir}")
    print(f"Workers    : {args.workers}")
    print()

    ok_count = err_count = 0
    t0 = time.time()

    if args.workers <= 1:
        # Single-process: friendlier for debugging
        for i, task in enumerate(tasks, 1):
            gif_path, out_path, det_conf, trk_conf = task
            print(f"[{i}/{len(tasks)}] {Path(gif_path).name} … ", end="", flush=True)
            _, ok, msg = process_gif(gif_path, out_path, det_conf, trk_conf)
            if ok:
                ok_count += 1
                print(f"✓  {msg}")
            else:
                err_count += 1
                print(f"✗  ERROR")
                print(msg)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futures):
                done += 1
                gif_path, ok, msg = fut.result()
                name = Path(gif_path).name
                if ok:
                    ok_count += 1
                    print(f"  [{done}/{len(tasks)}] ✓ {name}  {msg}")
                else:
                    err_count += 1
                    print(f"  [{done}/{len(tasks)}] ✗ {name}  ERROR")
                    print(msg)

    elapsed = time.time() - t0
    print()
    print(f"── Done ───────────────────────────────────────")
    print(f"  OK      : {ok_count}")
    print(f"  Errors  : {err_count}")
    print(f"  Time    : {elapsed:.1f}s  ({elapsed/max(len(tasks),1):.1f}s/gif)")
    print(f"  Output  : {output_dir}")
    print()
    print("Next steps:")
    print(f"  python -m sign_vq.data.zip_dataset --dir={output_dir} --out=normalized.zip")
    print(f"  python -m sign_vq.train --data-path=normalized.zip")


if __name__ == "__main__":
    main()