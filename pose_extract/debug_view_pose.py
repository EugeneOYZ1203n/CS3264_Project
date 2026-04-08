"""
Debug Pose Extractor
--------------------
pip install mediapipe Pillow numpy matplotlib

Processes a single GIF, runs MediaPipe Holistic frame by frame,
and displays each frame with the detected keypoints overlaid.
This lets you verify extraction is working before running the full pipeline.

Usage:
    python debug_extract.py output/gifs/Abuse.gif
    python debug_extract.py output/gifs/Abuse.gif --fps 5   # slow down playback
    python debug_extract.py output/gifs/Abuse.gif --save    # save overlaid frames as GIF
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp


# ── MediaPipe drawing connections ─────────────────────────────────────────────

mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles


# ── GIF loading ───────────────────────────────────────────────────────────────

def load_gif(path: str) -> tuple[list[np.ndarray], float]:
    gif = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(np.array(gif.copy().convert("RGB")))
            durations.append(gif.info.get("duration", 67))  # default ~15fps
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    avg_ms = sum(durations) / len(durations)
    return frames, 1000.0 / avg_ms
    


# ── Run holistic on all frames ────────────────────────────────────────────────

def extract_all(frames: list[np.ndarray]) -> list:
    results_list = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for i, frame in enumerate(frames):
            r = holistic.process(frame)
            results_list.append(r)
            # Print per-frame detection status
            pose_ok  = "✓" if r.pose_landmarks  else "✗"
            lhand_ok = "✓" if r.left_hand_landmarks  else "✗"
            rhand_ok = "✓" if r.right_hand_landmarks else "✗"
            print(f"  Frame {i+1:3d}/{len(frames)}  "
                  f"body={pose_ok}  left_hand={lhand_ok}  right_hand={rhand_ok}")
    return results_list


# ── Hand interpolation ───────────────────────────────────────────────────────

# MediaPipe Pose wrist indices (used as the anchor point for each hand)
POSE_LEFT_WRIST  = 15
POSE_RIGHT_WRIST = 16

def interpolate_hands(results_list: list) -> list:
    """
    Arm-tracked hand interpolation.

    Strategy for each missing hand frame:
      1. Find the nearest detected frames on either side (or edge-fill).
      2. Compute the hand pose *relative to the wrist* at those anchor frames
         (hand landmarks - wrist position, in normalised image coords).
      3. Linearly interpolate the relative hand shape between the two anchors.
      4. Re-anchor the interpolated shape to the *current frame's wrist position*
         so the hand moves with the arm even when detection drops out.

    This means:
      - Hand shape smoothly transitions between the two detected poses.
      - Hand position follows the arm's actual motion in each frame.
      - If pose (body) is also missing for a gap frame, falls back to
        pure linear interpolation of absolute coords (no wrist anchoring).
    """
    from mediapipe.framework.formats import landmark_pb2

    T = len(results_list)

    def lm_to_array(landmarks):
        if landmarks is None:
            return None
        return np.array([[l.x, l.y, l.z] for l in landmarks.landmark],
                        dtype=np.float32)  # (21, 3)

    def pose_wrist(result, side: str) -> np.ndarray | None:
        """Return (3,) wrist position from pose landmarks, or None."""
        if result.pose_landmarks is None:
            return None
        idx = POSE_LEFT_WRIST if side == "left" else POSE_RIGHT_WRIST
        lm  = result.pose_landmarks.landmark[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    def array_to_proto(arr: np.ndarray):
        proto = landmark_pb2.NormalizedLandmarkList()
        for row in arr:
            lm = proto.landmark.add()
            lm.x, lm.y, lm.z = float(row[0]), float(row[1]), float(row[2])
        return proto

    def interpolate_one_hand(hand_arrays, side: str):
        """
        hand_arrays: list of (21,3) or None, length T.
        Returns filled list and count of frames filled.
        """
        valid_idx = [i for i, a in enumerate(hand_arrays) if a is not None]
        if not valid_idx:
            return hand_arrays, 0

        filled = list(hand_arrays)
        count  = 0

        def fill_frame(t: int, s: int, e: int | None):
            """
            Fill frame t by interpolating between anchor frames s and e.
            If e is None, just hold s (edge fill).
            Shifts the interpolated hand to follow the wrist at frame t.
            """
            if e is None:
                alpha     = 0.0
                hand_s    = hand_arrays[s]
                hand_e    = hand_arrays[s]
            else:
                alpha     = (t - s) / (e - s)
                hand_s    = hand_arrays[s]
                hand_e    = hand_arrays[e]

            # Wrist positions at the anchor frames
            w_s = pose_wrist(results_list[s], side)
            w_e = pose_wrist(results_list[e if e is not None else s], side)

            if w_s is not None and w_e is not None:
                # Compute hand shape relative to wrist at each anchor
                rel_s = hand_s - w_s   # (21,3) shape relative to wrist
                rel_e = hand_e - (w_e if e is not None else w_s)

                # Interpolate the relative shape
                rel_t = (1 - alpha) * rel_s + alpha * rel_e

                # Re-anchor to the current frame's wrist
                w_t = pose_wrist(results_list[t], side)
                if w_t is not None:
                    return rel_t + w_t
                # Fallback: use linearly interpolated wrist position
                w_interp = (1 - alpha) * w_s + alpha * (w_e if e is not None else w_s)
                return rel_t + w_interp
            else:
                # No wrist available — plain linear interpolation
                return (1 - alpha) * hand_s + alpha * hand_e

        # Edge fill before first detection
        for t in range(valid_idx[0]):
            filled[t] = fill_frame(t, valid_idx[0], None)
            count += 1

        # Edge fill after last detection
        for t in range(valid_idx[-1] + 1, T):
            filled[t] = fill_frame(t, valid_idx[-1], None)
            count += 1

        # Interpolate across internal gaps
        for k in range(len(valid_idx) - 1):
            s, e = valid_idx[k], valid_idx[k + 1]
            for t in range(s + 1, e):
                filled[t] = fill_frame(t, s, e)
                count += 1

        return filled, count

    lhand_arrays = [lm_to_array(r.left_hand_landmarks)  for r in results_list]
    rhand_arrays = [lm_to_array(r.right_hand_landmarks) for r in results_list]

    lhand_filled, l_count = interpolate_one_hand(lhand_arrays, "left")
    rhand_filled, r_count = interpolate_one_hand(rhand_arrays, "right")

    print(f"  Interpolated: left_hand={l_count} frames  right_hand={r_count} frames")

    # Write back only the frames that were missing
    for i, r in enumerate(results_list):
        if lhand_arrays[i] is None and lhand_filled[i] is not None:
            r.left_hand_landmarks = array_to_proto(lhand_filled[i])
        if rhand_arrays[i] is None and rhand_filled[i] is not None:
            r.right_hand_landmarks = array_to_proto(rhand_filled[i])

    return results_list


# ── Draw landmarks onto a frame ───────────────────────────────────────────────

def draw_landmarks(frame_rgb: np.ndarray, results) -> np.ndarray:
    """Draw all detected landmarks on a copy of the frame using MediaPipe utils."""
    img = frame_rgb.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(66, 195, 244), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(66, 195, 244), thickness=2),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(105, 240, 174), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(105, 240, 174), thickness=2),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 110, 110), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 110, 110), thickness=2),
        )

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(150, 150, 150), thickness=1),
        )

    return img


# ── Skeleton connections (for normalised panel) ───────────────────────────────

BODY_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
]
def _hand_conns(o):
    c = [(o, o+1),(o, o+5),(o, o+9),(o, o+13),(o, o+17)]
    for b in [1,5,9,13,17]:
        for j in range(3):
            c.append((o+b+j, o+b+j+1))
    return c
LHAND_CONNS = _hand_conns(0)
RHAND_CONNS = _hand_conns(0)


def extract_normalised(results, frame_w: int, frame_h: int) -> dict:
    """
    Pull raw x,y pixel coords from MediaPipe results and apply
    shoulder-midpoint + shoulder-width normalisation.
    Returns dict with keys 'pose', 'left_hand', 'right_hand' →
    each a (N,2) float32 array in normalised space, or None if not detected.
    Normalised space: origin = shoulder midpoint, scale = shoulder width.
    x roughly in [-1, 1], y roughly in [-1, 1] (upright, y+ = up).
    """
    def lm_to_px(landmarks, n):
        if landmarks is None:
            return None
        return np.array(
            [[l.x * frame_w, l.y * frame_h] for l in landmarks.landmark],
            dtype=np.float32,
        )

    pose_px   = lm_to_px(results.pose_landmarks,       33)
    lhand_px  = lm_to_px(results.left_hand_landmarks,  21)
    rhand_px  = lm_to_px(results.right_hand_landmarks, 21)

    if pose_px is None:
        return {"pose": None, "left_hand": None, "right_hand": None,
                "origin": None, "scale": None}

    # Shoulder indices: 11=left, 12=right in MediaPipe Pose
    l_shoulder = pose_px[11]
    r_shoulder = pose_px[12]
    origin     = (l_shoulder + r_shoulder) / 2.0
    scale      = np.linalg.norm(l_shoulder - r_shoulder)
    if scale < 1e-6:
        scale = 1.0

    def normalise(pts):
        if pts is None:
            return None
        n = pts.copy()
        n -= origin
        n /= scale
        n[:, 1] *= -1   # flip y so up = positive
        return n

    return {
        "pose":       normalise(pose_px),
        "left_hand":  normalise(lhand_px),
        "right_hand": normalise(rhand_px),
        "origin":     origin,
        "scale":      scale,
    }


def draw_normalised(ax, norm: dict, result=None):
    """Draw the normalised skeleton on a matplotlib axes (no image background)."""
    ax.cla()
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Normalised skeleton", color="#aaaaaa", fontsize=10)

    pose  = norm["pose"]
    lhand = norm["left_hand"]
    rhand = norm["right_hand"]

    STYLE = {"body": "#4fc3f7", "lhand": "#69f0ae", "rhand": "#ff6e6e"}

    def draw_conns(pts, conns, color, lw=1.5):
        if pts is None:
            return
        for a, b in conns:
            if a < len(pts) and b < len(pts):
                ax.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]],
                        color=color, lw=lw, alpha=0.85, solid_capstyle="round")

    def scatter(pts, color, s):
        if pts is not None and len(pts):
            ax.scatter(pts[:,0], pts[:,1], s=s, c=color, zorder=5, linewidths=0)

    draw_conns(pose,  BODY_CONNECTIONS, STYLE["body"],  lw=2.0)
    draw_conns(lhand, LHAND_CONNS,      STYLE["lhand"], lw=1.4)
    draw_conns(rhand, RHAND_CONNS,      STYLE["rhand"], lw=1.4)

    scatter(pose,  STYLE["body"],  20)
    scatter(lhand, STYLE["lhand"], 12)
    scatter(rhand, STYLE["rhand"], 12)

    if pose is not None:
        ls, rs = pose[11], pose[12]
        ax.plot([ls[0], rs[0]], [ls[1], rs[1]],
                color="white", lw=1, alpha=0.3, linestyle="--")
        ax.scatter([0], [0], s=30, c="white", alpha=0.4, zorder=6, marker="+")

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-3.0, 2.0)


# ── Animate ───────────────────────────────────────────────────────────────────

def run_viewer(frames, results_list, fps, title, save_path=None):
    T    = len(frames)
    h, w = frames[0].shape[:2]

    print("  Pre-computing pre-interpolation overlays …")
    overlaid_raw = [draw_landmarks(frames[i], results_list[i]) for i in range(T)]

    print("  Interpolating missing hand frames …")
    results_interp = interpolate_hands(results_list)

    print("  Pre-computing post-interpolation overlays …")
    overlaid_interp = [draw_landmarks(frames[i], results_interp[i]) for i in range(T)]

    print("  Pre-computing normalised poses …")
    norm_data = [extract_normalised(results_interp[i], w, h) for i in range(T)]

    fig = plt.figure(figsize=(22, 6), facecolor="#111111")
    fig.suptitle(os.path.basename(title), color="white", fontsize=12)

    ax_raw   = fig.add_subplot(1, 4, 1)
    ax_raw_ov = fig.add_subplot(1, 4, 2)
    ax_interp = fig.add_subplot(1, 4, 3)
    ax_nrm   = fig.add_subplot(1, 4, 4)

    for ax, label in [
        (ax_raw,    "Original"),
        (ax_raw_ov, "Extracted (no interpolation)"),
        (ax_interp, "Extracted (with interpolation)"),
        (ax_nrm,    "Normalised skeleton"),
    ]:
        ax.axis("off")
        ax.set_facecolor("#111111")
        ax.set_title(label, color="#aaaaaa", fontsize=10)

    im_raw    = ax_raw.imshow(frames[0])
    im_raw_ov = ax_raw_ov.imshow(overlaid_raw[0])
    im_interp = ax_interp.imshow(overlaid_interp[0])
    draw_normalised(ax_nrm, norm_data[0])

    status_txt = fig.text(
        0.5, 0.01, "", ha="center", color="white", fontsize=10,
        fontfamily="monospace",
    )

    patches = [
        mpatches.Patch(color="#4fc3f7", label="Body"),
        mpatches.Patch(color="#69f0ae", label="Left hand"),
        mpatches.Patch(color="#ff6e6e", label="Right hand"),
        mpatches.Patch(color="#969696", label="Face"),
    ]
    fig.legend(handles=patches, loc="lower right",
               facecolor="#222222", labelcolor="white", fontsize=9)

    def update(t):
        im_raw.set_data(frames[t])
        im_raw_ov.set_data(overlaid_raw[t])
        im_interp.set_data(overlaid_interp[t])
        draw_normalised(ax_nrm, norm_data[t])

        r  = results_list[t]    # original (pre-interp) for status
        nd = norm_data[t]
        scale_str = f"scale={nd['scale']:.1f}px" if nd["scale"] else "scale=n/a"
        parts = [
            f"body={'✓' if r.pose_landmarks else '✗'}",
            f"L_hand={'✓' if r.left_hand_landmarks else '✗'}",
            f"R_hand={'✓' if r.right_hand_landmarks else '✗'}",
            f"face={'✓' if r.face_landmarks else '✗'}",
            f"  {scale_str}   frame {t+1}/{T}",
        ]
        status_txt.set_text("   ".join(parts))
        return [im_raw, im_raw_ov, im_interp, status_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T,
        interval=int(1000 / fps),
        blit=False, repeat=True,
    )

    if save_path:
        print(f"Saving animation → {save_path} …")
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        ani.save(save_path, writer=writer, fps=fps,
                 savefig_kwargs={"facecolor": "#111111"})
        print("Saved.")
    else:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(results_list, fps):
    T = len(results_list)
    pose_det  = sum(1 for r in results_list if r.pose_landmarks)
    lhand_det = sum(1 for r in results_list if r.left_hand_landmarks)
    rhand_det = sum(1 for r in results_list if r.right_hand_landmarks)
    face_det  = sum(1 for r in results_list if r.face_landmarks)
    print(f"\n── Summary ──────────────────────────────")
    print(f"  Total frames : {T}  ({T/fps:.1f}s @ {fps:.1f} fps)")
    print(f"  Body         : {pose_det}/{T}  ({100*pose_det/T:.0f}%)")
    print(f"  Left hand    : {lhand_det}/{T}  ({100*lhand_det/T:.0f}%)")
    print(f"  Right hand   : {rhand_det}/{T}  ({100*rhand_det/T:.0f}%)")
    print(f"  Face         : {face_det}/{T}  ({100*face_det/T:.0f}%)")
    print(f"─────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Debug pose extraction on a single GIF")
    parser.add_argument("gif_file",         help="Path to the .gif file")
    parser.add_argument("--fps",  type=float, default=None,
                        help="Override playback FPS (default: use GIF metadata)")
    parser.add_argument("--save", default=None,
                        help="Save overlaid animation to .gif or .mp4")
    args = parser.parse_args()

    if not os.path.exists(args.gif_file):
        print(f"File not found: {args.gif_file}")
        sys.exit(1)

    print(f"Loading {args.gif_file} …")
    frames, fps = load_gif(args.gif_file)
    if args.fps:
        fps = args.fps
    h, w = frames[0].shape[:2]
    print(f"  {len(frames)} frames  |  {w}×{h}px  |  {fps:.1f} fps")

    print(f"\nRunning MediaPipe Holistic …")
    results_list = extract_all(frames)
    print_summary(results_list, fps)  

    print("\nLaunching viewer …")
    run_viewer(frames, results_list, fps,
               title=args.gif_file,
               save_path=args.save)


if __name__ == "__main__":
    main()