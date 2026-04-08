"""
view_pose.py  –  Quick pose viewer
-----------------------------------
Visualise a .pose file in a matplotlib window, or save it as a GIF/MP4.

Usage:
    python view_pose.py path/to/file.pose
    python view_pose.py path/to/file.pose --save out.gif
    python view_pose.py path/to/file.pose --fps 12 --save out.mp4
    python view_pose.py path/to/file.pose --no_face    # hide face mesh
    python view_pose.py path/to/file.pose --component POSE_LANDMARKS  # single component

Or import and call directly:
    from view_pose import view_pose
    view_pose("file.pose")
    view_pose("file.pose", save="preview.gif", fps=15, no_face=True)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


# ── MediaPipe Holistic skeleton connections ────────────────────────────────────

POSE_CONNECTIONS = [
    # face outline
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    # shoulders / torso
    (9,10),(11,12),(11,23),(12,24),(23,24),
    # left arm
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    # right arm
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    # left leg
    (23,25),(25,27),(27,29),(27,31),(29,31),
    # right leg
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

def _hand_connections():
    conns = []
    fingers = [1, 5, 9, 13, 17]
    for f in fingers:
        conns.append((0, f))          # wrist → base
        for j in range(3):
            conns.append((f+j, f+j+1))
    return conns

HAND_CONNECTIONS = _hand_connections()

FACE_CONTOUR_IDX = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109,10,
]
FACE_CONTOUR_CONNECTIONS = list(zip(FACE_CONTOUR_IDX, FACE_CONTOUR_IDX[1:]))

STYLE = {
    "POSE_LANDMARKS":       {"color": "#4fc3f7", "lw": 2.0, "ms": 18},
    "LEFT_HAND_LANDMARKS":  {"color": "#69f0ae", "lw": 1.4, "ms": 10},
    "RIGHT_HAND_LANDMARKS": {"color": "#ff6e6e", "lw": 1.4, "ms": 10},
    "FACE_LANDMARKS":       {"color": "#888888", "lw": 0.8, "ms":  0},
}


# ── Load .pose file ────────────────────────────────────────────────────────────

def load_pose(path: str):
    """Return a pose_format.Pose object."""
    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody
    with open(path, "rb") as f:
        return Pose.read(f.read(), NumPyPoseBody)


# ── Extract per-component point arrays ────────────────────────────────────────

def get_component_data(pose, component_name: str, frame: int):
    """
    Return (pts, mask) for a component at a given frame.
    pts  – (N, 3) float32  x,y,z in normalised image coords
    mask – (N,)   bool     True = landmark was missing / masked
    """
    import numpy.ma as ma

    header = pose.header
    body   = pose.body   # NumPyPoseBody

    # Find component offset in the flat point list
    offset = 0
    target_n = None
    for comp in header.components:
        n = len(comp.points)
        if comp.name == component_name:
            target_n = n
            break
        offset += n
    if target_n is None:
        return None, None

    # body.data shape: (T, people, total_points, dims)
    # body.confidence shape: (T, people, total_points)
    data_slice = body.data[frame, 0, offset:offset + target_n]  # (N, dims)
    conf_slice = body.confidence[frame, 0, offset:offset + target_n]  # (N,)

    pts = np.array(data_slice)   # might be masked array – convert to plain
    if hasattr(pts, 'filled'):
        pts = pts.filled(0.0)

    # mask: missing if confidence == 0 OR the data array entry is masked
    missing = (conf_slice == 0)
    if hasattr(data_slice, 'mask'):
        missing = missing | np.any(data_slice.mask, axis=-1)

    return pts.astype(np.float32), missing.astype(bool)


# ── Draw one frame onto axes ───────────────────────────────────────────────────

def draw_frame(ax, pose, frame: int, width: int, height: int,
               no_face=False, only_component=None):
    ax.cla()
    ax.set_facecolor("#111111")
    ax.set_aspect("equal")
    ax.axis("off")

    # Coordinate space: normalised (0–1) → pixel.  Flip Y so up = up.
    def to_px(pts):
        """(N,3) normalised → (N,2) pixel, y-flipped."""
        px = pts[:, :2].copy()
        px[:, 0] *= width
        px[:, 1] = (1.0 - px[:, 1]) * height   # flip Y
        return px

    def draw_component(name, connections):
        if only_component and name != only_component:
            return
        pts, missing = get_component_data(pose, name, frame)
        if pts is None or len(pts) == 0:
            return

        px  = to_px(pts)
        sty = STYLE.get(name, {"color": "white", "lw": 1.0, "ms": 8})
        col = sty["color"]
        lw  = sty["lw"]
        ms  = sty["ms"]

        for a, b in connections:
            if a >= len(px) or b >= len(px):
                continue
            if missing[a] or missing[b]:
                continue
            ax.plot([px[a, 0], px[b, 0]], [px[a, 1], px[b, 1]],
                    color=col, lw=lw, alpha=0.85, solid_capstyle="round", zorder=2)

        if ms > 0:
            visible = ~missing
            if visible.any():
                ax.scatter(px[visible, 0], px[visible, 1],
                           s=ms, c=col, zorder=3, linewidths=0)

    draw_component("POSE_LANDMARKS",       POSE_CONNECTIONS)
    draw_component("LEFT_HAND_LANDMARKS",  HAND_CONNECTIONS)
    draw_component("RIGHT_HAND_LANDMARKS", HAND_CONNECTIONS)
    if not no_face:
        draw_component("FACE_LANDMARKS",   FACE_CONTOUR_CONNECTIONS)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)


# ── Main viewer function ───────────────────────────────────────────────────────

def view_pose(
    path: str,
    *,
    fps: float | None = None,
    save: str | None = None,
    no_face: bool = False,
    component: str | None = None,
):
    """
    View or save a .pose file as an animation.

    Parameters
    ----------
    path      : path to the .pose file
    fps       : playback / export FPS (defaults to the FPS stored in the file)
    save      : if given, save to this path (.gif or .mp4) instead of showing
    no_face   : hide face mesh (useful when face landmarks are noisy)
    component : if set, draw only this component
                e.g. "POSE_LANDMARKS", "LEFT_HAND_LANDMARKS"
    """
    pose = load_pose(path)

    T      = pose.body.data.shape[0]
    W      = pose.header.dimensions.width
    H      = pose.header.dimensions.height
    file_fps = float(pose.body.fps)
    play_fps = fps if fps is not None else file_fps
    if play_fps <= 0:
        play_fps = 15.0

    # Figure sizing: keep aspect ratio, cap at reasonable screen size
    scale  = min(800 / max(W, 1), 600 / max(H, 1), 1.0)
    fig_w  = max(W * scale / 96, 4)
    fig_h  = max(H * scale / 96, 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#111111")
    plt.subplots_adjust(left=0, right=1, top=0.93, bottom=0.04)

    title = fig.suptitle(
        f"{Path(path).name}   frame 1/{T}   {file_fps:.1f} fps",
        color="white", fontsize=9, fontfamily="monospace",
    )

    # Legend
    patches = [
        mpatches.Patch(color=STYLE["POSE_LANDMARKS"]["color"],       label="Body"),
        mpatches.Patch(color=STYLE["LEFT_HAND_LANDMARKS"]["color"],  label="Left hand"),
        mpatches.Patch(color=STYLE["RIGHT_HAND_LANDMARKS"]["color"], label="Right hand"),
    ]
    if not no_face:
        patches.append(mpatches.Patch(color=STYLE["FACE_LANDMARKS"]["color"], label="Face"))
    fig.legend(handles=patches, loc="lower right",
               facecolor="#222222", labelcolor="white", fontsize=7,
               framealpha=0.7, borderpad=0.5)

    def update(t):
        draw_frame(ax, pose, t, W, H, no_face=no_face, only_component=component)
        title.set_text(
            f"{Path(path).name}   frame {t+1}/{T}   {file_fps:.1f} fps"
        )
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=T,
        interval=int(1000 / play_fps),
        blit=False, repeat=True,
    )

    if save:
        ext = Path(save).suffix.lower()
        writer = "pillow" if ext == ".gif" else "ffmpeg"
        print(f"Saving → {save} …")
        ani.save(save, writer=writer, fps=play_fps,
                 savefig_kwargs={"facecolor": "#111111"})
        print("Done.")
    else:
        plt.show()

    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="View a .pose file")
    parser.add_argument("pose_file", help="Path to the .pose file")
    parser.add_argument("--fps",   type=float, default=None,
                        help="Override playback FPS")
    parser.add_argument("--save",  default=None,
                        help="Save animation to .gif or .mp4 instead of showing")
    parser.add_argument("--no_face", action="store_true",
                        help="Hide face mesh")
    parser.add_argument("--component", default=None,
                        help="Draw only one component "
                             "(POSE_LANDMARKS | LEFT_HAND_LANDMARKS | "
                             "RIGHT_HAND_LANDMARKS | FACE_LANDMARKS)")
    args = parser.parse_args()

    if not os.path.exists(args.pose_file):
        print(f"File not found: {args.pose_file}")
        raise SystemExit(1)

    view_pose(
        args.pose_file,
        fps=args.fps,
        save=args.save,
        no_face=args.no_face,
        component=args.component,
    )


if __name__ == "__main__":
    main()