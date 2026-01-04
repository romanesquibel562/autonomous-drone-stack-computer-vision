# src/main.py
from __future__ import annotations

import time
import random
import math
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.types import EgoState, Decision, Track, CandidateScore
from src.simulation.world import World
from src.perception.sim_perception import CameraModel, simulate_detections
from src.perception.hailo_detector import detect_from_image_path
from src.state_estimation.tracker import NearestNeighborTracker
from src.decision.policy import policy_step, PolicyConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cv_drone autonomy loop (sim or Hailo perception)")

    p.add_argument(
        "--perception",
        choices=["sim", "hailo"],
        default="sim",
        help="Perception backend: simulated detections or Hailo YOLO detections",
    )
    p.add_argument(
        "--image",
        default="/tmp/frame.jpg",
        help="Image path used by Hailo backend (a single frame file)",
    )
    p.add_argument(
        "--capture",
        action="store_true",
        help="If set (hailo mode), capture a new frame each tick into --image using rpicam-still",
    )
    p.add_argument(
        "--project-frame",
        default="artifacts/frame_latest.jpg",
        help="If (hailo + capture), copy the captured frame into the project at this path so you can open it in VS Code",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="If (hailo + capture), copy frame into project every N ticks (default: 1 = every tick)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Hailo confidence threshold",
    )
    p.add_argument(
        "--hef",
        default="/usr/share/hailo-models/yolov8s_h8l.hef",
        help="Path to HEF model for Hailo backend",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of loop iterations",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Timestep in seconds",
    )

    return p.parse_args()


def _capture_frame(
    image_path: str,
    *,
    project_copy_path: str | None = None,
    step: int | None = None,
    save_every: int = 1,
) -> None:
    """
    Capture one frame from the Pi camera to image_path using rpicam-still.
    Optionally copy it into the project (artifacts/...) so VS Code can preview it easily.

    - project_copy_path: e.g. "artifacts/frame_latest.jpg"
    - save_every: only copy every N steps to reduce disk IO (still captures every step)
    """
    image_path = str(image_path)
    cmd = ["rpicam-still", "-n", "-t", "1", "-o", image_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if project_copy_path:
        # Only copy into the repo every N ticks (default: every tick)
        if save_every <= 1 or step is None or (step % save_every == 0):
            dst = Path(project_copy_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(image_path).read_bytes())


def step_dynamics(ego: EgoState, commands: dict, dt: float) -> EgoState:
    """
    v1 ego dynamics:
    - heading integrates yaw_rate
    - speed integrates throttle
    - position integrates heading + speed (x,y plane)
    """
    x, y, z = ego.position_xyz

    yaw_rate = float(commands.get("yaw_rate", 0.0))
    heading = ego.heading_rad + yaw_rate * dt
    heading = (heading + math.pi) % (2.0 * math.pi) - math.pi

    throttle = float(commands.get("throttle", 0.0))
    speed = max(0.0, ego.speed_m_s + (throttle - 0.3) * 0.1)

    x = x + math.cos(heading) * speed * dt
    y = y + math.sin(heading) * speed * dt

    return EgoState(position_xyz=(x, y, z), heading_rad=heading, speed_m_s=speed)


# -----------------------------
# Logging helpers
# -----------------------------
def _fmt_dist(d: Optional[float]) -> str:
    return "NA" if d is None else f"{d:.2f}m"


def _fmt_err(ex: float, ey: float) -> str:
    return f"({ex:+6.1f},{ey:+6.1f})"


def _fmt_candidates(cands: list[CandidateScore]) -> str:
    parts = []
    for c in cands:
        parts.append(f"id{c.track_id}:{c.score:.3f} {c.label}({c.confidence:.2f})")
    return " | ".join(parts) if parts else "none"


def _find_target(tracks: list[Track], target_id: Optional[int]) -> Optional[Track]:
    if target_id is None:
        return None
    return next((t for t in tracks if t.track_id == target_id), None)


def _print_compact_line(
    t: float,
    ego: EgoState,
    decision: Decision,
    tracks: list[Track],
    img_cx: float,
    img_cy: float,
) -> None:
    yaw = float(decision.commands.get("yaw_rate", 0.0))
    thr = float(decision.commands.get("throttle", 0.0))

    target = _find_target(tracks, decision.target_id)
    if target is None:
        tgt_str = "tgt=None"
        err_str = "err=(  NA ,  NA )"
        dist_str = "dist=NA"
        conf_str = "conf=NA"
        lab_str = "lab=NA"
        n_str = f"n={len(tracks):2d}"
    else:
        tx, ty = target.center_px
        err_x, err_y = (tx - img_cx), (ty - img_cy)
        tgt_str = f"tgt=id{target.track_id}"
        lab_str = f"lab={target.label}"
        conf_str = f"conf={target.confidence:.2f}"
        err_str = f"err={_fmt_err(err_x, err_y)}"
        dist_str = f"dist={_fmt_dist(target.distance_m)}"
        n_str = f"n={len(tracks):2d}"

    print(
        f"t={t:4.1f}s | {decision.mode:5} | "
        f"pos=({ego.position_xyz[0]:6.2f},{ego.position_xyz[1]:6.2f}) "
        f"hdg={ego.heading_rad:+.2f} spd={ego.speed_m_s:4.2f} | "
        f"yaw={yaw:+.2f} thr={thr:.2f} | {n_str} | "
        f"{tgt_str} {lab_str} {conf_str} {err_str} {dist_str} | "
        f"{decision.reason}"
    )


def _print_details(decision: Decision) -> None:
    cands = decision.candidates if decision.candidates is not None else []
    cand_str = _fmt_candidates(cands)

    dbg = decision.debug or {}
    select = dbg.get("select", {})
    chosen_id = select.get("chosen_id", None)
    topk = select.get("topk", None)
    yaw_reason = dbg.get("yaw_reason", None)

    extras = []
    if chosen_id is not None:
        extras.append(f"chosen=id{chosen_id}")
    if topk is not None:
        extras.append(f"topk={topk}")
    if yaw_reason is not None:
        extras.append(f"yaw_reason={yaw_reason}")

    extras_str = (" | " + " ".join(extras)) if extras else ""
    print(f"          candidates: {cand_str}{extras_str}")


# -----------------------------
# Event-driven logging (reduces spam)
# -----------------------------
@dataclass
class _LogState:
    last_mode: Optional[str] = None
    last_target_id: Optional[int] = None
    last_yaw: Optional[float] = None
    last_print_t: float = 0.0


def _should_log(
    t: float,
    decision: Decision,
    yaw: float,
    state: _LogState,
    *,
    yaw_eps: float = 0.05,
    heartbeat_s: float = 1.0,
) -> bool:
    mode_changed = (state.last_mode is None) or (decision.mode != state.last_mode)
    tgt_changed = (state.last_target_id is None) or (decision.target_id != state.last_target_id)

    if state.last_yaw is None:
        yaw_changed = True
    else:
        yaw_changed = abs(yaw - state.last_yaw) >= yaw_eps

    heartbeat = (t - state.last_print_t) >= heartbeat_s

    return mode_changed or tgt_changed or yaw_changed or heartbeat


def main():
    args = parse_args()

    rng = random.Random(123)
    ego = EgoState(position_xyz=(0.0, 0.0, 1.0), heading_rad=0.0, speed_m_s=1.0)

    world = World.demo_scenario(seed=7)

    # NOTE: sim camera is 640x480; Hailo detector resizes internally to 640x640.
    frame_size = (640, 480)
    camera = CameraModel(image_size=frame_size)
    img_cx, img_cy = frame_size[0] / 2.0, frame_size[1] / 2.0

    tracker = NearestNeighborTracker(match_threshold_px=60.0, max_missed_time_s=0.5)

    cfg = PolicyConfig(
        center_deadband_px=40.0,
        avoid_distance_m=2.0,
        max_yaw_rate=1.0,
        patrol_yaw_rate=0.3,
        patrol_throttle=0.35,
        track_throttle=0.40,
        avoid_throttle=0.20,
        center_sigma_px=140.0,
        switch_margin=1.15,
        preferred_hold_bonus=1.10,
        top_k_candidates=3,
    )

    dt = float(args.dt)

    PRINT_DETAILS = True
    YAW_EPS = 0.05
    HEARTBEAT_S = 1.0

    log_state = _LogState()

    print("Running loop. Ctrl+C to stop.")
    print(f"Perception backend: {args.perception}")
    if args.perception == "hailo":
        print(f"HEF: {args.hef}")
        print(f"Image: {args.image}  (capture_each_tick={args.capture})  conf={args.conf}")
        if args.capture:
            print(f"Project frame: {args.project_frame}  (save_every={args.save_every})")
    print("-" * 100)

    for step in range(int(args.steps)):
        world.step(dt)

        # -----------------------------
        # Perception
        # -----------------------------
        if args.perception == "sim":
            detections = simulate_detections(world, camera, rng)
        else:
            # Capture a fresh frame each tick (optional)
            if args.capture:
                _capture_frame(
                    args.image,
                    project_copy_path=args.project_frame,
                    step=step,
                    save_every=int(args.save_every),
                )

            # Ensure image exists at least once (fallback)
            if not Path(args.image).exists():
                _capture_frame(
                    args.image,
                    project_copy_path=args.project_frame,
                    step=step,
                    save_every=int(args.save_every),
                )

            detections = detect_from_image_path(
                args.image,
                hef_path=args.hef,
                conf_thresh=float(args.conf),
            )

        # -----------------------------
        # Tracking -> Decision -> Dynamics
        # -----------------------------
        tracks = tracker.update(detections, now_ts=world.time_s, dt=dt)

        decision: Decision = policy_step(
            ego=ego,
            tracks=tracks,
            frame_size=frame_size,
            cfg=cfg,
        )

        ego = step_dynamics(ego, decision.commands, dt)

        # -----------------------------
        # Logging
        # -----------------------------
        yaw_cmd = float(decision.commands.get("yaw_rate", 0.0))

        if _should_log(
            t=world.time_s,
            decision=decision,
            yaw=yaw_cmd,
            state=log_state,
            yaw_eps=YAW_EPS,
            heartbeat_s=HEARTBEAT_S,
        ):
            _print_compact_line(
                t=world.time_s,
                ego=ego,
                decision=decision,
                tracks=tracks,
                img_cx=img_cx,
                img_cy=img_cy,
            )
            if PRINT_DETAILS:
                _print_details(decision)

            log_state.last_mode = decision.mode
            log_state.last_target_id = decision.target_id
            log_state.last_yaw = yaw_cmd
            log_state.last_print_t = world.time_s

        time.sleep(dt)

    print("-" * 100)
    print("Run complete.")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------------
# COPY/PASTE RUN COMMANDS (run from the Pi terminal)
#
# 1) Go to project + activate venv:
#    cd ~/cv_drone
#    source .venv/bin/activate
#
# 2) SIMULATION (no camera, uses fake detections):
#    python -m src.main --perception sim
#
# 3) HAILO, SINGLE FRAME (reuse an existing image file):
#    rpicam-still -n -t 1 -o /tmp/frame.jpg
#    python -m src.main --perception hailo --image /tmp/frame.jpg --conf 0.25
#
# 4) HAILO + CAPTURE EACH TICK (updates the camera frame every loop):
#    python -m src.main --perception hailo --capture --image /tmp/frame.jpg --conf 0.25
#
# 5) HAILO + CAPTURE EACH TICK + ALSO SAVE LATEST FRAME INSIDE YOUR PROJECT:
#    # This will continuously update: cv_drone/artifacts/frame_latest.jpg

#    python -m src.main --perception hailo --capture --image /tmp/frame.jpg 
#   --project-frame artifacts/frame_latest.jpg --save-every 1 --conf 0.25
#
# After running (5), open this file in VS Code to see the latest camera frame:
#    artifacts/frame_latest.jpg
# --------------------------------------------------------------------------------------



