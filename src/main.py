# src/main.py
from __future__ import annotations

import time
import random
import math
import argparse
from typing import Optional

from src.core.types import EgoState, Decision, Track, CandidateScore
from src.simulation.world import World
from src.perception.sim_perception import CameraModel, simulate_detections
from src.state_estimation.tracker import NearestNeighborTracker
from src.decision.policy import policy_step, PolicyConfig


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
    # ex: id3:0.659 obstacle(0.93) | id2:0.442 person(0.88)
    if not cands:
        return "none"
    return " | ".join([f"id{c.track_id}:{c.score:.3f} {c.label}({c.confidence:.2f})" for c in cands])


def _find_target(tracks: list[Track], target_id: Optional[int]) -> Optional[Track]:
    if target_id is None:
        return None
    return next((t for t in tracks if t.track_id == target_id), None)


def _select_debug_fields(decision: Decision) -> tuple[bool, bool, Optional[int], Optional[str]]:
    """
    Returns:
      switched: whether selection switched
      kept: whether selection kept
      chosen_id: chosen track_id
      topk: compact top-k scores string
    """
    dbg = decision.debug or {}
    sel = dbg.get("select", {}) if isinstance(dbg.get("select", {}), dict) else {}

    switched = bool(sel.get("switched", False))
    kept = bool(sel.get("kept", False))
    chosen_id = sel.get("chosen_id", None)
    topk = sel.get("topk", None)
    return switched, kept, chosen_id, topk


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
        tgt = "tgt=None"
        lab = "lab=NA"
        conf = "conf=NA"
        err = "err=(  NA ,  NA )"
        dist = "dist=NA"
    else:
        tx, ty = target.center_px
        err_x, err_y = (tx - img_cx), (ty - img_cy)
        tgt = f"tgt=id{target.track_id}"
        lab = f"lab={target.label}"
        conf = f"conf={target.confidence:.2f}"
        err = f"err={_fmt_err(err_x, err_y)}"
        dist = f"dist={_fmt_dist(target.distance_m)}"

    # Keep the tick line short. Reason is intentionally short in policy.py now.
    print(
        f"t={t:4.1f}s | {decision.mode:5} | "
        f"pos=({ego.position_xyz[0]:6.2f},{ego.position_xyz[1]:6.2f}) "
        f"hdg={ego.heading_rad:+.2f} spd={ego.speed_m_s:4.2f} | "
        f"yaw={yaw:+.2f} thr={thr:.2f} | "
        f"n={len(tracks):2d} | "
        f"{tgt} {lab} {conf} {err} {dist} | "
        f"{decision.reason}"
    )


def _print_details(decision: Decision) -> None:
    cands = decision.candidates if decision.candidates is not None else []
    cand_str = _fmt_candidates(cands)

    dbg = decision.debug or {}
    sel = dbg.get("select", {}) if isinstance(dbg.get("select", {}), dict) else {}

    yaw_reason = dbg.get("yaw_reason", None)
    err_x = dbg.get("err_x_px", None)
    dist_m = dbg.get("dist_m", None)

    bits = []
    if "select_reason" in sel:
        bits.append(str(sel["select_reason"]))
    if "topk" in sel:
        bits.append(f"topk={sel['topk']}")
    if yaw_reason is not None:
        bits.append(f"yaw={yaw_reason}")
    if err_x is not None:
        bits.append(f"err_x={float(err_x):+.1f}px")
    if dist_m is not None:
        bits.append(f"dist={dist_m:.2f}m")

    tail = " | ".join(bits) if bits else ""
    print(f"          candidates: {cand_str}")
    if tail:
        print(f"          debug: {tail}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200, help="simulation steps")
    p.add_argument("--dt", type=float, default=0.1, help="time step seconds")
    p.add_argument("--print-every", type=int, default=1, help="print compact tick line every N steps")
    p.add_argument("--verbose", action="store_true", help="print details on events (switch/mode change/avoid)")
    p.add_argument("--details-every", type=int, default=0, help="also print details every N steps (0=never)")
    return p.parse_args()


def main():
    args = parse_args()

    rng = random.Random(123)

    ego = EgoState(position_xyz=(0.0, 0.0, 1.0), heading_rad=0.0, speed_m_s=1.0)
    world = World.demo_scenario(seed=7)

    frame_size = (640, 480)
    camera = CameraModel(image_size=frame_size)
    img_cx, img_cy = frame_size[0] / 2.0, frame_size[1] / 2.0

    tracker = NearestNeighborTracker(match_threshold_px=60.0, max_missed_time_s=0.5)

    # Tune behavior here (no editing policy.py needed)
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

    print("Running simulation. Ctrl+C to stop.")
    print("-" * 110)

    last_mode: Optional[str] = None
    last_target_id: Optional[int] = None

    for step in range(args.steps):
        world.step(args.dt)
        detections = simulate_detections(world, camera, rng)
        tracks = tracker.update(detections, now_ts=world.time_s, dt=args.dt)

        decision: Decision = policy_step(
            ego=ego,
            tracks=tracks,
            frame_size=frame_size,
            cfg=cfg,
        )

        ego = step_dynamics(ego, decision.commands, args.dt)

        # Compact line frequency control
        if step % max(1, args.print_every) == 0:
            _print_compact_line(
                t=world.time_s,
                ego=ego,
                decision=decision,
                tracks=tracks,
                img_cx=img_cx,
                img_cy=img_cy,
            )

        # Event-driven details:
        switched, kept, chosen_id, topk = _select_debug_fields(decision)
        mode_changed = (last_mode is not None and decision.mode != last_mode)
        target_changed = (last_target_id is not None and decision.target_id != last_target_id)

        should_details = False
        if args.verbose:
            # print details on meaningful events
            if decision.mode == "AVOID" or switched or mode_changed:
                should_details = True

        # optional periodic details
        if args.details_every and args.details_every > 0 and step % args.details_every == 0:
            should_details = True

        if should_details:
            _print_details(decision)

        last_mode = decision.mode
        last_target_id = decision.target_id

        time.sleep(args.dt)

    print("-" * 110)
    print("Simulation complete.")


if __name__ == "__main__":
    main()


# Execute:
# python -m src.main



