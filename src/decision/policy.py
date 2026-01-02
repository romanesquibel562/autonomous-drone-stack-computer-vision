# src/decision/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math

from src.core.types import Track, EgoState, Decision, CandidateScore


@dataclass(frozen=True)
class PolicyConfig:
    center_deadband_px: float = 40.0

    avoid_distance_m: float = 2.0
    track_distance_m: float = 50.0

    max_yaw_rate: float = 1.0
    patrol_yaw_rate: float = 0.3

    patrol_throttle: float = 0.35
    track_throttle: float = 0.40
    hold_throttle: float = 0.0
    avoid_throttle: float = 0.20

    center_sigma_px: float = 140.0
    switch_margin: float = 1.15
    preferred_hold_bonus: float = 1.10

    top_k_candidates: int = 3


LABEL_WEIGHTS: Dict[str, float] = {
    "person": 1.80,
    "vehicle": 1.40,
    "bicycle": 1.20,
    "animal": 1.05,
    "obstacle": 0.65,
}
PREFERRED_LABELS = {"person", "vehicle"}
DEFAULT_LABEL_WEIGHT = 1.00


def _image_center(frame_size: Tuple[int, int]) -> Tuple[float, float]:
    w, h = frame_size
    return (w / 2.0, h / 2.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _center_factor(dx: float, dy: float, sigma_px: float) -> float:
    r2 = dx * dx + dy * dy
    sigma = max(1.0, sigma_px)
    return math.exp(-r2 / (2.0 * sigma * sigma))


def _distance_factor(distance_m: Optional[float]) -> float:
    if distance_m is None:
        return 1.0
    d = _clamp(float(distance_m), 0.5, 200.0)
    return 1.0 + (1.0 / d)


def _age_bonus(age_frames: int) -> float:
    return 1.0 + _clamp(float(age_frames), 0.0, 30.0) * 0.003


def _score_track(
    track: Track,
    img_cx: float,
    img_cy: float,
    cfg: PolicyConfig,
) -> CandidateScore:
    x, y = track.center_px
    dx = x - img_cx
    dy = y - img_cy

    center = _center_factor(dx, dy, cfg.center_sigma_px)
    conf = float(track.confidence)
    label_w = float(LABEL_WEIGHTS.get(track.label, DEFAULT_LABEL_WEIGHT))
    age_b = _age_bonus(track.age_frames)
    dist_f = _distance_factor(track.distance_m)

    score = conf * label_w * center * age_b * dist_f

    return CandidateScore(
        track_id=track.track_id,
        label=track.label,
        score=score,
        confidence=conf,
        center_factor=center,
        age_frames=track.age_frames,
        distance_m=track.distance_m,
        extra={
            "label_w": label_w,
            "dx": dx,
            "dy": dy,
            "dist_factor": dist_f,
            "age_bonus": age_b,
        },
    )


def _select_target_with_hysteresis(
    tracks: List[Track],
    img_cx: float,
    img_cy: float,
    cfg: PolicyConfig,
    current_target_id: Optional[int],
) -> Tuple[Track, str, List[CandidateScore], Dict[str, object]]:
    candidates: List[CandidateScore] = []
    by_id: Dict[int, CandidateScore] = {}

    for tr in tracks:
        cand = _score_track(tr, img_cx, img_cy, cfg)
        candidates.append(cand)
        by_id[tr.track_id] = cand

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]

    top_k = candidates[: max(1, cfg.top_k_candidates)]
    top_str = " | ".join([f"id{c.track_id}:{c.score:.3f}" for c in top_k])

    kept = False
    switched = False

    if current_target_id is not None and current_target_id in by_id:
        cur = by_id[current_target_id]
        hold_bonus = cfg.preferred_hold_bonus if cur.label in PREFERRED_LABELS else 1.0

        if cur.score * cfg.switch_margin * hold_bonus >= best.score:
            kept = True
            reason = f"keep id{cur.track_id}"
            chosen_id = cur.track_id
            chosen = next(t for t in tracks if t.track_id == cur.track_id)
        else:
            switched = True
            reason = f"switch -> id{best.track_id}"
            chosen_id = best.track_id
            chosen = next(t for t in tracks if t.track_id == best.track_id)
    else:
        switched = True
        reason = f"switch -> id{best.track_id}"
        chosen_id = best.track_id
        chosen = next(t for t in tracks if t.track_id == best.track_id)

    debug: Dict[str, object] = {
        "select_reason": reason,
        "chosen_id": chosen_id,
        "topk": top_str,
        "kept": kept,
        "switched": switched,
        "best_id": best.track_id,
        "best_score": best.score,
    }

    return chosen, reason, top_k, debug


_LAST_TARGET_ID: Optional[int] = None


def policy_step(
    ego: EgoState,
    tracks: List[Track],
    frame_size: Tuple[int, int],
    cfg: PolicyConfig = PolicyConfig(),
) -> Decision:
    img_cx, img_cy = _image_center(frame_size)

    global _LAST_TARGET_ID

    if not tracks:
        _LAST_TARGET_ID = None
        return Decision(
            mode="PATROL",
            target_id=None,
            commands={"yaw_rate": cfg.patrol_yaw_rate, "throttle": cfg.patrol_throttle},
            reason="PATROL",
            debug={"mode": "PATROL", "n_tracks": 0},
            candidates=None,
        )

    chosen, select_reason, top_candidates, select_debug = _select_target_with_hysteresis(
        tracks=tracks,
        img_cx=img_cx,
        img_cy=img_cy,
        cfg=cfg,
        current_target_id=_LAST_TARGET_ID,
    )
    _LAST_TARGET_ID = chosen.track_id

    if chosen.distance_m is not None and chosen.distance_m <= cfg.avoid_distance_m:
        tx, _ = chosen.center_px
        err_x = tx - img_cx
        yaw_dir = -1.0 if err_x > 0 else 1.0

        return Decision(
            mode="AVOID",
            target_id=chosen.track_id,
            commands={"yaw_rate": yaw_dir * cfg.max_yaw_rate, "throttle": cfg.avoid_throttle},
            reason=f"AVOID id{chosen.track_id}",
            debug={
                "mode": "AVOID",
                "select": select_debug,
                "err_x_px": float(err_x),
                "yaw_rate": float(yaw_dir * cfg.max_yaw_rate),
                "throttle": float(cfg.avoid_throttle),
                "dist_m": float(chosen.distance_m),
            },
            candidates=top_candidates,
        )

    tx, _ = chosen.center_px
    err_x = tx - img_cx
    half_w = frame_size[0] / 2.0
    norm_err = err_x / max(1.0, half_w)

    if abs(err_x) <= cfg.center_deadband_px:
        yaw_rate = 0.0
        yaw_reason = "deadband"
    else:
        yaw_rate = _clamp(norm_err * cfg.max_yaw_rate, -cfg.max_yaw_rate, cfg.max_yaw_rate)
        yaw_reason = "proportional"

    return Decision(
        mode="TRACK",
        target_id=chosen.track_id,
        commands={"yaw_rate": yaw_rate, "throttle": cfg.track_throttle},
        reason=f"TRACK id{chosen.track_id} ({select_reason})",
        debug={
            "mode": "TRACK",
            "select": select_debug,
            "err_x_px": float(err_x),
            "yaw_rate": float(yaw_rate),
            "yaw_reason": yaw_reason,
            "throttle": float(cfg.track_throttle),
            "label": chosen.label,
            "conf": float(chosen.confidence),
            "dist_m": None if chosen.distance_m is None else float(chosen.distance_m),
        },
        candidates=top_candidates,
    )
