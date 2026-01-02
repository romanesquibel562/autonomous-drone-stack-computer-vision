# src/state_estimation/tracker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from src.core.types import Detection, Track

Vec2 = Tuple[float, float]


def _euclidean(a: Vec2, b: Vec2) -> float:
    ax, ay = a
    bx, by = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


@dataclass
class _TrackState:
    """
    Internal mutable track state.

    Why internal + mutable?
    - We want to update it each frame.
    - But we output an immutable (frozen) Track snapshot for downstream modules.
    """
    track_id: int
    label: str
    confidence: float
    center_px: Vec2
    velocity_px_s: Vec2
    distance_m: Optional[float]
    age_frames: int
    last_seen_ts: float


class NearestNeighborTracker:
    """
    Minimal but real tracker:
    - Associate detections to tracks by nearest pixel distance (same label).
    - Estimate velocity as delta(center)/dt.
    - Create new tracks when unmatched.
    - Drop tracks if not seen for max_missed_time_s.
    """

    def __init__(self, match_threshold_px: float = 40.0, max_missed_time_s: float = 1.0):
        self.match_threshold_px = match_threshold_px
        self.max_missed_time_s = max_missed_time_s

        self._next_id = 1
        self._tracks: Dict[int, _TrackState] = {}

    def update(self, detections: List[Detection], now_ts: float, dt: float) -> List[Track]:
        # Keep track of which tracks/detections got matched this frame
        unmatched_track_ids = set(self._tracks.keys())
        assigned_detection_indices = set()

        # 1) Match detections to existing tracks (nearest neighbor)
        for di, det in enumerate(detections):
            best_id = None
            best_dist = float("inf")

            for tid, tr in self._tracks.items():
                # Avoid switching identities across classes
                if tr.label != det.label:
                    continue

                dist = _euclidean(det.center_px, tr.center_px)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            # Accept match only if close enough
            if best_id is not None and best_dist <= self.match_threshold_px:
                tr = self._tracks[best_id]

                old_cx, old_cy = tr.center_px
                new_cx, new_cy = det.center_px

                vx = (new_cx - old_cx) / dt if dt > 0 else 0.0
                vy = (new_cy - old_cy) / dt if dt > 0 else 0.0

                tr.center_px = det.center_px
                tr.velocity_px_s = (vx, vy)
                tr.confidence = det.confidence
                tr.age_frames += 1
                tr.last_seen_ts = now_ts

                unmatched_track_ids.discard(best_id)
                assigned_detection_indices.add(di)

        # 2) Create new tracks for detections that weren't matched
        for di, det in enumerate(detections):
            if di in assigned_detection_indices:
                continue

            tid = self._next_id
            self._next_id += 1

            self._tracks[tid] = _TrackState(
                track_id=tid,
                label=det.label,
                confidence=det.confidence,
                center_px=det.center_px,
                velocity_px_s=(0.0, 0.0),
                distance_m=None,
                age_frames=1,
                last_seen_ts=now_ts,
            )

        # 3) Drop tracks that haven't been seen recently
        to_delete = []
        for tid, tr in self._tracks.items():
            if (now_ts - tr.last_seen_ts) > self.max_missed_time_s:
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        # 4) Output frozen Track snapshots (what downstream modules should consume)
        snapshots: List[Track] = []
        for tr in self._tracks.values():
            snapshots.append(
                Track(
                    track_id=tr.track_id,
                    label=tr.label,
                    confidence=tr.confidence,
                    center_px=tr.center_px,
                    velocity_px_s=tr.velocity_px_s,
                    distance_m=tr.distance_m,
                    age_frames=tr.age_frames,
                    last_seen_ts=tr.last_seen_ts,
                )
            )

        snapshots.sort(key=lambda t: t.track_id)
        return snapshots
