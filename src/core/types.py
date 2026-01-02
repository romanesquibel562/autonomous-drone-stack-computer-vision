# src/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any

Px = float
Sec = float
Vec2 = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x1,y1,x2,y2


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox
    center_px: Vec2


@dataclass(frozen=True)
class Track:
    track_id: int
    label: str
    confidence: float
    center_px: Vec2
    velocity_px_s: Vec2
    distance_m: Optional[float]
    age_frames: int
    last_seen_ts: float


@dataclass(frozen=True)
class EgoState:
    position_xyz: Tuple[float, float, float]
    heading_rad: float
    speed_m_s: float


@dataclass(frozen=True)
class CandidateScore:
    """
    Compact per-track scoring breakdown for explainability + clean logs.
    Keep this small & numeric so it stays readable.
    """
    track_id: int
    label: str
    score: float
    confidence: float
    center_factor: float
    age_frames: int
    distance_m: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """
    mode: PATROL | TRACK | AVOID | HOLD
    commands: e.g. {"yaw_rate": 0.2, "throttle": 0.5}
    reason: short one-liner for humans
    debug: structured, stable fields for loggers (numbers/flags)
    candidates: optional top scoring candidates
    """
    mode: str
    target_id: Optional[int]
    commands: Dict[str, float]
    reason: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)
    candidates: Optional[List[CandidateScore]] = None

