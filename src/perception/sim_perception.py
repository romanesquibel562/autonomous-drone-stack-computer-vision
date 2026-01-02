# src/perception/sim_perception.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

from src.core.types import Detection
from src.simulation.world import World, SimObject

Vec2 = Tuple[float, float]


@dataclass
class CameraModel:
    """
    A very simple "camera" model for simulation.

    - image_size: (width_px, height_px)
    - world_bounds: (max_x, max_y) taken from the World
    - noise_px_std: how much random pixel jitter to add to measurements
    """
    image_size: Tuple[int, int] = (640, 480)
    noise_px_std: float = 1.5
    base_confidence: float = 0.90


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def world_to_pixel(
    pos_xy: Vec2,
    world_bounds: Vec2,
    image_size: Tuple[int, int],
) -> Vec2:
    """
    Map world coordinates (x,y) in [0,max] to pixel coordinates (px,py).
    """
    x, y = pos_xy
    max_x, max_y = world_bounds
    w_px, h_px = image_size

    px = (x / max_x) * (w_px - 1)
    py = (y / max_y) * (h_px - 1)

    return (px, py)


def object_to_bbox(
    center_px: Vec2,
    obj_size_world: Vec2,
    world_bounds: Vec2,
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """
    Convert an object's world size into an approximate pixel bbox around center_px.
    """
    w_px, h_px = image_size
    max_x, max_y = world_bounds
    size_x, size_y = obj_size_world

    # Convert world-size to pixel-size proportionally
    box_w = (size_x / max_x) * w_px
    box_h = (size_y / max_y) * h_px

    cx, cy = center_px
    x1 = cx - box_w / 2
    y1 = cy - box_h / 2
    x2 = cx + box_w / 2
    y2 = cy + box_h / 2

    # Clamp into image bounds
    x1 = _clamp(x1, 0, w_px - 1)
    y1 = _clamp(y1, 0, h_px - 1)
    x2 = _clamp(x2, 0, w_px - 1)
    y2 = _clamp(y2, 0, h_px - 1)

    return (x1, y1, x2, y2)


def simulate_detections(
    world: World,
    camera: CameraModel,
    rng: random.Random,
) -> List[Detection]:
    """
    Convert world objects into Detection objects, adding slight noise and confidence.
    """
    detections: List[Detection] = []

    world_bounds = world.bounds_xy
    image_size = camera.image_size

    for obj in world.objects:
        cx, cy = world_to_pixel(obj.position_xy, world_bounds, image_size)

        cx_noisy = cx + rng.gauss(0, camera.noise_px_std)
        cy_noisy = cy + rng.gauss(0, camera.noise_px_std)

        conf = camera.base_confidence + rng.uniform(-0.08, 0.05)
        conf = float(_clamp(conf, 0.0, 1.0))

        bbox = object_to_bbox(
            center_px=(cx_noisy, cy_noisy),
            obj_size_world=obj.size_xy,
            world_bounds=world_bounds,
            image_size=image_size,
        )

        detections.append(
            Detection(
                label=obj.label,
                confidence=conf,
                bbox=bbox,
                center_px=(cx_noisy, cy_noisy),
            )
        )

    return detections


