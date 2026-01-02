# src/simulation/world.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random

Vec2 = Tuple[float, float]

@dataclass
class SimObject:
    """
    A simple object that exists in the simulated world.

    label: semantic class (e.g., "vehicle", "person", "obstacle")
    position_xy: (x, y) in "world units" (we'll treat these like meters)
    velocity_xy: (vx, vy) in world units per second
    """
    obj_id: int
    label: str
    position_xy: Vec2
    velocity_xy: Vec2
    size_xy: Vec2 = (1.0, 1.0)  # default size

@dataclass
class World:
    """
    A minimal discrete-time simulation world.
    Holds objects and advances them forward in time.
    """
    objects: List[SimObject] = field(default_factory=list)   
    time_s: float = 0.0
    bounds_xy: Vec2 = (50.0, 50.0) # world boundry: x in [0, 50], y in [0, 50]

    def step(self, dt: float) -> None:
        """ Advance the world by dt seconds."""
        self.time_s += dt

        max_x, max_y = self.bounds_xy

        for obj in self.objects:
            x, y = obj.position_xy
            vx, vy = obj.velocity_xy

            # Basic Kinematics: new_position = old_position + velocity * dt
            x_new = x + vx * dt
            y_new = y + vy * dt

            # Simple boundary behavior: bounce off edges
            if x_new < 0 or x_new > max_x:
                vx = -vx
                x_new = max(0, min(x_new, max_x))
            if y_new < 0 or y_new > max_y:
                vy = -vy
                y_new = max(0, min(y_new, max_y))

            obj.position_xy = (x_new, y_new)
            obj.velocity_xy = (vx, vy)

    @staticmethod
    def demo_scenario(seed: int = 7) -> "World":
        """
        Create a deterministic, repeatable scenario with a few moving objects.
        The seed ensures the same initial conditions every run (good for demos).
        """
        rng = random.Random(seed)

        w = World(bounds_xy=(50.0, 50.0))

        # Spawn a target (e.g., a "vehicle") moving diagonally 
        w.objects.append(
            SimObject(
                obj_id=1,
                label="vehicle",
                position_xy=(10.0, 10.0),
                velocity_xy=(2.0, 1.2),
                size_xy=(2.0, 1.2)
            )
        )   

        # Spawn a person moving relatively slowly
        w.objects.append(
            SimObject(
                obj_id=2,
                label="person",
                position_xy=(30.0, 45.0),
                velocity_xy=(-0.6, -3.0),
                size_xy=(0.6, 0.6) # approximate human size (in shape)
            )
        )

        # Spawn a stationary obstacle
        w.objects.append(
            SimObject(
                obj_id=3,
                label="obstacle",
                position_xy=(25.0, 20.0),
                velocity_xy=(0.0, 0.0),
                size_xy=(3.0, 3.0)
            )
        )

        w.objects.append(
            SimObject(
                obj_id=4,
                label=rng.choice(["vehicle", "person"]),
                position_xy=(rng.uniform(5, 45), rng.uniform(5, 45)),
                velocity_xy=(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)),
                size_xy=(1.0, 1.0),
            )
        )

        return w
