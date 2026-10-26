"""NumPy sampling of a B=1 pivopt trajectory, in piece-local seconds."""

from dataclasses import dataclass
from math import factorial

import numpy as np

from pivopt import ForwardStatus
from pivopt import numpy as cpu


@dataclass
class Trajectory:
    coefficients: np.ndarray
    durations: np.ndarray

    @property
    def duration(self):
        return float(self.durations.sum())

    def sample(self, time, derivative=0):
        """Sample scalars or arrays of times; outside times clamp to an endpoint."""
        time = np.clip(np.asarray(time, dtype=float), 0, self.duration)
        boundaries = np.r_[0.0, np.cumsum(self.durations)]
        piece = np.minimum(
            np.searchsorted(boundaries[1:], time, side="right"),
            len(self.durations) - 1,
        )
        local = time - boundaries[piece]
        coefficients = self.coefficients[piece]
        value = np.zeros((*time.shape, coefficients.shape[-1]))
        for power in range(coefficients.shape[-2] - 1, derivative - 1, -1):
            factor = factorial(power) / factorial(power - derivative)
            value = value * local[..., None] + factor * coefficients[..., power, :]
        return value

    def reference(self, time):
        """Position/velocity/acceleration, with a stationary target after T."""
        position = self.sample(time)
        if time >= self.duration:
            return position, np.zeros_like(position), np.zeros_like(position)
        return position, self.sample(time, 1), self.sample(time, 2)


def solve(plan, inputs):
    result = cpu.build_solver(plan, dtype=np.float64).solve(inputs)
    if result.status[0] != ForwardStatus.RESULT_AVAILABLE:
        raise RuntimeError(f"trajectory solve failed: {result.diagnostics}")
    if not np.isfinite(result.coefficients).all():
        raise RuntimeError("trajectory solve returned nonfinite coefficients")
    print(f"Optimized {len(result.durations[0])} pieces; cost={result.cost[0]:.6g}")
    return Trajectory(result.coefficients[0].copy(), result.durations[0].copy())


def box_corridor(vertices, padding):
    """Per-piece AABBs as pivopt halfspaces and viewer line boxes."""
    vertices = np.asarray(vertices, dtype=float)
    lower = np.minimum(vertices[:-1], vertices[1:]) - padding
    upper = np.maximum(vertices[:-1], vertices[1:]) + padding
    dimension = vertices.shape[1]
    normals = np.r_[np.eye(dimension), -np.eye(dimension)]
    rows = np.broadcast_to(normals, (len(lower), 2 * dimension, dimension))
    halfspaces = np.concatenate((rows, np.c_[-upper, lower][..., None]), axis=-1)
    return halfspaces[None].copy(), list(zip(lower, upper, strict=True))


def corridor_violation(position, time, durations, halfspaces):
    piece = np.minimum(
        np.searchsorted(np.cumsum(durations), time, side="right"), len(durations) - 1
    )
    rows = halfspaces[0, piece]
    return float(
        np.maximum(
            np.einsum("nfd,nd->nf", rows[..., :-1], position) + rows[..., -1], 0
        ).max()
    )
