"""Learn corridor, uniform caps and total time for Unicycle gate tasks.

Stage 0 of ``agent_docs/gate_testbed_1017.md`` with the G4 revisions of
``agent_docs/unicycle_gate_testbed_1018.md``. Geometry warm-up never calls
the solver. Paired profile runs can load the same ``--warmup-checkpoint``.
Evaluation uses real wall boxes and polynomial gate crossings, independently
of the learned corridor. All policy and solver arrays are CPU float64.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
from constraints import Calibration, add_constraint_argument, term
from torch import Tensor, nn
from torch.nn import functional as F

from pivopt import ForwardStatus, OptimizerStatus
from pivopt import problem as p
from pivopt.torch import Solver, build_solver

FACES = 4
BODY_RADIUS = 0.3
MINIMUM_PROGRESS = 0.3
LATERAL_RANGE = 1.5
ROTATION_RANGE = math.pi / 6
LENGTH_SCALE = 0.25
SLACK_BIAS = -2.0
TRACK_WIDTH = 0.5
SPEED_SMOOTHING = 1e-3
REST_ACCELERATION = 0.3
BOUNDARY_YAW_RANGE = math.pi / 6
EVALUATION_SPEED = 0.05
NUMERICAL_SPEED_FLOOR = 1e-9
ROOM_LIMIT = 0.4
ROOM_SAMPLES = 41
WHEEL_LIMIT = 1.2
PHYSICAL_CAPS = (1.2, 1.5, 2.0)
CAP_NAMES = ("speed_cap", "centripetal_cap", "forward_acceleration_cap")
CONSTRAINT_NAMES = ("corridor", *CAP_NAMES, "wheel_speeds")
SOLVER_PATHS = ("corridor", *CAP_NAMES, "total_duration")
UNSAFE_QUERIES = 64
FREE_QUERIES = 16
SEED_QUERIES = 8
JAMB_ARC_SAMPLES = 4
MEMBERSHIP_SHARPNESS = 20.0
DENSE_INTERVALS = 128
VALIDITY_BOX_PENETRATION = 0.01
VALIDITY_CORRIDOR_TOLERANCE = 0.001
GEOMETRY_LOSS_KINDS = ("current", "diff2")
DIFF2_MEMBERSHIP_ALPHA = MEMBERSHIP_SHARPNESS / LENGTH_SCALE
DIFF2_LOG_EPSILON = 1e-6
DIFF2_GATE_BOUNDARY_SAMPLES = 48
DIFF2_GATE_BOUNDARY_SECTIONS = 6
DIFF2_LOCAL_FILL_SAMPLES = 24
DIFF2_LOCAL_LATERAL_CANDIDATES = 9
DIFF2_GLOBAL_FILL_SAMPLES = 64
DIFF2_FILL_RADIUS = 0.4
DIFF2_GLOBAL_WEIGHT_SIGMA = 0.4
DIFF2_LOCAL_WEIGHT = 0.75
DIFF2_GLOBAL_WEIGHT = 0.25
DIFF2_UNDERFILL_WEIGHT = 0.25
DIFF2_SIZE_WEIGHT = 0.05
DIFF2_RAY_EDGE_SAMPLES = 8
DIFF2_RAY_STEP = 0.01
DIFF2_RAY_REACH = 0.4
DIFF2_RAY_STEPS = round(DIFF2_RAY_REACH / DIFF2_RAY_STEP) + 1
SAFE_CLEARANCE = 0.1
SEED_CLEARANCE = 0.05
ACCELERATION_SCALE = 1.5
JERK_SCALE = 5.0
MAXIMUM_ITERATIONS = 256
REFERENCE_ITERATIONS = 2048
GRADIENT_TOLERANCE = 1e-3
FAILURE_COST = 100.0


@dataclass(frozen=True)
class TaskConfig:
    """Complete geometry and outer-loss contract for one gate task."""

    name: str
    pieces: int
    gates: int
    capsule_radius: float
    goal_x: float
    goal_half_y: float
    progress_jitter: float
    gate_offset: tuple[float, float]
    alternating_offsets: bool
    gate_angle: tuple[float, float]
    alternating_angles: bool
    gate_half_width: tuple[float, float]
    wall_reach: float
    boundary_x: tuple[float, float]
    speed_command: tuple[float, float]
    opening_margin: float
    seed_clearance: float
    train_clearance: bool
    corridor_activation: float
    perturbations: int = 0

    @property
    def progress(self) -> slice:
        return slice(0, self.pieces - 1)

    @property
    def lateral(self) -> slice:
        return slice(self.progress.stop, self.progress.stop + self.pieces - 1)

    @property
    def rotations(self) -> slice:
        return slice(self.lateral.stop, self.lateral.stop + self.pieces)

    @property
    def slack(self) -> slice:
        return slice(self.rotations.stop, self.rotations.stop + self.pieces * FACES)

    @property
    def geometry_dim(self) -> int:
        return self.slack.stop

    @property
    def caps(self) -> slice:
        return slice(self.geometry_dim, self.geometry_dim + len(CAP_NAMES))

    @property
    def time(self) -> int:
        return self.caps.stop

    @property
    def action_dim(self) -> int:
        return self.time + 1

    @property
    def feature_dim(self) -> int:
        return 4 + 2 + self.gates * 5 + 2


GATE2 = TaskConfig(
    name="gate2",
    pieces=6,
    gates=2,
    capsule_radius=0.1,
    goal_x=6.0,
    goal_half_y=0.5,
    progress_jitter=0.03,
    gate_offset=(-0.8, 0.8),
    alternating_offsets=False,
    gate_angle=(-math.pi / 12, math.pi / 12),
    alternating_angles=False,
    gate_half_width=(0.45, 0.65),
    wall_reach=2.0,
    boundary_x=(3.0, 5.0),
    speed_command=(0.5, 1.0),
    opening_margin=0.05,
    seed_clearance=0.05,
    train_clearance=True,
    corridor_activation=0.033,
)
SGATE3 = TaskConfig(
    name="sgate3",
    pieces=8,
    gates=3,
    capsule_radius=0.08,
    goal_x=9.0,
    goal_half_y=0.3,
    progress_jitter=0.02,
    gate_offset=(0.6, 0.9),
    alternating_offsets=True,
    gate_angle=(math.pi / 12, 5 * math.pi / 36),
    alternating_angles=True,
    gate_half_width=(0.40, 0.48),
    wall_reach=3.0,
    boundary_x=(4.5, 5.5),
    speed_command=(0.6, 0.9),
    opening_margin=0.0,
    seed_clearance=0.0,
    train_clearance=False,
    corridor_activation=0.10,
    perturbations=8,
)
TASKS = {task.name: task for task in (GATE2, SGATE3)}

# The historical names remain the gate2 contract. New code derives dimensions
# from its TaskConfig, so importing this example without --task is unchanged.
PIECES = GATE2.pieces
GATES = GATE2.gates
CAPSULE_RADIUS = GATE2.capsule_radius
GATE_HALF_WIDTH = GATE2.gate_half_width
_PROGRESS = GATE2.progress
_LATERAL = GATE2.lateral
_ROTATIONS = GATE2.rotations
_SLACK = GATE2.slack
GEOMETRY_DIM = GATE2.geometry_dim
_CAPS = GATE2.caps
_TIME = GATE2.time
ACTION_DIM = GATE2.action_dim
FEATURE_DIM = GATE2.feature_dim

# G4 S1 (Q29), seed-1000 development references: the corridor and wheel rho
# from 1.5x the time lower bound at physical caps, the speed cap from 2x at
# 0.625 caps; centripetal/forward caps never bind and take the speed-cap rho.
CORRIDOR = Calibration(1000.0, 0.05, 11.0, 0.011, 0.033)
SPEED = Calibration(1.0, 0.1, 0.022, 0.022, 0.12)
CENTRIPETAL = Calibration(1.0, 0.15, 0.033, 0.033, 0.15)
FORWARD_ACCELERATION = Calibration(1.0, 0.2, 0.044, 0.044, 0.2)
WHEEL = Calibration(3.0, 0.05, 0.033, 0.011, 0.12)
CALIBRATIONS = (CORRIDOR, SPEED, CENTRIPETAL, FORWARD_ACCELERATION, WHEEL)
SGATE3_CALIBRATIONS = (
    Calibration(3000.0, 0.05, 33.0, 0.011, SGATE3.corridor_activation),
    Calibration(10.0, 0.1, 0.22, 0.022, 0.12),
    Calibration(10.0, 0.15, 0.33, 0.033, 0.15),
    Calibration(10.0, 0.2, 0.44, 0.044, 0.2),
    WHEEL,
)


@dataclass(frozen=True)
class LossDefinition:
    fill_coverage: float = 0.1
    fill_weight: float = 0.1
    size_weight: float = 0.05
    clearance_tail_fraction: float = 0.05
    opening_margin: float = SEED_CLEARANCE
    seed_clearance: float = SEED_CLEARANCE
    train_clearance: bool = True
    exclusion: str = "conservative_smooth_margin"
    opening: str = "conditional_wall_band"
    room_weight: float = 1.0
    room_limit: float = ROOM_LIMIT


LOSS_DEFINITION = LossDefinition()


def loss_definition(task: TaskConfig) -> LossDefinition:
    return replace(
        LOSS_DEFINITION,
        opening_margin=task.opening_margin,
        seed_clearance=task.seed_clearance,
        train_clearance=task.train_clearance,
    )


def task_calibrations(task: TaskConfig) -> tuple[Calibration, ...]:
    return CALIBRATIONS if task is GATE2 else SGATE3_CALIBRATIONS


@dataclass(frozen=True)
class LossWeights:
    pace: float = 1.0
    acceleration_low: float = 0.07436507321617629
    jerk_low: float = 1.4498965412619294
    clearance: float = 1.0
    geometry: float = 1.0
    gate: float = 1.0


LOSS_WEIGHTS = LossWeights()
SGATE3_LOSS_WEIGHTS = LossWeights(
    acceleration_low=0.030042761353897805,
    jerk_low=0.20803044809551105,
)
WARMUP_WEIGHTS = LossWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
LEARNING_RATE = 2e-4
WARMUP_LEARNING_RATE = 1e-3


def task_weights(task: TaskConfig) -> LossWeights:
    return LOSS_WEIGHTS if task is GATE2 else SGATE3_LOSS_WEIGHTS


@dataclass(frozen=True)
class SceneBatch:
    start: Tensor
    goal: Tensor
    start_yaw: Tensor
    goal_yaw: Tensor
    gate_centers: Tensor
    gate_angles: Tensor
    gate_half_widths: Tensor
    gate_thickness: Tensor
    speed_command: Tensor
    smoothness: Tensor
    wall_centers: Tensor
    wall_rotations: Tensor
    wall_half_extents: Tensor
    unsafe_queries: Tensor
    free_queries: Tensor

    @property
    def batch_size(self) -> int:
        return self.start.shape[0]

    @property
    def task_distance(self) -> Tensor:
        path = torch.cat(
            (self.start[:, None], self.gate_centers, self.goal[:, None]), dim=1
        )
        return torch.linalg.vector_norm(path[:, 1:] - path[:, :-1], dim=-1).sum(-1)

    @property
    def main_yaw(self) -> Tensor:
        delta = self.goal - self.start
        return torch.atan2(delta[:, 1], delta[:, 0])

    def features(self) -> Tensor:
        task = task_for_scenes(self)
        gates = torch.cat(
            (
                self.gate_centers / task.goal_x,
                self.gate_angles[..., None] / math.pi,
                self.gate_half_widths[..., None],
                self.gate_thickness[..., None],
            ),
            dim=-1,
        ).flatten(1)
        return torch.cat(
            (
                self.start / task.goal_x,
                self.goal / task.goal_x,
                (self.start_yaw - self.main_yaw)[:, None] / BOUNDARY_YAW_RANGE,
                (self.goal_yaw - self.main_yaw)[:, None] / BOUNDARY_YAW_RANGE,
                gates,
                self.speed_command[:, None] / PHYSICAL_CAPS[0],
                self.smoothness[:, None],
            ),
            dim=-1,
        )

    def select(self, index) -> SceneBatch:
        return SceneBatch(*(getattr(self, field.name)[index] for field in fields(self)))


def task_for_scenes(scenes: SceneBatch) -> TaskConfig:
    gates = scenes.gate_centers.shape[1]
    for task in TASKS.values():
        if task.gates == gates:
            return task
    raise ValueError(f"no gate task has {gates} gates")


def perpendicular(vectors: Tensor) -> Tensor:
    return torch.stack((-vectors[..., 1], vectors[..., 0]), dim=-1)


def rotation(angles: Tensor) -> Tensor:
    first = torch.stack((angles.cos(), angles.sin()), dim=-1)
    return torch.stack((first, perpendicular(first)), dim=-1)


def box_signed_distances(
    points: Tensor, centers: Tensor, rotations: Tensor, half_extents: Tensor
) -> Tensor:
    """Exact point/OBB signed distance, with explicit point and wall axes."""
    local = torch.einsum(
        "bnwd,bwde->bnwe", points[:, :, None] - centers[:, None], rotations
    )
    residual = local.abs() - half_extents[:, None]
    return torch.linalg.vector_norm(residual.clamp_min(0), dim=-1) + residual.amax(
        -1
    ).clamp_max(0)


def environment_distance(points: Tensor, scenes: SceneBatch) -> Tensor:
    shape = points.shape[:-1]
    distances = box_signed_distances(
        points.reshape(scenes.batch_size, -1, 2),
        scenes.wall_centers,
        scenes.wall_rotations,
        scenes.wall_half_extents,
    )
    return distances.amin(-1).reshape(shape)


def build_wall_geometry(
    task: TaskConfig,
    centers: Tensor,
    angles: Tensor,
    half_widths: Tensor,
    thickness: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build gate jambs and the two fixed horizontal domain walls."""
    batch_size = centers.shape[0]
    gate_rotations = rotation(angles)
    side_centers = centers[:, :, None] + (
        torch.tensor((-1.0, 1.0), dtype=centers.dtype)[None, None, :, None]
        * ((task.wall_reach + half_widths) / 2)[..., None, None]
        * gate_rotations[..., :, 1][:, :, None]
    )
    side_extents = torch.stack((thickness / 2, (task.wall_reach - half_widths) / 2), -1)
    center_x, half_x = task.boundary_x
    domain_centers = centers.new_tensor(((center_x, -2.075), (center_x, 2.075)))
    wall_centers = torch.cat(
        (side_centers.flatten(1, 2), domain_centers[None].expand(batch_size, -1, -1)),
        1,
    )
    wall_rotations = torch.cat(
        (
            gate_rotations[:, :, None].expand(-1, -1, 2, -1, -1).flatten(1, 2),
            torch.eye(2, dtype=centers.dtype)[None, None].expand(batch_size, 2, -1, -1),
        ),
        1,
    )
    wall_extents = torch.cat(
        (
            side_extents[:, :, None].expand(-1, -1, 2, -1).flatten(1, 2),
            centers.new_tensor((half_x, 0.075))[None, None].expand(batch_size, 2, -1),
        ),
        1,
    )
    return wall_centers, wall_rotations, wall_extents


def sample_scenes(
    batch_size: int,
    *,
    generator: torch.Generator,
    task: TaskConfig = GATE2,
    geometry_loss: str = "current",
    capsule_radius: float | None = None,
) -> SceneBatch:
    """Scene-only query labels use the same RNG as their scene, never actions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if geometry_loss not in GEOMETRY_LOSS_KINDS:
        raise ValueError(f"unknown geometry loss: {geometry_loss!r}")
    if capsule_radius is None:
        capsule_radius = task.capsule_radius
    if not math.isfinite(capsule_radius) or capsule_radius <= 0:
        raise ValueError("capsule radius must be finite and positive")

    def uniform(*shape):
        return torch.rand(
            (batch_size, *shape), generator=generator, dtype=torch.float64
        )

    start = torch.zeros((batch_size, 2), dtype=torch.float64)
    if task is GATE2:
        # Preserve the original operation order for the gate2 replay.
        goal_y = uniform(1) - 0.5
    else:
        goal_y = (2 * uniform(1) - 1) * task.goal_half_y
    goal = torch.cat((torch.full_like(start[:, :1], task.goal_x), goal_y), -1)
    delta = goal - start
    tangent = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    normal = perpendicular(tangent)
    progress = (
        uniform(task.gates) * (2 * task.progress_jitter)
        - task.progress_jitter
        + torch.arange(1, task.gates + 1, dtype=torch.float64) / (task.gates + 1)
    )
    if task.alternating_offsets:
        offset_low, offset_high = task.gate_offset
        magnitude = offset_low + (offset_high - offset_low) * uniform(task.gates)
        first_sign = torch.where(
            uniform() < 0.5,
            torch.full((batch_size,), -1.0, dtype=torch.float64),
            torch.ones(batch_size, dtype=torch.float64),
        )
        alternating = torch.tensor((-1.0, 1.0, -1.0), dtype=torch.float64)
        lateral_offset = magnitude * first_sign[:, None] * alternating[None]
    else:
        low_offset, high_offset = task.gate_offset
        lateral_offset = low_offset + (high_offset - low_offset) * uniform(task.gates)
    centers = (
        start[:, None]
        + progress[..., None] * delta[:, None]
        + lateral_offset[..., None] * normal[:, None]
    )
    main_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    if task.alternating_angles:
        angle_low, angle_high = task.gate_angle
        magnitude = angle_low + (angle_high - angle_low) * uniform(task.gates)
        first_sign = torch.where(
            uniform() < 0.5,
            torch.full((batch_size,), -1.0, dtype=torch.float64),
            torch.ones(batch_size, dtype=torch.float64),
        )
        alternating = torch.tensor((-1.0, 1.0, -1.0), dtype=torch.float64)
        angle_offset = magnitude * first_sign[:, None] * alternating[None]
    elif task is GATE2:
        angle_offset = (uniform(task.gates) * 2 - 1) * (math.pi / 12)
    else:
        low_angle, high_angle = task.gate_angle
        angle_offset = low_angle + (high_angle - low_angle) * uniform(task.gates)
    angles = main_yaw[:, None] + angle_offset
    low, high = task.gate_half_width
    half_widths = low + (high - low) * uniform(task.gates)
    thickness = torch.full_like(half_widths, 0.15)
    gate_rotations = rotation(angles)
    if task is GATE2:
        # Keep these expressions byte-for-byte equivalent to the U3 generator.
        side_centers = centers[:, :, None] + (
            torch.tensor((-1.0, 1.0), dtype=torch.float64)[None, None, :, None]
            * ((2.0 + half_widths) / 2)[..., None, None]
            * gate_rotations[..., :, 1][:, :, None]
        )
        side_extents = torch.stack((thickness / 2, (2.0 - half_widths) / 2), -1)
        domain_centers = torch.tensor(
            ((3.0, -2.075), (3.0, 2.075)), dtype=torch.float64
        )
        wall_centers = torch.cat(
            (
                side_centers.flatten(1, 2),
                domain_centers[None].expand(batch_size, -1, -1),
            ),
            1,
        )
        wall_rotations = torch.cat(
            (
                gate_rotations[:, :, None].expand(-1, -1, 2, -1, -1).flatten(1, 2),
                torch.eye(2, dtype=torch.float64)[None, None].expand(
                    batch_size, 2, -1, -1
                ),
            ),
            1,
        )
        wall_extents = torch.cat(
            (
                side_extents[:, :, None].expand(-1, -1, 2, -1).flatten(1, 2),
                torch.tensor((5.0, 0.075), dtype=torch.float64)[None, None].expand(
                    batch_size, 2, -1
                ),
            ),
            1,
        )
    else:
        wall_centers, wall_rotations, wall_extents = build_wall_geometry(
            task, centers, angles, half_widths, thickness
        )
    empty = torch.empty((batch_size, 0, 2), dtype=torch.float64)
    start_yaw = main_yaw + (2 * uniform() - 1) * BOUNDARY_YAW_RANGE
    goal_yaw = main_yaw + (2 * uniform() - 1) * BOUNDARY_YAW_RANGE
    scenes = SceneBatch(
        start,
        goal,
        start_yaw,
        goal_yaw,
        centers,
        angles,
        half_widths,
        thickness,
        task.speed_command[0]
        + (task.speed_command[1] - task.speed_command[0]) * uniform(),
        uniform(),
        wall_centers,
        wall_rotations,
        wall_extents,
        empty,
        empty,
    )
    # Rejection sampling from expanded OBBs respects rounded inflated corners.
    wall_indices = torch.arange(UNSAFE_QUERIES) % wall_centers.shape[1]
    query_centers = wall_centers[:, wall_indices]
    query_rotations = wall_rotations[:, wall_indices]
    query_extents = wall_extents[:, wall_indices] + BODY_RADIUS
    unsafe = query_centers.clone()
    pending = torch.ones((batch_size, UNSAFE_QUERIES), dtype=torch.bool)
    while bool(pending.any()):
        local = (2 * uniform(UNSAFE_QUERIES, 2) - 1) * query_extents
        candidates = query_centers + torch.einsum(
            "bqde,bqe->bqd", query_rotations, local
        )
        accepted = environment_distance(candidates, scenes) <= BODY_RADIUS
        take = pending & accepted
        unsafe = torch.where(take[..., None], candidates, unsafe)
        pending &= ~accepted
    # Cover the inflated jamb boundaries explicitly. Uniform points alone can
    # miss the small part of a long wall that decides opening clearance.
    arc = torch.linspace(0, math.pi / 2, JAMB_ARC_SAMPLES, dtype=torch.float64)
    front = torch.tensor((-1.0, 1.0), dtype=torch.float64)[None, None, None, :, None]
    side = torch.tensor((-1.0, 1.0), dtype=torch.float64)[None, None, :, None, None]
    radius = BODY_RADIUS - capsule_radius / 10
    local_x = front * (thickness[..., None, None, None] / 2 + radius * arc.cos())
    local_y = side * (half_widths[..., None, None, None] - radius * arc.sin())
    critical = (
        centers[:, :, None, None, None]
        + local_x[..., None] * gate_rotations[..., :, 0][:, :, None, None, None]
        + local_y[..., None] * gate_rotations[..., :, 1][:, :, None, None, None]
    ).reshape(batch_size, -1, 2)
    unsafe = torch.cat((critical, unsafe[:, critical.shape[1] :]), dim=1)
    free = torch.zeros((batch_size, task.pieces, FREE_QUERIES, 2), dtype=torch.float64)
    pending = torch.ones(free.shape[:-1], dtype=torch.bool)
    while bool(pending.any()):
        phase = (
            torch.arange(task.pieces)[None, :, None]
            + uniform(task.pieces, FREE_QUERIES)
        ) / task.pieces
        lateral = 2.0 * (2 * uniform(task.pieces, FREE_QUERIES) - 1)
        candidates = (
            start[:, None, None]
            + phase[..., None] * delta[:, None, None]
            + lateral[..., None] * normal[:, None, None]
        )
        accepted = (environment_distance(candidates, scenes) >= BODY_RADIUS) & (
            candidates[..., 1].abs() <= 2.0 - BODY_RADIUS
        )
        take = pending & accepted
        free = torch.where(take[..., None], candidates, free)
        pending &= ~accepted
    if geometry_loss == "diff2":
        # The diff2 objective owns a fresh set of random wall-interior points.
        # This draw happens only for the explicit override, preserving every
        # default generator state and frozen replay.
        random_unsafe = query_centers.clone()
        pending = torch.ones((batch_size, UNSAFE_QUERIES), dtype=torch.bool)
        while bool(pending.any()):
            local = (2 * uniform(UNSAFE_QUERIES, 2) - 1) * query_extents
            candidates = query_centers + torch.einsum(
                "bqde,bqe->bqd", query_rotations, local
            )
            accepted = environment_distance(candidates, scenes) <= BODY_RADIUS
            take = pending & accepted
            random_unsafe = torch.where(take[..., None], candidates, random_unsafe)
            pending &= ~accepted
        unsafe = random_unsafe
    return replace(scenes, unsafe_queries=unsafe, free_queries=free)


class GatePolicy(nn.Module):
    def __init__(self, task: TaskConfig = GATE2) -> None:
        super().__init__()
        self.task = task
        self.network = nn.Sequential(
            nn.Linear(task.feature_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, task.action_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


@dataclass(frozen=True)
class DecodedAction:
    anchors: Tensor
    rotations: Tensor
    box_centers: Tensor
    half_extents: Tensor
    corridor: Tensor
    caps: Tensor
    total_duration: Tensor


def corridor_with_margin(corridor: Tensor, margin: float) -> Tensor:
    """Inset every normalized half-space without changing the policy record."""
    if margin < 0:
        raise ValueError("corridor margin must be nonnegative")
    if margin == 0:
        return corridor
    return torch.cat((corridor[..., :2], corridor[..., 2:] + margin), -1)


def decode_action(
    action: Tensor, scenes: SceneBatch, *, capsule_radius: float | None = None
) -> DecodedAction:
    task = task_for_scenes(scenes)
    if capsule_radius is None:
        capsule_radius = task.capsule_radius
    if not math.isfinite(capsule_radius) or capsule_radius <= 0:
        raise ValueError("capsule radius must be finite and positive")
    if action.shape != (scenes.batch_size, task.action_dim):
        raise ValueError(f"action must have shape [B,{task.action_dim}]")
    delta = scenes.goal - scenes.start
    distance = torch.linalg.vector_norm(delta, dim=-1)
    tangent = delta / distance[:, None]
    normal = perpendicular(tangent)
    alpha = torch.cat(
        (action[:, task.progress], torch.zeros_like(action[:, :1])), -1
    ).softmax(-1)
    progress = (
        MINIMUM_PROGRESS + (distance[:, None] - task.pieces * MINIMUM_PROGRESS) * alpha
    )
    lateral_raw = action[:, task.lateral]
    lateral = (
        LATERAL_RANGE
        * lateral_raw
        / torch.hypot(lateral_raw, torch.ones_like(lateral_raw))
    )
    interior = (
        scenes.start[:, None]
        + progress.cumsum(-1)[:, :-1, None] * tangent[:, None]
        + lateral[..., None] * normal[:, None]
    )
    anchors = torch.cat((scenes.start[:, None], interior, scenes.goal[:, None]), 1)
    segment = anchors[:, 1:] - anchors[:, :-1]
    direction = segment / torch.linalg.vector_norm(segment, dim=-1, keepdim=True)
    segment_frame = torch.stack((direction, perpendicular(direction)), -1)
    raw_rotation = action[:, task.rotations]
    angles = (
        ROTATION_RANGE
        * raw_rotation
        / torch.hypot(raw_rotation, torch.ones_like(raw_rotation))
    )
    rotations = segment_frame @ rotation(angles)
    midpoint = 0.5 * (anchors[:, :-1] + anchors[:, 1:])
    local_half_segment = torch.einsum("bkde,bkd->bke", rotations, 0.5 * segment)
    base_extents = local_half_segment.abs() + capsule_radius
    slack = LENGTH_SCALE * F.softplus(
        action[:, task.slack].reshape(-1, task.pieces, FACES) + SLACK_BIAS
    )
    normals = torch.stack(
        (
            rotations[..., :, 0],
            -rotations[..., :, 0],
            rotations[..., :, 1],
            -rotations[..., :, 1],
        ),
        -2,
    )
    offsets = (
        torch.einsum("bkfd,bkd->bkf", normals, midpoint)
        + base_extents.repeat_interleave(2, -1)
        + slack
    )
    corridor = torch.cat((normals, -offsets[..., None]), -1)
    paired_slack = slack.reshape(-1, task.pieces, 2, 2)
    half_extents = base_extents + paired_slack.mean(-1)
    center_shift = 0.5 * (paired_slack[..., 0] - paired_slack[..., 1])
    box_centers = midpoint + torch.einsum("bkde,bke->bkd", rotations, center_shift)
    caps = action.new_tensor(PHYSICAL_CAPS) * (
        0.25 + 0.75 * action[:, task.caps].sigmoid()
    )
    lower_time = scenes.task_distance / PHYSICAL_CAPS[0]
    total_duration = lower_time * (1 + 3 * action[:, task.time].sigmoid())
    return DecodedAction(
        anchors, rotations, box_centers, half_extents, corridor, caps, total_duration
    )


def gate_coordinates(points: Tensor, scenes: SceneBatch) -> tuple[Tensor, Tensor]:
    points = points.reshape(scenes.batch_size, -1, 2)
    normal = torch.stack((scenes.gate_angles.cos(), scenes.gate_angles.sin()), -1)
    relative = points[:, None] - scenes.gate_centers[:, :, None]
    return (
        torch.einsum("bgqd,bgd->bgq", relative, normal),
        torch.einsum("bgqd,bgd->bgq", relative, perpendicular(normal)),
    )


def opening_violation(lateral: Tensor, scenes: SceneBatch) -> Tensor:
    definition = loss_definition(task_for_scenes(scenes))
    return F.relu(
        (
            lateral.abs()
            + BODY_RADIUS
            + definition.opening_margin
            - scenes.gate_half_widths[..., None]
        )
        / LENGTH_SCALE
    ).square()


def seed_opening_loss(anchors: Tensor, scenes: SceneBatch) -> Tensor:
    """Opening error at actual forward plane crossings of the seed polyline.

    This is a soft task loss, not a gate-center waypoint label. The rational
    intersection retains its derivative within the selected segment branch.
    """
    normal, lateral = gate_coordinates(anchors, scenes)
    left, right = normal[..., :-1], normal[..., 1:]
    crossing = (left <= 0) & (right > 0)
    denominator = torch.where(crossing, right - left, torch.ones_like(left))
    fraction = -left / denominator
    at_plane = lateral[..., :-1] + fraction * (lateral[..., 1:] - lateral[..., :-1])
    penalties = torch.where(crossing, opening_violation(at_plane, scenes), 0)
    return penalties.amax(-1).mean(-1)


def trajectory_gate_loss(points: Tensor, scenes: SceneBatch) -> Tensor:
    """Per-gate conditional opening proxy on the inflated wall band.

    Normalization is local to each gate, so shortening time inside a wall
    cannot dilute its opening penalty by the rest of the trajectory. Stable
    softmax also avoids an empty-band zero-loss escape. Evaluation still uses
    independent polynomial roots and continuous collision checks.
    """
    normal, lateral = gate_coordinates(points, scenes)
    band = scenes.gate_thickness / 2 + BODY_RADIUS
    membership = (-(normal / band[..., None]).square()).softmax(-1)
    return (membership * opening_violation(lateral, scenes)).sum(-1).mean(-1)


def lateral_room_target(decoded: DecodedAction, scenes: SceneBatch) -> Tensor:
    """Half of the free interval across each box, from the seed segment midpoint.

    Rays run along +/- the box's lateral axis to the body-inflated walls and are
    capped at the room limit. The target is a scene measurement at the current
    geometry and carries no gradient, so the seed cannot gain by moving to
    narrower space.
    """
    definition = loss_definition(task_for_scenes(scenes))
    with torch.no_grad():
        midpoint = 0.5 * (decoded.anchors[:, :-1] + decoded.anchors[:, 1:])
        axis = decoded.rotations[..., :, 1]
        distance = torch.linspace(
            0, definition.room_limit, ROOM_SAMPLES, dtype=midpoint.dtype
        )
        reach = []
        for side in (1.0, -1.0):
            points = (
                midpoint[:, :, None]
                + side * distance[None, None, :, None] * axis[:, :, None]
            )
            free = environment_distance(points, scenes) > BODY_RADIUS
            # First blocked sample along the ray; an unblocked ray keeps the cap.
            blocked = (~free).cumsum(-1) > 0
            reach.append(torch.where(blocked, distance, definition.room_limit).amin(-1))
        return 0.5 * (reach[0] + reach[1])


def lateral_room_loss(decoded: DecodedAction, scenes: SceneBatch) -> Tensor:
    target = lateral_room_target(decoded, scenes)
    narrow = decoded.half_extents[..., 1]
    return F.relu((target - narrow) / LENGTH_SCALE).square().mean(-1)


def current_geometry_losses(
    decoded: DecodedAction, scenes: SceneBatch
) -> dict[str, Tensor]:
    definition = loss_definition(task_for_scenes(scenes))
    normals, offsets = decoded.corridor[..., :2], decoded.corridor[..., 2]
    unsafe_faces = (
        torch.einsum("bkfd,bqd->bkqf", normals, scenes.unsafe_queries)
        + offsets[:, :, None]
    )
    unsafe_distance = (
        torch.logsumexp(MEMBERSHIP_SHARPNESS * unsafe_faces / LENGTH_SCALE, -1)
        / MEMBERSHIP_SHARPNESS
    )
    # Subtract the log(F)/kappa bias to obtain a lower bound on the hard face
    # maximum. A zero exclusion loss cannot hide a query inside a box.
    conservative_distance = unsafe_distance - math.log(FACES) / MEMBERSHIP_SHARPNESS
    exclude = F.relu(-conservative_distance).square().sum(1).mean(-1)
    free_faces = (
        torch.einsum("bkfd,bkqd->bkqf", normals, scenes.free_queries)
        + offsets[:, :, None]
    )
    free_distance = (
        torch.logsumexp(MEMBERSHIP_SHARPNESS * free_faces / LENGTH_SCALE, -1)
        / MEMBERSHIP_SHARPNESS
    )
    coverage = (-MEMBERSHIP_SHARPNESS * free_distance).sigmoid().mean(-1)
    fill = F.relu(definition.fill_coverage - coverage).square().mean(-1)
    fraction = torch.linspace(0, 1, SEED_QUERIES, dtype=decoded.anchors.dtype)
    seed_points = (1 - fraction[None, None, :, None]) * decoded.anchors[
        :, :-1, None
    ] + fraction[None, None, :, None] * decoded.anchors[:, 1:, None]
    clearance = environment_distance(seed_points, scenes) - BODY_RADIUS
    seed_clearance = (
        F.relu((definition.seed_clearance - clearance) / LENGTH_SCALE)
        .square()
        .mean((1, 2))
    )
    seed_opening = seed_opening_loss(decoded.anchors, scenes)
    seed = seed_clearance + seed_opening
    size = F.relu(torch.log(decoded.half_extents / 0.6)).square().mean((1, 2))
    room = lateral_room_loss(decoded, scenes)
    return {
        "exclude": exclude,
        "fill": fill,
        "seed": seed,
        "seed_clearance": seed_clearance,
        "seed_opening": seed_opening,
        "size": size,
        "room": room,
        "fill_coverage": coverage.mean(-1),
        "geometry": exclude
        + seed
        + definition.fill_weight * fill
        + definition.size_weight * size
        + definition.room_weight * room,
    }


def corridor_membership_logits(corridor: Tensor, points: Tensor) -> Tensor:
    """Per-box logits for points shaped [B,N,2]."""
    faces = (
        torch.einsum("bkfd,bnd->bnkf", corridor[..., :2], points)
        + corridor[:, None, :, :, 2]
    )
    return -torch.logsumexp(DIFF2_MEMBERSHIP_ALPHA * faces, -1)


def union_membership(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Stable union probability and log complement over the final box axis."""
    log_complement = F.logsigmoid(-logits).sum(-1)
    return -torch.expm1(log_complement), log_complement


@torch.no_grad()
def diff2_gate_boundary_queries(scenes: SceneBatch) -> Tensor:
    """Exact body-inflated jamb boundary near each opening, 48 points/gate."""
    if DIFF2_GATE_BOUNDARY_SAMPLES % DIFF2_GATE_BOUNDARY_SECTIONS:
        raise RuntimeError("gate boundary sample count must divide its sections")
    section = DIFF2_GATE_BOUNDARY_SAMPLES // DIFF2_GATE_BOUNDARY_SECTIONS
    axis_x = torch.stack((scenes.gate_angles.cos(), scenes.gate_angles.sin()), -1)
    axis_y = perpendicular(axis_x)
    along_face = torch.linspace(-0.5, 0.5, section, dtype=axis_x.dtype)
    angle = torch.linspace(0, math.pi / 2, section, dtype=axis_x.dtype)
    points = []
    for side in (-1.0, 1.0):
        inner_y = side * (scenes.gate_half_widths - BODY_RADIUS)
        face_x = scenes.gate_thickness[..., None] * along_face
        points.append(
            scenes.gate_centers[..., None, :]
            + face_x[..., None] * axis_x[..., None, :]
            + inner_y[..., None, None] * axis_y[..., None, :]
        )
        for front in (-1.0, 1.0):
            corner = (
                scenes.gate_centers
                + front * scenes.gate_thickness[..., None] / 2 * axis_x
                + side * scenes.gate_half_widths[..., None] * axis_y
            )
            radial = (
                front * angle.sin()[None, None, :, None] * axis_x[..., None, :]
                - side * angle.cos()[None, None, :, None] * axis_y[..., None, :]
            )
            points.append(corner[..., None, :] + BODY_RADIUS * radial)
    return torch.cat(points, dim=-2).flatten(1, 2)


@torch.no_grad()
def diff2_local_fill_queries(decoded: DecodedAction, scenes: SceneBatch) -> Tensor:
    phase = torch.linspace(0, 1, DIFF2_LOCAL_FILL_SAMPLES, dtype=decoded.anchors.dtype)
    lateral = torch.linspace(
        -DIFF2_FILL_RADIUS,
        DIFF2_FILL_RADIUS,
        DIFF2_LOCAL_LATERAL_CANDIDATES,
        dtype=decoded.anchors.dtype,
    )
    start, end = decoded.anchors[:, :-1], decoded.anchors[:, 1:]
    segment = end - start
    direction = segment / torch.linalg.vector_norm(segment, dim=-1, keepdim=True)
    centerline = (1 - phase)[None, None, :, None] * start[:, :, None] + phase[
        None, None, :, None
    ] * end[:, :, None]
    candidates = (
        centerline[:, :, :, None]
        + lateral[None, None, None, :, None]
        * perpendicular(direction)[:, :, None, None]
    ).flatten(2, 3)
    clearance = environment_distance(candidates, scenes) - BODY_RADIUS
    score = torch.where(
        clearance > 0, clearance, torch.full_like(clearance, -torch.inf)
    )
    selected = score.topk(DIFF2_LOCAL_FILL_SAMPLES, dim=-1).indices
    chosen_clearance = score.gather(-1, selected)
    if not bool(torch.isfinite(chosen_clearance).all()):
        raise RuntimeError("insufficient local C-space free samples")
    return candidates.gather(
        2, selected[..., None].expand(-1, -1, -1, candidates.shape[-1])
    )


@torch.no_grad()
def diff2_global_fill_queries(scenes: SceneBatch) -> Tensor:
    candidates = scenes.free_queries.flatten(1, 2)
    indices = (
        torch.linspace(
            0,
            candidates.shape[1] - 1,
            DIFF2_GLOBAL_FILL_SAMPLES,
            dtype=torch.float64,
        )
        .round()
        .long()
    )
    return candidates[:, indices]


@torch.no_grad()
def diff2_face_rays(decoded: DecodedAction) -> Tensor:
    axes = decoded.rotations.detach()
    centers = decoded.box_centers.detach()
    half = decoded.half_extents.detach()
    edge_fraction = torch.linspace(-1, 1, DIFF2_RAY_EDGE_SAMPLES, dtype=centers.dtype)
    distance = torch.linspace(0, DIFF2_RAY_REACH, DIFF2_RAY_STEPS, dtype=centers.dtype)
    origins, normals = [], []
    for axis in range(2):
        tangent = 1 - axis
        for side in (1.0, -1.0):
            normal = side * axes[..., :, axis]
            face_center = centers + half[..., axis, None] * normal
            origin = (
                face_center[..., None, :]
                + edge_fraction[None, None, :, None]
                * half[..., tangent, None, None]
                * axes[..., tangent][:, :, None]
            )
            origins.append(origin)
            normals.append(normal)
    origin = torch.stack(origins, 2)
    normal = torch.stack(normals, 2)
    return (
        origin[..., None, :]
        + distance[None, None, None, None, :, None] * normal[:, :, :, None, None]
    )


def point_segment_distance(points: Tensor, anchors: Tensor) -> Tensor:
    """Minimum distance from [B,N,2] points to a [B,K+1,2] polyline."""
    start, end = anchors[:, :-1], anchors[:, 1:]
    segment = end - start
    relative = points[:, :, None] - start[:, None]
    fraction = (relative * segment[:, None]).sum(-1) / segment.square().sum(-1)[:, None]
    nearest = start[:, None] + fraction.clamp(0, 1)[..., None] * segment[:, None]
    return torch.linalg.vector_norm(points[:, :, None] - nearest, dim=-1).amin(-1)


def diff2_geometry_losses(
    decoded: DecodedAction, scenes: SceneBatch
) -> dict[str, Tensor]:
    batch_size = scenes.batch_size
    definition = loss_definition(task_for_scenes(scenes))
    boundary = diff2_gate_boundary_queries(scenes)
    unsafe_points = torch.cat((boundary, scenes.unsafe_queries), 1)
    unsafe_logits = corridor_membership_logits(decoded.corridor, unsafe_points)
    _, log_complement = union_membership(unsafe_logits)
    unsafe_values = -torch.log(torch.exp(log_complement) + DIFF2_LOG_EPSILON)

    rays = diff2_face_rays(decoded)
    ray_clearance = environment_distance(rays, scenes) - BODY_RADIUS
    ray_unsafe = ray_clearance < 0
    origin_unsafe = ray_unsafe[..., 0]
    has_hit = ray_unsafe.any(-1)
    first_hit = ray_unsafe.to(torch.int64).argmax(-1)
    valid_hit = has_hit & ~origin_unsafe & (first_hit > 0)
    hit = rays.gather(
        -2,
        first_hit[..., None, None].expand(*first_hit.shape, 1, 2),
    ).squeeze(-2)
    last_index = (first_hit - 1).clamp_min(0)
    last = rays.gather(
        -2,
        last_index[..., None, None].expand(*last_index.shape, 1, 2),
    ).squeeze(-2)

    origin = rays[..., 0, :].flatten(1, 3)
    origin_logits = corridor_membership_logits(decoded.corridor, origin)
    _, origin_log_complement = union_membership(origin_logits)
    origin_values = -torch.log(
        torch.exp(origin_log_complement) + DIFF2_LOG_EPSILON
    ).reshape_as(origin_unsafe)
    unsafe_sum = unsafe_values.sum(-1) + (origin_values * origin_unsafe).sum(
        dim=(1, 2, 3)
    )
    unsafe_count = unsafe_values.new_full(
        (batch_size,), unsafe_values.shape[-1]
    ) + origin_unsafe.sum(dim=(1, 2, 3))
    unsafe = unsafe_sum / unsafe_count

    hit_logits = corridor_membership_logits(decoded.corridor, hit.flatten(1, 3))
    hit_union, _ = union_membership(hit_logits)
    hit_union = hit_union.reshape_as(valid_hit)
    overrun = (0.5 * hit_union.square() * valid_hit).sum(dim=(1, 2, 3)) / valid_hit.sum(
        dim=(1, 2, 3)
    ).clamp_min(1)

    last_faces = (
        torch.einsum("bkhc,bkfec->bkfeh", decoded.corridor[..., :2], last)
        + decoded.corridor[:, :, None, None, :, 2]
    )
    last_own = torch.sigmoid(-torch.logsumexp(DIFF2_MEMBERSHIP_ALPHA * last_faces, -1))
    underfill = (0.5 * (1 - last_own).square() * valid_hit).sum(
        dim=(1, 2, 3)
    ) / valid_hit.sum(dim=(1, 2, 3)).clamp_min(1)

    local_points = diff2_local_fill_queries(decoded, scenes)
    local_faces = (
        torch.einsum("bkfd,bknd->bknf", decoded.corridor[..., :2], local_points)
        + decoded.corridor[:, :, None, :, 2]
    )
    local_logits = -torch.logsumexp(DIFF2_MEMBERSHIP_ALPHA * local_faces, -1)
    local_fill = F.softplus(-local_logits).mean(dim=(1, 2))

    global_points = diff2_global_fill_queries(scenes)
    global_logits = corridor_membership_logits(decoded.corridor, global_points)
    global_union, _ = union_membership(global_logits)
    distance = point_segment_distance(global_points, decoded.anchors.detach())
    weight = torch.exp(-0.5 * (distance / DIFF2_GLOBAL_WEIGHT_SIGMA).square()).detach()
    global_fill = (weight * -torch.log(global_union + DIFF2_LOG_EPSILON)).mean(-1)
    fill = DIFF2_LOCAL_WEIGHT * local_fill + DIFF2_GLOBAL_WEIGHT * global_fill
    global_faces = (
        torch.einsum("bkfd,bnd->bnkf", decoded.corridor[..., :2], global_points)
        + decoded.corridor[:, None, :, :, 2]
    )
    coverage = (global_faces.amax(-1) <= 0).any(-1).double().mean(-1)

    fraction = torch.linspace(0, 1, SEED_QUERIES, dtype=decoded.anchors.dtype)
    seed_points = (1 - fraction[None, None, :, None]) * decoded.anchors[
        :, :-1, None
    ] + fraction[None, None, :, None] * decoded.anchors[:, 1:, None]
    clearance = environment_distance(seed_points, scenes) - BODY_RADIUS
    seed_clearance = (
        F.relu((definition.seed_clearance - clearance) / LENGTH_SCALE)
        .square()
        .mean((1, 2))
    )
    seed_opening = seed_opening_loss(decoded.anchors, scenes)
    seed = seed_clearance + seed_opening
    size = F.relu(torch.log(decoded.half_extents / 0.6)).square().mean((1, 2))
    room = torch.zeros_like(size)
    geometry = (
        unsafe
        + overrun
        + DIFF2_UNDERFILL_WEIGHT * underfill
        + fill
        + seed
        + DIFF2_SIZE_WEIGHT * size
    )
    return {
        "exclude": unsafe,
        "unsafe": unsafe,
        "overrun": overrun,
        "underfill": underfill,
        "fill": fill,
        "local_fill": local_fill,
        "global_fill": global_fill,
        "seed": seed,
        "seed_clearance": seed_clearance,
        "seed_opening": seed_opening,
        "size": size,
        "room": room,
        "fill_coverage": coverage,
        "geometry": geometry,
    }


def geometry_losses(
    decoded: DecodedAction,
    scenes: SceneBatch,
    *,
    geometry_loss: str = "current",
) -> dict[str, Tensor]:
    if geometry_loss == "current":
        return current_geometry_losses(decoded, scenes)
    if geometry_loss == "diff2":
        return diff2_geometry_losses(decoded, scenes)
    raise ValueError(f"unknown geometry loss: {geometry_loss!r}")


def pace_loss(decoded: DecodedAction, scenes: SceneBatch) -> Tensor:
    return torch.log(
        decoded.total_duration * scenes.speed_command / scenes.task_distance
    ).square()


@dataclass(frozen=True)
class TrajectorySetup:
    plan: p.Plan
    solver: Solver
    task: TaskConfig = GATE2
    corridor_margin: float = 0.0

    def bind(self, scenes: SceneBatch, decoded: DecodedAction) -> p.Inputs:
        if task_for_scenes(scenes) is not self.task:
            raise ValueError("scenes do not match the prepared gate task")
        # Rest endpoints carry their heading in the boundary acceleration.
        head = torch.from_numpy(
            p.unicycle_state(
                scenes.start.numpy(),
                yaw=scenes.start_yaw.numpy(),
                acceleration=REST_ACCELERATION,
            )
        )
        tail = torch.from_numpy(
            p.unicycle_state(
                scenes.goal.numpy(),
                yaw=scenes.goal_yaw.numpy(),
                acceleration=-REST_ACCELERATION,
            )
        )
        initial = torch.from_numpy(
            self.plan.initial_guess(
                waypoints=decoded.anchors[:, 1:-1].detach().numpy(),
                durations=(
                    decoded.total_duration[:, None].expand(-1, self.task.pieces)
                    / self.task.pieces
                )
                .detach()
                .numpy(),
            )
        )
        corridor = corridor_with_margin(decoded.corridor, self.corridor_margin)
        return self.plan.inputs(
            head=head.contiguous(),
            tail=tail.contiguous(),
            total_duration=decoded.total_duration.contiguous(),
            corridor=corridor.contiguous(),
            initial=initial,
            **{
                name: decoded.caps[:, index].contiguous()
                for index, name in enumerate(CAP_NAMES)
            },
        )


def prepare_problem(
    batch_size: int,
    *,
    constraints: str = "hinge",
    maximum_iterations: int = MAXIMUM_ITERATIONS,
    calibrations: tuple[Calibration, ...] | None = None,
    task: TaskConfig = GATE2,
    corridor_margin: float = 0.0,
) -> TrajectorySetup:
    if corridor_margin < 0:
        raise ValueError("corridor margin must be nonnegative")
    if corridor_margin and constraints != "hinge":
        raise ValueError("corridor margin is only defined for hinge constraints")
    if calibrations is None:
        calibrations = task_calibrations(task)

    def ref(name):
        return p.Input(name, differentiate=True)

    trajectory = p.MincoSpec(
        p.Space(position=2),
        derivative_order=3,
        pieces=task.pieces,
        head_state=p.Input("head"),
        tail_state=p.Input("tail"),
        time=p.FixedTotalTime(
            total_duration=ref("total_duration"), minimum_piece_duration=0.1
        ),
        waypoints=p.DirectWaypoints(),
    )
    drive = p.Unicycle(track_width=TRACK_WIDTH, speed_smoothing_factor=SPEED_SMOOTHING)
    makes = (
        lambda profile: p.Inside(trajectory.position, ref("corridor"), profile=profile),
        lambda profile: p.NormBelow(
            drive.forward_velocity, ref(CAP_NAMES[0]), profile=profile
        ),
        lambda profile: p.NormBelow(
            drive.centripetal_acceleration, ref(CAP_NAMES[1]), profile=profile
        ),
        lambda profile: p.NormBelow(
            drive.forward_acceleration, ref(CAP_NAMES[2]), profile=profile
        ),
        lambda profile: p.Within(
            drive.wheel_speeds, -WHEEL_LIMIT, WHEEL_LIMIT, profile=profile
        ),
    )
    spec = p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=16),
        objective=(
            p.Energy("energy"),
            *(
                term(p.Integral, name, make, calibration, constraints)
                for name, make, calibration in zip(
                    CONSTRAINT_NAMES, makes, calibrations, strict=True
                )
            ),
        ),
        forward=p.LBFGS(
            initial_guess=p.Input("initial"),
            memory_size=8,
            settings=p.OptimizerSettings(
                maximum_iterations=maximum_iterations,
                gradient_tolerance=GRADIENT_TOLERANCE,
                past=0,
            ),
        ),
        backward=p.Implicit(
            damping=p.MinimumShift(),
            checks=p.BackwardChecks(
                stationarity_tolerance=None,
                hessian_symmetry_tolerance=None,
                hessian_pivot_tolerance=None,
                adjoint_residual_tolerance=None,
            ),
        ),
    )
    plan = p.build_plan(
        spec,
        p.PlanShape(
            batch_size=batch_size,
            input_shapes=(
                p.InputShape(
                    p.InputRef("corridor"), (batch_size, task.pieces, FACES, 3)
                ),
            ),
        ),
    )
    return TrajectorySetup(
        plan,
        build_solver(plan, dtype=torch.float64, backward=True),
        task,
        corridor_margin,
    )


def sample_derivative(
    coefficients: Tensor,
    durations: Tensor,
    order: int,
    *,
    intervals: int = DENSE_INTERVALS,
) -> Tensor:
    fraction = torch.linspace(0, 1, intervals + 1, dtype=coefficients.dtype)
    local_time = durations[..., None] * fraction
    value = torch.zeros(
        (*coefficients.shape[:2], intervals + 1, coefficients.shape[-1]),
        dtype=coefficients.dtype,
    )
    for degree in range(coefficients.shape[2] - 1, order - 1, -1):
        multiplier = math.factorial(degree) / math.factorial(degree - order)
        value = (
            value * local_time[..., None]
            + multiplier * coefficients[:, :, degree, None]
        )
    return value


def drive_samples(
    coefficients: Tensor, durations: Tensor, *, intervals: int = DENSE_INTERVALS
) -> dict[str, Tensor]:
    # The same smoothed flat unicycle readout that the solver constrains.
    position = sample_derivative(coefficients, durations, 0, intervals=intervals)
    velocity = sample_derivative(coefficients, durations, 1, intervals=intervals)
    acceleration = sample_derivative(coefficients, durations, 2, intervals=intervals)
    speed_squared = velocity.square().sum(-1) + SPEED_SMOOTHING
    speed = speed_squared.sqrt()
    cross = (
        velocity[..., 0] * acceleration[..., 1]
        - velocity[..., 1] * acceleration[..., 0]
    )
    yaw_rate = cross / speed_squared
    return {
        "position": position,
        "speed_cap": speed,
        "centripetal_cap": cross / speed,
        "forward_acceleration_cap": (velocity * acceleration).sum(-1) / speed,
        "wheel_speeds": torch.stack(
            (
                speed - 0.5 * TRACK_WIDTH * yaw_rate,
                speed + 0.5 * TRACK_WIDTH * yaw_rate,
            ),
            -1,
        ),
    }


def trajectory_losses(
    result,
    decoded: DecodedAction,
    scenes: SceneBatch,
    *,
    weights: LossWeights | None = None,
    train_clearance: bool | None = None,
    geometry_loss: str = "current",
) -> dict[str, Tensor]:
    task = task_for_scenes(scenes)
    definition = loss_definition(task)
    if train_clearance is not None:
        definition = replace(definition, train_clearance=train_clearance)
    if weights is None:
        weights = task_weights(task)
    available = result.status == int(ForwardStatus.RESULT_AVAILABLE)
    coefficients = torch.where(
        available[:, None, None, None],
        result.coefficients,
        torch.zeros_like(result.coefficients),
    )
    durations = torch.where(
        available[:, None], result.durations, torch.ones_like(result.durations)
    )
    position = sample_derivative(coefficients, durations, 0)[..., :2]
    acceleration = sample_derivative(coefficients, durations, 2)[..., :2]
    jerk = sample_derivative(coefficients, durations, 3)[..., :2]

    def integral_mean(value):
        return (
            torch.trapezoid(value, dx=1.0 / DENSE_INTERVALS, dim=-1) * durations
        ).sum(-1) / durations.sum(-1)

    a = integral_mean((acceleration / ACCELERATION_SCALE).square().sum(-1))
    j = integral_mean((jerk / JERK_SCALE).square().sum(-1))
    clearance = environment_distance(position, scenes) - BODY_RADIUS
    penetration = (
        F.relu((SAFE_CLEARANCE - clearance) / LENGTH_SCALE).square().flatten(1)
    )
    tail_count = max(
        1, math.ceil(penetration.shape[-1] * LOSS_DEFINITION.clearance_tail_fraction)
    )
    clear = penetration.topk(tail_count, dim=-1).values.mean(-1)
    gate = trajectory_gate_loss(position, scenes)
    pace = pace_loss(decoded, scenes)
    acceleration_weight = weights.acceleration_low * torch.exp(
        math.log(10) * scenes.smoothness
    )
    jerk_weight = weights.jerk_low * torch.exp(math.log(10) * scenes.smoothness)
    weighted_clearance = (
        weights.clearance * clear
        if definition.train_clearance
        else torch.zeros_like(clear)
    )
    task = (
        weights.pace * pace
        + acceleration_weight * a
        + jerk_weight * j
        + weighted_clearance
        + weights.gate * gate
    )
    geo = geometry_losses(decoded, scenes, geometry_loss=geometry_loss)
    return {
        **geo,
        "pace": pace,
        "acceleration": a,
        "jerk": j,
        "clearance_loss": clear,
        "gate": gate,
        "task": task,
        "total": task + weights.geometry * geo["geometry"],
        "weighted_pace": weights.pace * pace,
        "weighted_smoothness": acceleration_weight * a + jerk_weight * j,
        "weighted_clearance": weighted_clearance,
        "weighted_gate": weights.gate * gate,
        "weighted_geometry": weights.geometry * geo["geometry"],
        "pace_error": pace.sqrt(),
        "dense_clearance": clearance,
    }


def valid_lanes(result, losses: dict[str, Tensor]) -> Tensor:
    return (result.status == int(ForwardStatus.RESULT_AVAILABLE)) & torch.isfinite(
        losses["total"]
    )


def batch_loss(result, losses: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    valid = valid_lanes(result, losses)
    if not bool(valid.any()):
        raise RuntimeError("all gate solver lanes failed")
    return losses["total"][valid].mean() + FAILURE_COST * (
        ~valid
    ).double().mean(), valid


def interval_roots(coefficients: np.ndarray) -> list[float]:
    """Real roots on normalized piece time, including shared endpoints once."""
    coefficients = np.trim_zeros(np.asarray(coefficients, dtype=np.float64), "b")
    if len(coefficients) < 2:
        return []
    roots = np.polynomial.polynomial.polyroots(coefficients)
    real = sorted(
        float(np.clip(value.real, 0, 1))
        for value in roots
        if abs(value.imag) < 1e-7 and -1e-8 <= value.real <= 1 + 1e-8
    )
    return [
        value
        for index, value in enumerate(real)
        if index == 0 or abs(value - real[index - 1]) > 1e-7
    ]


def polynomial_wall_collision(
    coefficients: np.ndarray,
    duration: float,
    center: np.ndarray,
    frame: np.ndarray,
    half_extents: np.ndarray,
    body_radius: float = BODY_RADIUS,
) -> bool:
    """Check a polynomial against a body-inflated OBB, including corner arcs.

    Face-coordinate extrema, region transitions and corner-distance extrema
    cover the minima of the piecewise smooth exact box distance. This is
    independent of the dense evaluation grid and catches thin-wall tunneling.
    """
    local = (coefficients * duration ** np.arange(len(coefficients))[:, None]) @ frame
    local[0] -= center @ frame
    derivative = [np.polynomial.polynomial.polyder(local[:, axis]) for axis in range(2)]
    candidates = {0.0, 1.0}
    for axis in range(2):
        critical = [0.0, 1.0, *interval_roots(derivative[axis])]
        values = np.polynomial.polynomial.polyval(critical, local[:, axis])
        if (
            values.min() > half_extents[axis] + body_radius
            or values.max() < -half_extents[axis] - body_radius
        ):
            return False
        candidates.update(critical)
        for side in (-1, 1):
            boundary = local[:, axis].copy()
            boundary[0] -= side * half_extents[axis]
            candidates.update(interval_roots(boundary))
    for side_x in (-1, 1):
        for side_y in (-1, 1):
            x, y = local[:, 0].copy(), local[:, 1].copy()
            x[0] -= side_x * half_extents[0]
            y[0] -= side_y * half_extents[1]
            squared = np.polynomial.polynomial.polyadd(
                np.polynomial.polynomial.polymul(x, x),
                np.polynomial.polynomial.polymul(y, y),
            )
            candidates.update(interval_roots(np.polynomial.polynomial.polyder(squared)))
            inside_switch = side_x * local[:, 0] - side_y * local[:, 1]
            inside_switch[0] += half_extents[1] - half_extents[0]
            candidates.update(interval_roots(inside_switch))
    points = np.polynomial.polynomial.polyval(sorted(candidates), local).T
    residual = np.abs(points) - half_extents
    distances = np.linalg.norm(np.maximum(residual, 0), axis=-1) + np.minimum(
        residual.max(-1), 0
    )
    return bool((distances < body_radius - 1e-8).any())


def gate_traversal(
    result,
    scenes: SceneBatch,
    *,
    body_radius: float | Tensor = BODY_RADIUS,
    collision_method: str = "continuous",
) -> dict[str, list]:
    if collision_method not in ("continuous", "dense"):
        raise ValueError(f"unknown collision method: {collision_method!r}")
    coefficients = result.coefficients.detach().numpy()
    durations = result.durations.detach().numpy()
    if isinstance(body_radius, Tensor):
        radii = body_radius.detach().numpy().reshape(-1)
    else:
        radii = np.full(scenes.batch_size, body_radius)
    dense_collision = None
    if collision_method == "dense":
        points = sample_derivative(result.coefficients, result.durations, 0)[..., :2]
        clearance = (
            environment_distance(points, scenes)
            - torch.as_tensor(radii, dtype=points.dtype)[:, None, None]
        )
        dense_collision = clearance.flatten(1).amin(-1) < 0
    report = {"gate_success": [], "crossing_times": [], "continuous_collision": []}
    for lane in range(scenes.batch_size):
        if int(result.status[lane]) != int(ForwardStatus.RESULT_AVAILABLE):
            report["gate_success"].append(False)
            report["crossing_times"].append([])
            report["continuous_collision"].append(True)
            continue
        events = []
        legal = True
        for gate in range(scenes.gate_centers.shape[1]):
            angle = float(scenes.gate_angles[lane, gate])
            normal = np.array((math.cos(angle), math.sin(angle)))
            transverse = np.array((-normal[1], normal[0]))
            center = scenes.gate_centers[lane, gate].numpy()
            times = []
            elapsed = 0.0
            for piece, duration in enumerate(durations[lane]):
                xy = (
                    coefficients[lane, piece, :, :2]
                    * duration ** np.arange(coefficients.shape[2])[:, None]
                )
                plane = xy @ normal
                plane[0] -= normal @ center
                for root in interval_roots(plane):
                    when = elapsed + root * duration
                    if times and abs(when - times[-1][0]) < 1e-7:
                        continue
                    position = np.polynomial.polynomial.polyval(root, xy)
                    forward = (
                        np.polynomial.polynomial.polyval(
                            root, np.polynomial.polynomial.polyder(plane)
                        )
                        > 1e-8
                    )
                    opening = (
                        abs(transverse @ (position - center)) + radii[lane]
                        <= float(scenes.gate_half_widths[lane, gate]) + 1e-8
                    )
                    times.append((when, bool(forward and opening)))
                elapsed += duration
            legal &= len(times) == 1 and times[0][1]
            events.append(times[0][0] if len(times) == 1 else None)
        legal &= all(value is not None for value in events) and all(
            left < right for left, right in zip(events, events[1:])
        )
        if dense_collision is not None:
            collision = bool(dense_collision[lane])
        else:
            collision = any(
                polynomial_wall_collision(
                    coefficients[lane, piece, :, :2],
                    float(durations[lane, piece]),
                    center.numpy(),
                    frame.numpy(),
                    extent.numpy(),
                    float(radii[lane]),
                )
                for piece in range(coefficients.shape[1])
                for center, frame, extent in zip(
                    scenes.wall_centers[lane],
                    scenes.wall_rotations[lane],
                    scenes.wall_half_extents[lane],
                    strict=True,
                )
            )
        report["gate_success"].append(bool(legal and not collision))
        report["crossing_times"].append(events)
        report["continuous_collision"].append(collision)
    return report


def boundary_headings(result) -> tuple[Tensor, Tensor]:
    """Velocity direction just after departure and just before arrival."""
    offset = 1e-6

    def velocity(coefficients, time):
        powers = torch.arange(1, coefficients.shape[-2], dtype=coefficients.dtype)
        return (
            powers[:, None]
            * coefficients[..., 1:, :]
            * time[..., None, None] ** (powers[:, None] - 1)
        ).sum(-2)

    first = velocity(
        result.coefficients[:, 0], torch.full_like(result.durations[:, 0], offset)
    )
    last = velocity(result.coefficients[:, -1], result.durations[:, -1] - offset)
    return torch.atan2(first[:, 1], first[:, 0]), torch.atan2(last[:, 1], last[:, 0])


def wrapped(angle: Tensor) -> Tensor:
    return torch.atan2(angle.sin(), angle.cos())


def exact_motion(result, *, intervals: int = DENSE_INTERVALS) -> dict[str, Tensor]:
    """Unsmoothed unicycle quantities, including rest-endpoint yaw limits.

    At a prescribed rest endpoint, both ``cross(v, a)`` and ``|v|^2`` vanish.
    Dividing their floating-point residuals produces a meaningless large yaw
    rate.  With nonzero boundary acceleration, the one-sided polynomial limit
    is ``cross(a, jerk) / (2 |a|^2)`` at both ends.  Any other sampled zero-speed
    point remains infinite because its heading rate is genuinely undefined.
    """
    velocity = sample_derivative(
        result.coefficients, result.durations, 1, intervals=intervals
    )
    acceleration = sample_derivative(
        result.coefficients, result.durations, 2, intervals=intervals
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    moving = speed >= EVALUATION_SPEED
    safe = torch.where(moving, speed, torch.ones_like(speed))
    cross = (
        velocity[..., 0] * acceleration[..., 1]
        - velocity[..., 1] * acceleration[..., 0]
    )
    jerk = sample_derivative(
        result.coefficients, result.durations, 3, intervals=intervals
    )
    acceleration_jerk_cross = (
        acceleration[..., 0] * jerk[..., 1] - acceleration[..., 1] * jerk[..., 0]
    )
    endpoint_yaw_rate = 0.5 * acceleration_jerk_cross / acceleration.square().sum(-1)
    low_speed_yaw_rate = torch.where(
        speed > NUMERICAL_SPEED_FLOOR,
        cross / speed.square(),
        torch.full_like(speed, float("inf")),
    )
    low_speed_yaw_rate[:, 0, 0] = endpoint_yaw_rate[:, 0, 0]
    low_speed_yaw_rate[:, -1, -1] = endpoint_yaw_rate[:, -1, -1]
    yaw_rate = cross / safe.square()
    return {
        "speed": speed,
        "moving": moving,
        "tangential": (velocity * acceleration).sum(-1) / safe,
        "centripetal": cross / safe,
        "yaw_rate": yaw_rate,
        "wheel_speeds": torch.stack(
            (
                speed - 0.5 * TRACK_WIDTH * yaw_rate,
                speed + 0.5 * TRACK_WIDTH * yaw_rate,
            ),
            -1,
        ),
        # Low-speed rotation rate without the native smoothing. Rest endpoints
        # use the analytic one-sided limit; unexpected interior stops stay inf.
        "low_speed_yaw_rate": low_speed_yaw_rate,
    }


def gate_pieces(result, crossings: dict[str, list]) -> list[list]:
    """Piece (= corridor) index containing each recorded gate crossing time."""
    durations = result.durations.detach().numpy()
    pieces = []
    for lane, times in enumerate(crossings["crossing_times"]):
        ends = np.cumsum(durations[lane])
        pieces.append(
            [
                None
                if when is None
                else int(
                    min(
                        np.searchsorted(ends, when, side="right"),
                        result.durations.shape[1] - 1,
                    )
                )
                for when in times
            ]
        )
    return pieces


def rectangle_corners(centers, rotations, half_extents):
    """Counter-clockwise corners for the box frames used by the gate task."""
    signs = centers.new_tensor(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    local = signs * half_extents[..., None, :]
    return centers[..., None, :] + torch.einsum("...de,...ve->...vd", rotations, local)


def convex_polygon_distance(left, right):
    """Exact Euclidean distance between batched convex polygons."""
    shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = left.expand(*shape, *left.shape[-2:])
    right = right.expand(*shape, *right.shape[-2:])

    def edges(polygon):
        return torch.roll(polygon, shifts=-1, dims=-2) - polygon

    def axes(polygon):
        delta = edges(polygon)
        normal = perpendicular(delta)
        return normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)

    separating = torch.zeros(shape, dtype=torch.bool, device=left.device)
    for normal in (axes(left), axes(right)):
        left_projection = torch.einsum("...vd,...ed->...ev", left, normal)
        right_projection = torch.einsum("...vd,...ed->...ev", right, normal)
        separating |= (
            (left_projection.amax(-1) < right_projection.amin(-1))
            | (right_projection.amax(-1) < left_projection.amin(-1))
        ).any(-1)

    def vertex_edge_distance(points, polygon):
        start = polygon
        delta = edges(polygon)
        relative = points[..., :, None, :] - start[..., None, :, :]
        fraction = (relative * delta[..., None, :, :]).sum(-1) / delta.square().sum(-1)[
            ..., None, :
        ]
        nearest = (
            start[..., None, :, :]
            + fraction.clamp(0, 1)[..., None] * delta[..., None, :, :]
        )
        return torch.linalg.vector_norm(points[..., :, None, :] - nearest, dim=-1)

    distance = torch.minimum(
        vertex_edge_distance(left, right).amin(dim=(-2, -1)),
        vertex_edge_distance(right, left).amin(dim=(-2, -1)),
    )
    return torch.where(separating, distance, torch.zeros_like(distance))


@torch.no_grad()
def gate_validity(
    decoded, result, scenes, *, intervals=DENSE_INTERVALS, traversal=None
):
    """Return the V1--V3 gate-validity evidence for every lane and box."""
    box_corners = rectangle_corners(
        decoded.box_centers, decoded.rotations, decoded.half_extents
    )
    wall_corners = rectangle_corners(
        scenes.wall_centers, scenes.wall_rotations, scenes.wall_half_extents
    )
    box_wall_distance = convex_polygon_distance(
        box_corners[:, :, None], wall_corners[:, None]
    )
    box_intrusion = (BODY_RADIUS - box_wall_distance).clamp_min(0).amax(-1)
    unsafe_box = box_intrusion > VALIDITY_BOX_PENETRATION

    seed_segments = torch.stack((decoded.anchors[:, :-1], decoded.anchors[:, 1:]), -2)
    seed_wall_distance = convex_polygon_distance(
        seed_segments[:, :, None], wall_corners[:, None]
    )
    seed_intrusion = (BODY_RADIUS - seed_wall_distance).clamp_min(0).amax(dim=(-1, -2))
    seed_collision = seed_intrusion > 0

    points = sample_derivative(
        result.coefficients, result.durations, 0, intervals=intervals
    )[..., :2]
    signed_wall_distance = box_signed_distances(
        points.flatten(1, 2),
        scenes.wall_centers,
        scenes.wall_rotations,
        scenes.wall_half_extents,
    ).reshape(scenes.batch_size, task_for_scenes(scenes).pieces, intervals + 1, -1)
    clearance = signed_wall_distance.amin(-1) - BODY_RADIUS
    flat_clearance = clearance.flatten(1)
    dense_minimum, deepest = flat_clearance.min(-1)
    deepest_piece = torch.div(deepest, intervals + 1, rounding_mode="floor")
    deepest_sample = deepest.remainder(intervals + 1)
    lane = torch.arange(scenes.batch_size)
    deepest_point = points[lane, deepest_piece, deepest_sample]
    deepest_wall = signed_wall_distance[lane, deepest_piece, deepest_sample].argmin(-1)
    selected_corridor = decoded.corridor[lane, deepest_piece]
    face_value = (
        torch.einsum("bfd,bd->bf", selected_corridor[..., :2], deepest_point)
        + selected_corridor[..., 2]
    )
    deepest_inside = face_value.amax(-1) <= VALIDITY_CORRIDOR_TOLERANCE
    if traversal is None:
        traversal = gate_traversal(result, scenes)
    continuous_collision = torch.tensor(
        traversal["continuous_collision"], dtype=torch.bool
    )
    dense_found = dense_minimum < 0
    inside_collision = continuous_collision & dense_found & deepest_inside
    outside_collision = continuous_collision & dense_found & ~deepest_inside
    dense_missed = continuous_collision & ~dense_found

    query_faces = (
        torch.einsum("bkfd,bqd->bkqf", decoded.corridor[..., :2], scenes.unsafe_queries)
        + decoded.corridor[:, :, None, :, 2]
    )
    unsafe_query_inside = (query_faces.amax(-1) <= 0).any(-1)
    return {
        "v1": float(unsafe_box.double().mean()),
        "v2": float(seed_collision.double().mean()),
        "v3": float(inside_collision.double().mean()),
        "valid": bool(
            unsafe_box.double().mean() <= 0.01
            and seed_collision.double().mean() <= 0.01
            and inside_collision.double().mean() <= 0.01
        ),
        "box_wall_distance": box_wall_distance,
        "box_intrusion": box_intrusion,
        "unsafe_box": unsafe_box,
        "unsafe_query_inside": unsafe_query_inside,
        "seed_intrusion": seed_intrusion,
        "seed_collision": seed_collision,
        "continuous_collision": continuous_collision,
        "dense_minimum_clearance": dense_minimum,
        "deepest_piece": deepest_piece,
        "deepest_point": deepest_point,
        "deepest_wall": deepest_wall,
        "deepest_face_value": face_value.amax(-1),
        "inside_collision": inside_collision,
        "outside_collision": outside_collision,
        "dense_missed": dense_missed,
        "traversal": traversal,
    }


def constraint_violations(
    result,
    decoded: DecodedAction,
    *,
    intervals: int = DENSE_INTERVALS,
    corridor_margin: float = 0.0,
) -> dict[str, Tensor]:
    samples = drive_samples(result.coefficients, result.durations, intervals=intervals)
    corridor = corridor_with_margin(decoded.corridor, corridor_margin)
    faces = (
        torch.einsum("bkfd,bkqd->bkqf", corridor[..., :2], samples["position"])
        + corridor[:, :, None, :, 2]
    )
    violations = {"corridor": faces}
    for index, name in enumerate(CAP_NAMES):
        limit = decoded.caps[:, index, None, None]
        violations[name] = (samples[name].square() - limit.square()) / (2 * limit)
    wheels = samples["wheel_speeds"]
    violations["wheel_speeds"] = (wheels.square() - WHEEL_LIMIT**2) / (2 * WHEEL_LIMIT)
    return violations


@dataclass(frozen=True)
class PerturbationBatch:
    scenes: SceneBatch
    body_radius: Tensor
    repeats: int
    source_batch_size: int
    speed_limit: float = 0.95 * PHYSICAL_CAPS[0]
    wheel_limit: float = 0.95 * WHEEL_LIMIT
    scale: float = 1.0


@dataclass(frozen=True)
class GeometryResult:
    coefficients: Tensor
    durations: Tensor
    status: Tensor


def sample_perturbations(
    scenes: SceneBatch,
    *,
    generator: torch.Generator,
    repeats: int = 8,
    scale: float = 1.0,
) -> PerturbationBatch:
    """Draw fixed model-error environments around nominal planned scenes."""
    if repeats < 1:
        raise ValueError("perturbation repeats must be positive")
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("perturbation scale must be finite and in [0, 1]")
    task = task_for_scenes(scenes)
    batch_size, gates = scenes.batch_size, task.gates

    def uniform(*shape):
        return torch.rand(
            (batch_size, repeats, *shape),
            generator=generator,
            dtype=torch.float64,
        )

    shift_length = scale * (0.02 + 0.02 * uniform(gates))
    shift_angle = 2 * math.pi * uniform(gates)
    shift = shift_length[..., None] * torch.stack(
        (shift_angle.cos(), shift_angle.sin()), -1
    )
    centers = scenes.gate_centers[:, None] + shift
    half_widths = scenes.gate_half_widths[:, None] - scale * (
        0.02 + 0.02 * uniform(gates)
    )
    angles = scenes.gate_angles[:, None].expand(-1, repeats, -1)
    thickness = scenes.gate_thickness[:, None].expand(-1, repeats, -1)
    indices = torch.arange(batch_size).repeat_interleave(repeats)
    expanded = scenes.select(indices)
    flat_centers = centers.flatten(0, 1)
    flat_angles = angles.flatten(0, 1)
    flat_widths = half_widths.flatten(0, 1)
    flat_thickness = thickness.flatten(0, 1)
    wall_centers, wall_rotations, wall_extents = build_wall_geometry(
        task, flat_centers, flat_angles, flat_widths, flat_thickness
    )
    environments = replace(
        expanded,
        gate_centers=flat_centers,
        gate_angles=flat_angles,
        gate_half_widths=flat_widths,
        gate_thickness=flat_thickness,
        wall_centers=wall_centers,
        wall_rotations=wall_rotations,
        wall_half_extents=wall_extents,
    )
    body_draw = uniform()
    body_radius = (
        BODY_RADIUS + 0.01 + 0.01 * body_draw
        if scale == 1.0
        else BODY_RADIUS + scale * (0.01 + 0.01 * body_draw)
    ).flatten()
    return PerturbationBatch(
        environments,
        body_radius,
        repeats,
        batch_size,
        speed_limit=(1 - 0.05 * scale) * PHYSICAL_CAPS[0],
        wheel_limit=(1 - 0.05 * scale) * WHEEL_LIMIT,
        scale=scale,
    )


def expand_result(result, repeats: int) -> GeometryResult:
    return GeometryResult(
        result.coefficients.repeat_interleave(repeats, 0),
        result.durations.repeat_interleave(repeats, 0),
        result.status.repeat_interleave(repeats, 0),
    )


def evaluate_perturbations(
    result,
    perturbations: PerturbationBatch,
    *,
    final: bool,
) -> tuple[dict, dict]:
    """Evaluate one planned trajectory against fixed perturbed environments."""
    expanded = expand_result(result, perturbations.repeats)
    dense = gate_traversal(
        expanded,
        perturbations.scenes,
        body_radius=perturbations.body_radius,
        collision_method="dense",
    )

    points = sample_derivative(expanded.coefficients, expanded.durations, 0)[..., :2]
    clearance = (
        environment_distance(points, perturbations.scenes)
        - (perturbations.body_radius[:, None, None])
    )
    valid = expanded.status == int(ForwardStatus.RESULT_AVAILABLE)
    minimum_clearance = torch.where(
        valid,
        clearance.flatten(1).amin(-1),
        torch.full_like(valid, -torch.inf, dtype=clearance.dtype),
    ).detach()

    def reduce_crossings(prefix: str, crossings: dict[str, list]) -> dict:
        collision = torch.tensor(crossings["continuous_collision"], dtype=torch.bool)
        success = torch.tensor(crossings["gate_success"], dtype=torch.bool)
        robust = success.reshape(
            perturbations.source_batch_size, perturbations.repeats
        ).all(-1)
        return {
            f"{prefix}_collision_fraction": float(collision.double().mean()),
            f"{prefix}_gate_success_fraction": float(success.double().mean()),
            f"{prefix}_robust_gate_success_fraction": float(robust.double().mean()),
        }

    summary = {
        **reduce_crossings("dense", dense),
        "minimum_clearance_p05": float(torch.quantile(minimum_clearance, 0.05)),
        "clearance_min": float(minimum_clearance.min()),
        "environments": len(minimum_clearance),
        "repeats": perturbations.repeats,
    }
    continuous = None
    if final:
        continuous = gate_traversal(
            expanded,
            perturbations.scenes,
            body_radius=perturbations.body_radius,
            collision_method="continuous",
        )
        summary.update(reduce_crossings("continuous", continuous))

    motion = {name: value.detach() for name, value in exact_motion(result).items()}
    moving = (
        motion["moving"]
        & (result.status == int(ForwardStatus.RESULT_AVAILABLE))[:, None, None]
    ).repeat_interleave(perturbations.repeats, 0)
    speed = motion["speed"].repeat_interleave(perturbations.repeats, 0)
    wheels = (
        motion["wheel_speeds"]
        .abs()
        .amax(-1)
        .repeat_interleave(perturbations.repeats, 0)
    )
    for name, values, limit in (
        ("speed", speed, perturbations.speed_limit),
        ("wheel", wheels, perturbations.wheel_limit),
    ):
        excess = values[moving] - limit
        summary[f"physical_{name}_max_violation"] = float(excess.clamp_min(0).max())
        summary[f"physical_{name}_violation_fraction"] = float(
            (excess > 0).double().mean()
        )

    lanes = {
        "dense_gate_success": dense["gate_success"],
        "dense_collision": dense["continuous_collision"],
        "continuous_gate_success": None
        if continuous is None
        else continuous["gate_success"],
        "continuous_collision": None
        if continuous is None
        else continuous["continuous_collision"],
        "minimum_clearance": minimum_clearance.tolist(),
    }
    return summary, lanes


def evaluate(
    policy: GatePolicy,
    setup: TrajectorySetup,
    scenes: SceneBatch,
    *,
    weights: LossWeights | None = None,
    train_clearance: bool | None = None,
    geometry_loss: str = "current",
    capsule_radius: float | None = None,
    corridor_activation: float | None = None,
    transform=None,
    perturbations: PerturbationBatch | None = None,
    final_perturbations: bool = False,
) -> tuple[dict, dict]:
    task = task_for_scenes(scenes)
    if weights is None:
        weights = task_weights(task)
    started = time.perf_counter()
    with torch.no_grad():
        decoded = decode_action(
            policy(scenes.features()), scenes, capsule_radius=capsule_radius
        )
        if transform is not None:
            decoded = transform(decoded)
        result = setup.solver.solve(setup.bind(scenes, decoded))
        solve_seconds = time.perf_counter() - started
        losses = trajectory_losses(
            result,
            decoded,
            scenes,
            weights=weights,
            train_clearance=train_clearance,
            geometry_loss=geometry_loss,
        )
        valid = valid_lanes(result, losses)
        summary = {
            "failure_fraction": float((~valid).double().mean()),
            "valid_lanes": int(valid.sum()),
            "lanes": scenes.batch_size,
            "solve_wall_seconds": solve_seconds,
        }
        if not bool(valid.any()):
            return summary, {"valid": valid.tolist()}
        for name in (
            "total",
            "task",
            "pace_error",
            "acceleration",
            "jerk",
            "clearance_loss",
            "gate",
            "exclude",
            "fill",
            "seed",
            "seed_clearance",
            "seed_opening",
            "fill_coverage",
            "size",
            "room",
            "weighted_pace",
            "weighted_smoothness",
            "weighted_clearance",
            "weighted_gate",
            "weighted_geometry",
        ):
            summary[name] = float(losses[name][valid].mean())
        for name in ("unsafe", "overrun", "underfill", "local_fill", "global_fill"):
            if name in losses:
                summary[name] = float(losses[name][valid].mean())
        clearance = losses["dense_clearance"][valid]
        summary.update(
            clearance_p05=float(torch.quantile(clearance.flatten(1).amin(-1), 0.05)),
            sample_clearance_p05=float(torch.quantile(clearance.flatten(), 0.05)),
            clearance_min=float(clearance.min()),
            collision_fraction=float(
                (clearance.flatten(1).amin(-1) < 0).double().mean()
            ),
        )
        violations = constraint_violations(
            result, decoded, corridor_margin=setup.corridor_margin
        )
        for name, violation in violations.items():
            values = violation[valid]
            summary[f"{name}_max_violation"] = float(values.clamp_min(0).max())
            summary[f"{name}_violation_fraction"] = float((values > 0).double().mean())
        # Physical limits use the exact unicycle on moving samples (Q18).
        motion = exact_motion(result)
        moving = motion["moving"][valid]
        for name, values, limit in (
            ("wheel_speeds", motion["wheel_speeds"].abs().amax(-1), WHEEL_LIMIT),
            (CAP_NAMES[0], motion["speed"], PHYSICAL_CAPS[0]),
            (CAP_NAMES[1], motion["centripetal"].abs(), PHYSICAL_CAPS[1]),
            (CAP_NAMES[2], motion["tangential"].abs(), PHYSICAL_CAPS[2]),
        ):
            excess = values[valid][moving] - limit
            summary[f"physical_{name}_max_violation"] = float(excess.clamp_min(0).max())
            summary[f"physical_{name}_violation_fraction"] = float(
                (excess > 0).double().mean()
            )
        slow = ~moving
        summary["low_speed_sample_fraction"] = float(slow.double().mean())
        summary["low_speed_max_abs_yaw_rate"] = (
            float(motion["low_speed_yaw_rate"][valid][slow].abs().max())
            if bool(slow.any())
            else 0.0
        )
        summary["low_speed_max_wheel_speed"] = (
            float(
                (
                    motion["speed"][valid][slow]
                    + 0.5
                    * TRACK_WIDTH
                    * motion["low_speed_yaw_rate"][valid][slow].abs()
                ).max()
            )
            if bool(slow.any())
            else 0.0
        )
        start_heading, end_heading = boundary_headings(result)
        summary["boundary_heading_error_max"] = float(
            torch.maximum(
                wrapped(start_heading - scenes.start_yaw).abs(),
                wrapped(end_heading - scenes.goal_yaw).abs(),
            )[valid].max()
        )
        corridor_slack = -violations["corridor"].amax(-1)
        if corridor_activation is None:
            corridor_activation = task.corridor_activation
        corridor_active = corridor_slack < corridor_activation
        summary["corridor_active_sample_fraction"] = float(
            corridor_active[valid].double().mean()
        )
        narrow = decoded.half_extents[..., 1]
        summary["narrow_half_extent_median"] = [
            float(value) for value in narrow[valid].median(0).values
        ]
        normalized_caps = decoded.caps / decoded.caps.new_tensor(PHYSICAL_CAPS)
        for index, name in enumerate(CAP_NAMES):
            summary[f"{name}_saturation_fraction"] = float(
                (
                    (normalized_caps[valid, index] < 0.25 + 0.01 * 0.75)
                    | (normalized_caps[valid, index] > 1 - 0.01 * 0.75)
                )
                .double()
                .mean()
            )
        diagnostics = result.diagnostics
        summary.update(
            stationarity_mean=float(diagnostics.stationarity[valid].mean()),
            stationarity_max=float(diagnostics.stationarity[valid].max()),
            maximum_iterations_fraction=float(
                (
                    diagnostics.optimizer_status[valid]
                    == int(OptimizerStatus.MAXIMUM_ITERATIONS)
                )
                .double()
                .mean()
            ),
            objective_evaluations_mean=float(
                diagnostics.objective_evaluations[valid].double().mean()
            ),
        )
        crossings = gate_traversal(result, scenes)
        validity = gate_validity(decoded, result, scenes, traversal=crossings)
        summary.update(
            validity_v1=validity["v1"],
            validity_v2=validity["v2"],
            validity_v3=validity["v3"],
            corridor_valid=validity["valid"],
            box_intrusion_max=float(validity["box_intrusion"].max()),
            box_intrusion_p95=float(torch.quantile(validity["box_intrusion"], 0.95)),
            seed_intrusion_max=float(validity["seed_intrusion"].max()),
            seed_intrusion_p95=float(torch.quantile(validity["seed_intrusion"], 0.95)),
        )
        pieces = gate_pieces(result, crossings)
        gate_active, gate_narrow = [], []
        for lane, lane_pieces in enumerate(pieces):
            if not bool(valid[lane]):
                continue
            known = [piece for piece in lane_pieces if piece is not None]
            gate_active.append(
                any(bool(corridor_active[lane, piece].any()) for piece in known)
            )
            gate_narrow.extend(float(narrow[lane, piece]) for piece in known)
        summary["gate_corridor_active_scene_fraction"] = (
            float(np.mean(gate_active)) if gate_active else 0.0
        )
        summary["gate_corridor_narrow_half_extent_median"] = (
            float(np.median(gate_narrow)) if gate_narrow else None
        )
        summary["gate_success_fraction"] = float(np.mean(crossings["gate_success"]))
        summary["continuous_collision_fraction"] = float(
            np.mean(crossings["continuous_collision"])
        )
        lanes = {
            **crossings,
            "valid": valid.tolist(),
            "task": losses["task"].tolist(),
            "total": losses["total"].tolist(),
            "dense_minimum_clearance": losses["dense_clearance"]
            .flatten(1)
            .amin(-1)
            .tolist(),
            "stationarity": diagnostics.stationarity.tolist(),
            "optimizer_status": diagnostics.optimizer_status.tolist(),
            "objective_evaluations": diagnostics.objective_evaluations.tolist(),
        }
        if perturbations is not None:
            perturbed_summary, perturbed_lanes = evaluate_perturbations(
                result, perturbations, final=final_perturbations
            )
            summary["perturbed"] = perturbed_summary
            lanes["perturbed"] = perturbed_lanes
    return summary, lanes


def save_json(path: Path, value) -> None:
    """Finite JSON artifacts keep failures explicit instead of emitting NaN."""

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(part) for part in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")


def warmup_policy(
    *,
    seed: int,
    steps: int = 300,
    batch_size: int = 48,
    checkpoint: Path | None = None,
    weights: LossWeights = WARMUP_WEIGHTS,
    task: TaskConfig = GATE2,
    geometry_loss: str | None = None,
    capsule_radius: float | None = None,
) -> dict:
    requested_geometry_loss = geometry_loss
    requested_capsule_radius = capsule_radius
    if geometry_loss is None:
        geometry_loss = "current"
    if capsule_radius is None:
        capsule_radius = task.capsule_radius
    torch.manual_seed(seed)
    policy = (GatePolicy() if task is GATE2 else GatePolicy(task)).double()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=WARMUP_LEARNING_RATE)
    history = []
    for step in range(steps):
        scenes = sample_scenes(
            batch_size,
            generator=generator,
            task=task,
            geometry_loss=geometry_loss,
            capsule_radius=capsule_radius,
        )
        decoded = decode_action(
            policy(scenes.features()), scenes, capsule_radius=capsule_radius
        )
        geometry = geometry_losses(decoded, scenes, geometry_loss=geometry_loss)
        loss = (
            weights.geometry * geometry["geometry"]
            + weights.pace * pace_loss(decoded, scenes)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in policy.parameters()
        ):
            raise RuntimeError("nonfinite gate geometry warm-up gradient")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                **{
                    name: float(value.detach().mean())
                    for name, value in geometry.items()
                },
            }
        )
    payload = {
        "policy": policy.state_dict(),
        "seed": seed,
        "warmup_steps": steps,
        "batch_size": batch_size,
        "scene_generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "weights": asdict(weights),
        "task": task.name,
        "loss_definition": asdict(loss_definition(task)),
        "history": history,
    }
    if requested_geometry_loss is not None:
        payload["geometry_loss"] = geometry_loss
    if requested_capsule_radius is not None:
        payload["capsule_radius"] = capsule_radius
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint)
        save_json(
            checkpoint.with_suffix(".json"),
            {
                key: value
                for key, value in payload.items()
                if key not in ("policy", "scene_generator_state", "torch_rng_state")
            },
        )
    return payload


def diagnose_backward(
    setup: TrajectorySetup,
    result,
    state,
    scenes: SceneBatch,
    decoded: DecodedAction,
    weights: LossWeights,
    train_clearance: bool,
    geometry_loss: str,
) -> dict:
    names = p.ResultSeeds._fields
    leaves = tuple(
        getattr(result, name).detach().requires_grad_(True) for name in names
    )
    diagnostic = replace(result, **dict(zip(names, leaves, strict=True)))
    fixed = replace(
        decoded,
        **{
            field.name: getattr(decoded, field.name).detach()
            for field in fields(decoded)
        },
    )
    loss, _ = batch_loss(
        diagnostic,
        trajectory_losses(
            diagnostic,
            fixed,
            scenes,
            weights=weights,
            train_clearance=train_clearance,
            geometry_loss=geometry_loss,
        ),
    )
    seeds = torch.autograd.grad(loss, leaves, allow_unused=True)
    pulled = setup.solver.pullback(
        state,
        p.ResultSeeds(
            **{
                name: None if seed is None else seed.detach().contiguous()
                for name, seed in zip(names, seeds, strict=True)
            }
        ),
    )
    return {
        field.name: getattr(pulled.diagnostics, field.name).tolist()
        for field in fields(pulled.diagnostics)
    }


def run_training(
    *,
    constraints: str = "hinge",
    steps: int = 2000,
    warmup_steps: int = 300,
    batch_size: int = 48,
    seed: int = 42,
    validation_size: int = 128,
    checkpoint_every: int | None = None,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    weights: LossWeights | None = None,
    train_clearance: bool | None = None,
    calibrations: tuple[Calibration, ...] | None = None,
    task: TaskConfig = GATE2,
    corridor_margin: float = 0.0,
    geometry_loss: str | None = None,
    capsule_radius: float | None = None,
    output_dir: str | Path | None = None,
    warmup_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    log_every: int = 25,
) -> dict:
    requested_geometry_loss = geometry_loss
    requested_capsule_radius = capsule_radius
    if checkpoint_every is None:
        checkpoint_every = 250 if task is GATE2 else 200
    if weights is None:
        weights = task_weights(task)
    if train_clearance is None:
        train_clearance = task.train_clearance
    if calibrations is None:
        calibrations = task_calibrations(task)
    if geometry_loss is None:
        geometry_loss = "current"
    if capsule_radius is None:
        capsule_radius = task.capsule_radius
    if (
        steps < 0
        or warmup_steps < 0
        or batch_size < 1
        or validation_size < 1
        or checkpoint_every < 1
        or learning_rate <= 0
        or corridor_margin < 0
        or geometry_loss not in GEOMETRY_LOSS_KINDS
        or not math.isfinite(capsule_radius)
        or capsule_radius <= 0
    ):
        raise ValueError("invalid training budget or learning rate")
    if corridor_margin and constraints != "hinge":
        raise ValueError("corridor margin is only defined for hinge constraints")
    group = "hinge-margin" if corridor_margin else constraints
    root = (
        Path(output_dir)
        if output_dir
        else (
            Path("artifacts/gate2/runs")
            if task is GATE2
            else Path("artifacts/sgate3/runs")
        )
        / f"{group}_seed{seed}"
    )
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() and resume is None:
        raise FileExistsError(
            f"run exists: {root}; choose another output directory or --resume"
        )
    settings = {
        "task": task.name,
        "group": group,
        "constraints": constraints,
        "corridor_margin": corridor_margin,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "seed": seed,
        "validation_seed": seed + 1,
        "validation_size": validation_size,
        "checkpoint_every": checkpoint_every,
        "maximum_iterations": maximum_iterations,
        "learning_rate": learning_rate,
        "weights": asdict(weights),
        "loss_definition": asdict(
            replace(loss_definition(task), train_clearance=train_clearance)
        ),
        "warmup_weights": asdict(WARMUP_WEIGHTS),
        "calibrations": [asdict(value) for value in calibrations],
        "warmup_checkpoint": str(warmup_checkpoint)
        if warmup_checkpoint
        else str(root / "warmup.pt"),
    }
    if requested_geometry_loss is not None:
        settings["geometry_loss"] = geometry_loss
    if requested_capsule_radius is not None:
        settings["capsule_radius"] = capsule_radius
    if resume is not None:
        payload = torch.load(resume, weights_only=True)
        previous_settings = payload["settings"]
        comparable_settings = {**settings, "steps": previous_settings["steps"]}
        if previous_settings != comparable_settings or payload["step"] > steps:
            raise ValueError("resume settings differ from the checkpoint")
    else:
        payload = (
            torch.load(warmup_checkpoint, weights_only=True)
            if warmup_checkpoint
            else warmup_policy(
                seed=seed,
                steps=warmup_steps,
                batch_size=batch_size,
                checkpoint=root / "warmup.pt",
                weights=WARMUP_WEIGHTS,
                task=task,
                geometry_loss=requested_geometry_loss,
                capsule_radius=requested_capsule_radius,
            )
        )
        if payload["seed"] != seed:
            raise ValueError("warm-up checkpoint seed differs from training seed")
        if payload.get("task", "gate2") != task.name:
            raise ValueError("warm-up checkpoint task differs from training task")
        if (
            payload.get("geometry_loss", "current") != geometry_loss
            or payload.get("capsule_radius", task.capsule_radius) != capsule_radius
        ):
            raise ValueError("warm-up checkpoint geometry differs from training")
    policy = (GatePolicy() if task is GATE2 else GatePolicy(task)).double()
    policy.load_state_dict(payload["policy"])
    generator = torch.Generator()
    generator.set_state(payload["scene_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history, evaluations, first_step = [], [], 0
    if resume is not None:
        optimizer.load_state_dict(payload["optimizer"])
        first_step = payload["step"]
        history = payload["history"]
        evaluations = payload["evaluations"]
    setup = prepare_problem(
        batch_size,
        constraints=constraints,
        maximum_iterations=maximum_iterations,
        calibrations=calibrations,
        task=task,
        corridor_margin=corridor_margin,
    )
    validation_setup = prepare_problem(
        validation_size,
        constraints=constraints,
        maximum_iterations=maximum_iterations,
        calibrations=calibrations,
        task=task,
        corridor_margin=corridor_margin,
    )
    validation = sample_scenes(
        validation_size,
        generator=torch.Generator().manual_seed(seed + 1),
        task=task,
        geometry_loss=geometry_loss,
        capsule_radius=capsule_radius,
    )
    perturbations = (
        sample_perturbations(
            validation,
            generator=torch.Generator().manual_seed(seed + 10001),
            repeats=task.perturbations,
        )
        if task.perturbations
        else None
    )
    save_json(root / "config.json", {**settings, "plan": setup.plan.explain()})

    def save_checkpoint(step):
        torch.save(
            {
                "step": step,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scene_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "settings": settings,
                "history": history,
                "evaluations": evaluations,
            },
            root / f"checkpoint_{step:06d}.pt",
        )

    if not evaluations:
        initial, lanes = evaluate(
            policy,
            validation_setup,
            validation,
            weights=weights,
            train_clearance=train_clearance,
            geometry_loss=geometry_loss,
            capsule_radius=capsule_radius,
            corridor_activation=calibrations[0].activation,
            perturbations=perturbations,
        )
        evaluations.append({"step": first_step, **initial})
        save_checkpoint(first_step)
        save_json(root / "validation_lanes.json", lanes)
    started = time.perf_counter()
    outcome = "completed"
    rejection = None
    for step in range(first_step, steps):
        scenes = sample_scenes(
            batch_size,
            generator=generator,
            task=task,
            geometry_loss=geometry_loss,
            capsule_radius=capsule_radius,
        )
        decoded = decode_action(
            policy(scenes.features()), scenes, capsule_radius=capsule_radius
        )
        call_started = time.perf_counter()
        result, state = setup.solver.solve_with_state(setup.bind(scenes, decoded))
        forward_seconds = time.perf_counter() - call_started
        losses = trajectory_losses(
            result,
            decoded,
            scenes,
            weights=weights,
            train_clearance=train_clearance,
            geometry_loss=geometry_loss,
        )
        if not bool(valid_lanes(result, losses).any()):
            outcome = "all_lanes_failed"
            rejection = {
                "attempted_step": step + 1,
                "forward_status": result.status.tolist(),
            }
            break
        loss, valid = batch_loss(result, losses)
        optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_started
        norm = torch.nn.utils.get_total_norm(
            [
                parameter.grad
                for parameter in policy.parameters()
                if parameter.grad is not None
            ]
        )
        if not bool(torch.isfinite(norm)):
            outcome = "nonfinite_parameter_gradient"
            rejection = {
                "attempted_step": step + 1,
                "forward_status": result.status.tolist(),
                "backward": diagnose_backward(
                    setup,
                    result,
                    state,
                    scenes,
                    decoded,
                    weights,
                    train_clearance,
                    geometry_loss,
                ),
            }
            break
        torch.nn.utils.clip_grads_with_norm_(list(policy.parameters()), 10, norm)
        optimizer.step()
        record = {
            "step": step + 1,
            "loss": float(loss.detach()),
            "failure_fraction": float((~valid).double().mean()),
            "gradient_norm": float(norm),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "objective_evaluations_mean": float(
                result.diagnostics.objective_evaluations[valid].double().mean()
            ),
            "maximum_iterations_fraction": float(
                (
                    result.diagnostics.optimizer_status[valid]
                    == int(OptimizerStatus.MAXIMUM_ITERATIONS)
                )
                .double()
                .mean()
            ),
            "stationarity_mean": float(result.diagnostics.stationarity[valid].mean()),
            **{
                name: float(losses[name][valid].detach().mean())
                for name in (
                    "task",
                    "pace_error",
                    "acceleration",
                    "jerk",
                    "clearance_loss",
                    "gate",
                    "geometry",
                    "exclude",
                    "fill",
                    "seed_clearance",
                    "seed_opening",
                    "size",
                    "room",
                )
            },
        }
        for name in ("unsafe", "overrun", "underfill", "local_fill", "global_fill"):
            if name in losses:
                record[name] = float(losses[name][valid].detach().mean())
        history.append(record)
        if step == first_step or (step + 1) % log_every == 0 or step + 1 == steps:
            print(
                f"{constraints} seed={seed} step={step + 1}/{steps} task={record['task']:.5f} loss={record['loss']:.5f} failures={record['failure_fraction']:.3f}",
                flush=True,
            )
            save_json(root / "progress.json", record)
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            summary, lanes = evaluate(
                policy,
                validation_setup,
                validation,
                weights=weights,
                train_clearance=train_clearance,
                geometry_loss=geometry_loss,
                capsule_radius=capsule_radius,
                corridor_activation=calibrations[0].activation,
                perturbations=perturbations,
            )
            evaluations.append({"step": step + 1, **summary})
            save_checkpoint(step + 1)
            save_json(root / "validation_lanes.json", lanes)
            save_json(
                root / "training_history.json",
                {"history": history, "evaluations": evaluations},
            )
    completed = history[-1]["step"] if history else first_step
    save_checkpoint(completed)
    if rejection is not None:
        save_json(root / "rejection.json", rejection)
    summary, lanes = evaluate(
        policy,
        validation_setup,
        validation,
        weights=weights,
        train_clearance=train_clearance,
        geometry_loss=geometry_loss,
        capsule_radius=capsule_radius,
        corridor_activation=calibrations[0].activation,
        perturbations=perturbations,
        final_perturbations=True,
    )
    final = {
        "outcome": outcome,
        "completed_steps": completed,
        "run_dir": str(root),
        "training_wall_seconds": time.perf_counter() - started,
        **summary,
    }
    save_json(
        root / "training_history.json",
        {"history": history, "evaluations": evaluations, "final": final},
    )
    save_json(root / "validation_lanes.json", lanes)
    save_json(root / "summary.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_constraint_argument(parser)
    parser.add_argument("--task", choices=TASKS, default="gate2")
    parser.add_argument("--corridor-margin", type=float, default=0.0)
    parser.add_argument("--geometry-loss", choices=GEOMETRY_LOSS_KINDS)
    parser.add_argument("--capsule-radius", type=float)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--maximum-iterations", type=int, default=MAXIMUM_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = run_training(
        constraints=args.constraints,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch,
        seed=args.seed,
        validation_size=args.validation_size,
        checkpoint_every=args.checkpoint_every,
        maximum_iterations=args.maximum_iterations,
        learning_rate=args.learning_rate,
        task=TASKS[args.task],
        corridor_margin=args.corridor_margin,
        geometry_loss=args.geometry_loss,
        capsule_radius=args.capsule_radius,
        output_dir=args.output_dir,
        warmup_checkpoint=args.warmup_checkpoint,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))
    if summary["outcome"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
