"""End-to-end CPU training of a small polytope-generating policy.

The policy receives structured 2D scene parameters (no image input), decodes a
77-dimensional simplified V9 action, and supplies a planar corridor to the
3D CPU solver.  The native solver allocates the learned total duration across
four pieces; PyTorch then backpropagates goal, length, smoothness, analytic
ESDF, learned-polytope obstacle exclusion, and corridor regularization losses
into the policy. Strict IFT publishes NaN solver gradients for rejected
endpoints; direct losses on policy outputs retain their own gradients.

Run::

    .venv/bin/python examples/cpu_polytope_training.py --steps 300

The default run is written below ``artifacts/cpu_polytope_training``.
The loss and trajectory plots are rendered with matplotlib.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

_MATPLOTLIB_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / ".matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from constraints import Calibration, add_constraint_argument, term  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from pivopt import (  # noqa: E402
    BackwardStatus,
    ForwardStatus,
    OptimizerStatus,
    Result,
)
from pivopt import problem as p  # noqa: E402
from pivopt.torch import Solver, build_solver  # noqa: E402

PIECE_COUNT = 4
FACE_COUNT = 16
ACTION_DIM = 77
SAFETY_MARGIN = 0.25
TOTAL_DURATION_MIN = 1.8
TOTAL_DURATION_MAX = 4.0
TAIL_VELOCITY_MAX = 2.0
TAIL_ACCELERATION_MAX = 5.0
LONGITUDINAL_RESIDUAL_MAX = 0.25
LATERAL_RESIDUAL_MAX = 2.0
SLACK_MAX = 0.75
CLEARANCE_MARGIN = 0.25
POLYTOPE_OBSTACLE_WEIGHT = 0.2
POLYTOPE_OBSTACLE_SAMPLE_COUNT = 32
POLYTOPE_OBSTACLE_MARGIN = CLEARANCE_MARGIN
POLYTOPE_FACE_TEMPERATURE = 16.0
POLYTOPE_MEMBERSHIP_SHARPNESS = 16.0
LEARNING_RATE = 5.0e-5
SOLVER_COST_WEIGHT = 0.0
MAXIMUM_ITERATIONS = 256
GRADIENT_TOLERANCE = 1.0e-3
NATIVE_OPTIMIZER = "lbfgs"

_PATH_SLICE = slice(0, 6)
_TOTAL_DURATION_INDEX = 6
_TAIL_VELOCITY_SLICE = slice(7, 9)
_TAIL_ACCELERATION_SLICE = slice(9, 11)
_SLACK_SLICE = slice(11, 75)
_TERMINAL_RESIDUAL_SLICE = slice(75, ACTION_DIM)
_FACE_ANGLES = torch.arange(FACE_COUNT, dtype=torch.float64) * (
    2.0 * math.pi / FACE_COUNT
)
FACE_DIRECTIONS = torch.stack(
    (_FACE_ANGLES.cos(), _FACE_ANGLES.sin(), torch.zeros_like(_FACE_ANGLES)), dim=-1
)


@dataclass(frozen=True)
class SceneBatch:
    start_xy: Tensor
    goal_xy: Tensor
    circle_centers_xy: Tensor
    circle_radii: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.start_xy.shape[0])

    def features(self) -> Tensor:
        """Return normalized structured inputs in the fixed 11D chart."""
        margin = torch.full_like(self.circle_radii[:, :1], SAFETY_MARGIN)
        return torch.cat(
            (
                self.start_xy / 4.0,
                self.goal_xy / 4.0,
                self.circle_centers_xy.reshape(self.batch_size, 4) / 1.5,
                self.circle_radii,
                margin,
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class DecodedAction:
    waypoints: Tensor
    terminal_position: Tensor
    total_duration: Tensor
    tail_state: Tensor
    corridor_interiors: Tensor
    corridor_halfspaces: Tensor
    slack: Tensor
    corridor_b: Tensor


class PolytopePolicy(nn.Module):
    """Small deterministic scene-to-action network."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(11, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, ACTION_DIM),
        )
        # Zero action is a straight, finite, symmetric initial corridor.
        output_layer = cast(nn.Linear, self.network[-1])
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(cast(Tensor, output_layer.bias))

    def forward(self, features: Tensor) -> Tensor:
        action = self.network(features)
        if action.shape[-1] != ACTION_DIM:
            raise RuntimeError("policy emitted an invalid action width")
        return action


def sample_scenes(
    batch_size: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> SceneBatch:
    """Sample feasible two-circle scenes with deterministic rejection sampling."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    accepted: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    remaining = batch_size
    while remaining:
        count = max(remaining * 3, 8)

        def uniform(shape):
            return torch.rand(shape, generator=generator, dtype=dtype, device=device)

        start = torch.cat(
            (-4.5 + uniform((count, 1)), -1.0 + 2.0 * uniform((count, 1))), dim=-1
        )
        goal = torch.cat(
            (3.5 + uniform((count, 1)), -1.0 + 2.0 * uniform((count, 1))), dim=-1
        )
        centers = -1.0 + 2.0 * uniform((count, 2, 2))
        radii = 0.45 + 0.40 * uniform((count, 2))
        center_distance = torch.linalg.vector_norm(
            centers[:, 0] - centers[:, 1], dim=-1
        )
        start_clearance = torch.linalg.vector_norm(centers - start[:, None], dim=-1)
        goal_clearance = torch.linalg.vector_norm(centers - goal[:, None], dim=-1)
        valid = center_distance > radii.sum(-1) + 0.20
        valid &= (start_clearance > radii + 0.45).all(-1)
        valid &= (goal_clearance > radii + 0.45).all(-1)
        indices = torch.nonzero(valid, as_tuple=False).flatten()[:remaining]
        if indices.numel() == 0:
            continue
        accepted.append(
            (
                start[indices],
                goal[indices],
                centers[indices],
                radii[indices],
            )
        )
        remaining -= int(indices.numel())
    start, goal, centers, radii = (
        torch.cat(values, dim=0) for values in zip(*accepted)
    )
    return SceneBatch(start, goal, centers, radii)


def analytic_esdf(points_xy: Tensor, scenes: SceneBatch) -> Tensor:
    """Exact signed distance to the union of the two circular obstacles."""
    distance = (
        torch.linalg.vector_norm(
            points_xy[..., None, :] - scenes.circle_centers_xy[:, None, :, :], dim=-1
        )
        - scenes.circle_radii[:, None, :]
    )
    return distance.amin(dim=-1)


def _sample_obstacle_points(
    scenes: SceneBatch, *, samples: int = POLYTOPE_OBSTACLE_SAMPLE_COUNT
) -> Tensor:
    """Return deterministic center, inner-ring, and surface samples.

    The returned tensor has shape ``[B, obstacle_count, 1 + 2 * samples, 2]``.
    Keeping these points analytic and fixed makes the corridor loss fully
    differentiable with respect to the learned half-space offsets.
    """
    if samples < 4:
        raise ValueError("obstacle sample count must be at least four")
    angles = torch.arange(
        samples, dtype=scenes.circle_radii.dtype, device=scenes.circle_radii.device
    ) * (2.0 * math.pi / samples)
    unit = torch.stack((angles.cos(), angles.sin()), dim=-1)
    ring_fractions = scenes.circle_radii.new_tensor((0.5, 1.0))
    offsets = ring_fractions[:, None, None] * unit[None, :, :]
    rings = (
        scenes.circle_centers_xy[:, :, None, None, :]
        + scenes.circle_radii[:, :, None, None, None] * offsets[None, None, :, :, :]
    )
    rings = rings.flatten(2, 3)
    return torch.cat((scenes.circle_centers_xy[:, :, None, :], rings), dim=2)


def polytope_obstacle_metrics(
    decoded: DecodedAction,
    scenes: SceneBatch,
    *,
    samples: int = POLYTOPE_OBSTACLE_SAMPLE_COUNT,
    clearance_margin: float = POLYTOPE_OBSTACLE_MARGIN,
) -> dict[str, Tensor]:
    """Measure obstacle occupancy of every learned corridor.

    A corridor is represented by ``A x - b <= 0``.  As in Diff2, logsumexp
    over faces gives a smooth polytope signed distance and softplus of its
    negated logit is the target-zero NLL for unsafe membership in the union
    of corridor pieces.  Expanding the corridor by ``clearance_margin``
    penalizes both direct penetration and insufficient obstacle clearance.
    """
    if decoded.corridor_halfspaces.shape[:2] != (
        scenes.batch_size,
        PIECE_COUNT,
    ):
        raise ValueError("decoded corridor batch shape does not match scenes")
    if clearance_margin < 0.0:
        raise ValueError("clearance margin must be non-negative")
    points_xy = _sample_obstacle_points(scenes, samples=samples).to(
        dtype=decoded.corridor_halfspaces.dtype,
        device=decoded.corridor_halfspaces.device,
    )
    points = torch.cat((points_xy, torch.zeros_like(points_xy[..., :1])), dim=-1)
    halfspaces = decoded.corridor_halfspaces
    signed_faces = (
        torch.einsum("bkfd,bosd->bksof", halfspaces[..., :3], points)
        + halfspaces[..., 3][:, :, None, None, :]
    )
    max_violation = signed_faces.amax(dim=-1)
    expanded_faces = signed_faces - decoded.corridor_halfspaces.new_tensor(
        clearance_margin
    )
    smooth_signed_distance = torch.logsumexp(
        POLYTOPE_FACE_TEMPERATURE * expanded_faces, dim=-1
    ) / decoded.corridor_halfspaces.new_tensor(POLYTOPE_FACE_TEMPERATURE)
    membership_logits = -POLYTOPE_MEMBERSHIP_SHARPNESS * smooth_signed_distance
    # Target-zero NLL for the union is the sum of per-piece softplus logits.
    loss = F.softplus(membership_logits).sum(dim=1).mean(dim=(1, 2))
    penetration = torch.relu(-max_violation).amax(dim=(1, 2, 3))
    minimum_exclusion = max_violation.amin(dim=(1, 2, 3))
    return {
        "loss": loss,
        "penetration": penetration,
        "minimum_exclusion": minimum_exclusion,
    }


def polytope_obstacle_loss(decoded: DecodedAction, scenes: SceneBatch) -> Tensor:
    """Return the per-lane obstacle-exclusion loss for learned polytopes."""
    return polytope_obstacle_metrics(decoded, scenes)["loss"]


def decode_action(
    action: Tensor, scenes: SceneBatch, *, endpoint_mode: str = "predicted_terminal"
) -> DecodedAction:
    """Decode the simplified V9 chart into planar pivopt inputs."""
    if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must have shape [B,{ACTION_DIM}]")
    if action.shape[0] != scenes.batch_size:
        raise ValueError("action and scene batch sizes must match")
    if endpoint_mode not in ("fixed_goal", "predicted_terminal"):
        raise ValueError(f"unknown endpoint mode: {endpoint_mode}")
    dtype, device = action.dtype, action.device
    directions = FACE_DIRECTIONS.to(dtype=dtype, device=device)
    start = torch.cat((scenes.start_xy, torch.zeros_like(scenes.start_xy[:, :1])), -1)
    delta = scenes.goal_xy - scenes.start_xy
    tangent = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(
        1.0e-6
    )
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
    fractions = action.new_tensor((0.25, 0.50, 0.75))[None, :, None]
    base_xy = (1.0 - fractions) * scenes.start_xy[:, None] + fractions * scenes.goal_xy[
        :, None
    ]
    residual = action[..., _PATH_SLICE].reshape(-1, 3, 2).tanh()
    longitudinal = LONGITUDINAL_RESIDUAL_MAX * residual[..., :1]
    lateral = LATERAL_RESIDUAL_MAX * residual[..., 1:2]
    internal_xy = (
        base_xy + tangent[:, None, :] * longitudinal + normal[:, None, :] * lateral
    )
    internal = torch.cat((internal_xy, torch.zeros_like(internal_xy[..., :1])), -1)
    terminal_residual = action[..., _TERMINAL_RESIDUAL_SLICE].tanh()
    predicted_terminal_xy = (
        scenes.goal_xy
        + tangent * (LONGITUDINAL_RESIDUAL_MAX * terminal_residual[..., :1])
        + normal * (LATERAL_RESIDUAL_MAX * terminal_residual[..., 1:2])
    )
    terminal_xy = (
        scenes.goal_xy if endpoint_mode == "fixed_goal" else predicted_terminal_xy
    )
    terminal_position = torch.cat(
        (terminal_xy, torch.zeros_like(terminal_xy[:, :1])), dim=-1
    )
    waypoints = torch.cat((start[:, None], internal, terminal_position[:, None]), dim=1)

    total_duration = (
        TOTAL_DURATION_MIN
        + (TOTAL_DURATION_MAX - TOTAL_DURATION_MIN)
        * action[..., _TOTAL_DURATION_INDEX].sigmoid()
    )
    tail_velocity_xy = TAIL_VELOCITY_MAX * action[..., _TAIL_VELOCITY_SLICE].tanh()
    tail_acceleration_xy = (
        TAIL_ACCELERATION_MAX * action[..., _TAIL_ACCELERATION_SLICE].tanh()
    )
    tail_position = terminal_position
    tail_velocity = torch.cat(
        (tail_velocity_xy, torch.zeros_like(tail_velocity_xy[:, :1])), dim=-1
    )
    tail_acceleration = torch.cat(
        (tail_acceleration_xy, torch.zeros_like(tail_acceleration_xy[:, :1])), dim=-1
    )
    tail_state = torch.stack((tail_position, tail_velocity, tail_acceleration), dim=-1)

    slack = (
        SLACK_MAX
        * action[..., _SLACK_SLICE].reshape(-1, PIECE_COUNT, FACE_COUNT).sigmoid()
    )
    starts = waypoints[:, :-1]
    ends = waypoints[:, 1:]
    start_support = torch.einsum("fd,bkd->bkf", directions, starts)
    end_support = torch.einsum("fd,bkd->bkf", directions, ends)
    corridor_b = torch.maximum(start_support, end_support) + SAFETY_MARGIN + slack
    halfspace_normals = directions[None, None].expand(
        action.shape[0], PIECE_COUNT, FACE_COUNT, 3
    )
    corridor_halfspaces = torch.cat((halfspace_normals, -corridor_b[..., None]), dim=-1)
    corridor_interiors = 0.5 * (starts + ends)
    return DecodedAction(
        waypoints=waypoints,
        terminal_position=terminal_position,
        total_duration=total_duration,
        tail_state=tail_state,
        corridor_interiors=corridor_interiors,
        corridor_halfspaces=corridor_halfspaces,
        slack=slack,
        corridor_b=corridor_b,
    )


@dataclass(frozen=True)
class TrajectorySetup:
    """Fixed structure and operations reused across policy updates."""

    plan: p.Plan
    solver: Solver
    handles: tuple[p.InputHandle, ...]

    def bind(self, scenes: SceneBatch, decoded: DecodedAction) -> p.Inputs:
        """Keep learned tensors in their public dtype and preserve their graph."""
        zeros = torch.zeros_like(decoded.waypoints[:, 0])
        head = torch.stack(
            (
                torch.cat((scenes.start_xy, zeros[:, 2:3]), dim=-1),
                zeros,
                zeros,
            ),
            dim=-1,
        )
        # Zero Hard coordinates map exactly to each learned interior.
        initial = torch.zeros(
            (scenes.batch_size, 4 * (PIECE_COUNT - 1)),
            dtype=decoded.waypoints.dtype,
            device=decoded.waypoints.device,
        )
        values = (
            head.contiguous(),
            decoded.tail_state.contiguous(),
            decoded.total_duration.contiguous(),
            decoded.corridor_halfspaces.contiguous(),
            decoded.waypoints[:, 1:-1].contiguous(),
            torch.cat(
                (
                    decoded.corridor_halfspaces[:, :-1],
                    decoded.corridor_halfspaces[:, 1:],
                ),
                dim=2,
            ).contiguous(),
            initial,
        )
        return p.Inputs(
            plan=self.plan, values=tuple(zip(self.handles, values, strict=True))
        )


def prepare_problem(
    batch_size: int,
    *,
    dtype: torch.dtype = torch.float64,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
    optimizer: str = NATIVE_OPTIMIZER,
    gradient_tolerance: float = GRADIENT_TOLERANCE,
    maximum_line_searches: int = 64,
    backward: p.Implicit | None = None,
    constraints: str = "hinge",
    barrier_truncated: bool = True,
) -> TrajectorySetup:
    """Prepare the Hard/FixedTotal model once, before training or validation."""

    def calibrated(calibration: Calibration) -> Calibration:
        # Ablation only: drop barrier truncation, keeping every explicit literal.
        if barrier_truncated:
            return calibration
        return replace(calibration, activation=None)

    names = (
        "head",
        "tail",
        "total_duration",
        "corridor",
        "waypoint_interiors",
        "waypoint_halfspaces",
        "initial",
    )

    def ref(name: str):
        return p.Input(name, differentiate=name != "initial")

    trajectory = p.MincoSpec(
        p.Space(position=3),
        derivative_order=3,
        pieces=PIECE_COUNT,
        head_state=ref("head"),
        tail_state=ref("tail"),
        time=p.FixedTotalTime(
            total_duration=ref("total_duration"), minimum_piece_duration=0.10
        ),
        waypoints=p.HardWaypoints(
            interiors=ref("waypoint_interiors"),
            halfspaces=ref("waypoint_halfspaces"),
        ),
    )
    quadrotor = p.Quadrotor()
    spec = p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=16),
        objective=(
            p.Energy("energy"),
            term(
                p.Integral,
                "corridor",
                lambda profile: p.Inside(
                    trajectory.position, ref("corridor"), profile=profile
                ),
                calibrated(
                    Calibration(
                        hinge_weight=1000,
                        width=0.10,
                        barrier_weight=22.0,
                        delta=0.022,
                        activation=0.066,
                    )
                ),
                constraints,
            ),
            term(
                p.Integral,
                "velocity",
                lambda profile: p.NormBelow(trajectory.velocity, 5, profile=profile),
                calibrated(
                    Calibration(
                        hinge_weight=1000,
                        width=0.10,
                        barrier_weight=22.0,
                        delta=0.022,
                        activation=0.5,
                    )
                ),
                constraints,
            ),
            term(
                p.Integral,
                "acceleration",
                lambda profile: p.NormBelow(
                    trajectory.acceleration, 30, profile=profile
                ),
                calibrated(
                    Calibration(
                        hinge_weight=1000,
                        width=0.10,
                        barrier_weight=22.0,
                        delta=0.022,
                        activation=3.0,
                    )
                ),
                constraints,
            ),
            term(
                p.Integral,
                "jerk",
                lambda profile: p.NormBelow(trajectory.jerk, 150, profile=profile),
                calibrated(
                    Calibration(
                        hinge_weight=1000,
                        width=0.10,
                        barrier_weight=22.0,
                        delta=0.022,
                        activation=15.0,
                    )
                ),
                constraints,
            ),
            term(
                p.Integral,
                "body_rate",
                lambda profile: p.NormBelow(quadrotor.body_rate, 10, profile=profile),
                calibrated(
                    Calibration(
                        hinge_weight=1000,
                        width=0.10,
                        barrier_weight=22.0,
                        delta=0.022,
                        activation=1.0,
                    )
                ),
                constraints,
            ),
            term(
                p.Integral,
                "tilt",
                lambda profile: p.TiltBelow(quadrotor.attitude, 1, profile=profile),
                calibrated(
                    Calibration(
                        hinge_weight=1000,
                        width=0.10,
                        barrier_weight=22.0,
                        delta=0.022,
                        activation=0.1,
                    )
                ),
                constraints,
            ),
            term(
                p.Integral,
                "thrust",
                lambda profile: p.Within(quadrotor.thrust, 0, 30, profile=profile),
                calibrated(
                    Calibration(
                        hinge_weight=1000,
                        width=0.10,
                        barrier_weight=22.0,
                        delta=0.022,
                        activation=1.5,
                    )
                ),
                constraints,
            ),
        ),
        forward=(p.BFGS if optimizer == "bfgs" else p.LBFGS)(
            initial_guess=ref("initial"),
            settings=p.OptimizerSettings(
                past=0,
                gradient_tolerance=gradient_tolerance,
                maximum_iterations=maximum_iterations,
                maximum_line_searches=maximum_line_searches,
            ),
        ),
        backward=backward
        if backward is not None
        else p.Implicit(
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
                p.InputShape(ref("corridor"), (batch_size, PIECE_COUNT, FACE_COUNT, 4)),
                p.InputShape(
                    ref("waypoint_halfspaces"),
                    (batch_size, PIECE_COUNT - 1, 2 * FACE_COUNT, 4),
                ),
            ),
        ),
    )
    return TrajectorySetup(
        plan,
        build_solver(plan, dtype=dtype, backward=True),
        tuple(plan.input(name) for name in names),
    )


def sample_trajectory(
    coefficients: Tensor, durations: Tensor, samples: int = 16
) -> Tensor:
    """Evaluate MINCO coefficient rows; result shape is [B,K,samples,3]."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    local_fraction = torch.linspace(
        0.0, 1.0, samples, dtype=coefficients.dtype, device=coefficients.device
    )
    local_time = durations[..., None] * local_fraction
    position = torch.zeros(
        (*coefficients.shape[:2], samples, 3),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    for degree in range(coefficients.shape[2] - 1, -1, -1):
        position = (
            position * local_time[..., None]
            + coefficients[:, :, degree, :][:, :, None, :]
        )
    return position


def trajectory_metrics(
    result,
    scenes: SceneBatch,
    decoded: DecodedAction,
    *,
    solver_cost_weight: float = SOLVER_COST_WEIGHT,
    solver_cost_transform: str = "linear",
    solver_cost_scale: float = 100.0,
) -> dict[str, Tensor]:
    if solver_cost_weight < 0.0:
        raise ValueError("solver cost weight must be non-negative")
    samples = sample_trajectory(result.coefficients, result.durations)
    flat = samples.reshape(scenes.batch_size, -1, 3)
    step = flat[:, 1:, :2] - flat[:, :-1, :2]
    length = torch.linalg.vector_norm(step, dim=-1).sum(-1)
    second = flat[:, 2:, :2] - 2.0 * flat[:, 1:-1, :2] + flat[:, :-2, :2]
    smoothness = second.square().mean(dim=(-1, -2))
    final_xy = samples[:, -1, -1, :2]
    goal = (final_xy - scenes.goal_xy).square().sum(-1)
    esdf = analytic_esdf(flat[:, :, :2], scenes)
    esdf_loss = torch.relu(CLEARANCE_MARGIN - esdf).square().mean(-1)
    min_clearance = esdf.amin(-1)
    polytope = polytope_obstacle_metrics(decoded, scenes)
    widths = (
        decoded.corridor_b[..., : FACE_COUNT // 2]
        + decoded.corridor_b[..., FACE_COUNT // 2 :]
    )
    width_loss = torch.relu(0.50 - widths).square().mean(dim=(-1, -2))
    slack_loss = decoded.slack.square().mean(dim=(-1, -2))
    corridor = width_loss + 0.05 * slack_loss
    duration = 0.02 * (decoded.total_duration - 2.5).square()
    solver_cost = torch.nan_to_num(result.cost, nan=0.0, posinf=1.0e4, neginf=1.0e4)
    trajectory_loss = 10.0 * goal + 0.10 * length + 0.05 * smoothness + 5.0 * esdf_loss
    direct_loss = POLYTOPE_OBSTACLE_WEIGHT * polytope["loss"] + corridor + duration
    task_per_lane = trajectory_loss + direct_loss
    weighted_solver_cost = solver_cost_weight * transform_solver_cost(
        solver_cost, solver_cost_transform, solver_cost_scale
    )
    per_lane = task_per_lane + weighted_solver_cost
    return {
        "per_lane": per_lane,
        "task_loss": task_per_lane,
        "trajectory_loss": trajectory_loss,
        "direct_loss": direct_loss,
        "goal": goal,
        "length": length,
        "smoothness": smoothness,
        "esdf": esdf_loss,
        "polytope_obstacle": polytope["loss"],
        "polytope_penetration": polytope["penetration"],
        "polytope_minimum_exclusion": polytope["minimum_exclusion"],
        "corridor": corridor,
        "duration": duration,
        "solver_cost": solver_cost,
        "weighted_solver_cost": weighted_solver_cost,
        "minimum_clearance": min_clearance,
        "samples": samples,
    }


def transform_solver_cost(cost: Tensor, transform: str, scale: float) -> Tensor:
    """Compress large costs, preserving unit slope at zero and native solve."""
    if scale <= 0.0:
        raise ValueError("solver cost scale must be positive")
    if transform == "linear":
        return cost
    if transform == "log1p":
        return scale * torch.log1p(cost / scale)
    if transform == "sqrt":
        # Algebraically 2*scale*(sqrt(1+cost/scale)-1), stable near zero.
        return 2.0 * cost / (torch.sqrt(1.0 + cost / scale) + 1.0)
    raise ValueError(f"unknown solver cost transform: {transform}")


def _valid_lanes(result, metrics: dict[str, Tensor]) -> Tensor:
    return (result.status == int(ForwardStatus.RESULT_AVAILABLE)) & torch.isfinite(
        metrics["per_lane"]
    )


def _loss_from_result(result, metrics: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    valid = _valid_lanes(result, metrics)
    if not bool(valid.any()):
        raise RuntimeError(
            "all CPU solver lanes failed; bounded decoder produced no trainable lane"
        )
    core = metrics["per_lane"][valid].mean()
    failure_rate = (~valid).to(metrics["per_lane"].dtype).mean()
    return core + 100.0 * failure_rate, valid


def _diagnose_backward(
    solver: Solver,
    result,
    state,
    scenes: SceneBatch,
    decoded: DecodedAction,
    *,
    solver_cost_weight: float,
    solver_cost_transform: str,
    solver_cost_scale: float,
):
    """Recover output seeds after ordinary AD, only for a rejected update."""
    names = ("x", "waypoints", "coefficients", "durations", "cost")
    diagnostic = replace(
        result,
        **{name: getattr(result, name).detach().requires_grad_(True) for name in names},
    )
    scenes = replace(
        scenes,
        **{
            field.name: getattr(scenes, field.name).detach() for field in fields(scenes)
        },
    )
    decoded = replace(
        decoded,
        **{
            field.name: getattr(decoded, field.name).detach()
            for field in fields(decoded)
        },
    )
    # Direct losses still contribute their values to lane validity and the
    # normalization, even though this diagnostic does not revisit the network.
    metrics = trajectory_metrics(
        diagnostic,
        scenes,
        decoded,
        solver_cost_weight=solver_cost_weight,
        solver_cost_transform=solver_cost_transform,
        solver_cost_scale=solver_cost_scale,
    )
    loss, _ = _loss_from_result(diagnostic, metrics)
    seeds = torch.autograd.grad(
        loss, tuple(getattr(diagnostic, name) for name in names), allow_unused=True
    )
    return solver.pullback(
        state, p.ResultSeeds(**dict(zip(names, seeds, strict=True)))
    ).diagnostics


def _metric_means(metrics: dict[str, Tensor], valid: Tensor) -> dict[str, float]:
    return {
        name: float(metrics[name][valid].detach().mean())
        for name in (
            "task_loss",
            "trajectory_loss",
            "direct_loss",
            "goal",
            "length",
            "smoothness",
            "esdf",
            "polytope_obstacle",
            "polytope_penetration",
            "polytope_minimum_exclusion",
            "corridor",
            "duration",
            "solver_cost",
            "weighted_solver_cost",
        )
    }


def _solve_means(result: Result, valid: Tensor) -> dict[str, float]:
    status = result.diagnostics.optimizer_status[valid]
    summary = {
        "stationarity_mean": float(
            result.diagnostics.stationarity[valid].detach().mean()
        ),
        "stationarity_max": float(
            result.diagnostics.stationarity[valid].detach().max()
        ),
        "iterations_mean": float(
            result.diagnostics.iterations[valid].to(torch.float64).mean()
        ),
        "optimizer_converged_fraction": float(
            (
                (status == int(OptimizerStatus.GRADIENT_CONVERGENCE))
                | (status == int(OptimizerStatus.RELATIVE_VALUE_CONVERGENCE))
            )
            .to(torch.float64)
            .mean()
        ),
        "maximum_iterations_fraction": float(
            (status == int(OptimizerStatus.MAXIMUM_ITERATIONS)).to(torch.float64).mean()
        ),
        "objective_evaluations_mean": float(
            result.diagnostics.objective_evaluations[valid].double().mean()
        ),
    }
    return summary


def _new_run_dir(root: Path, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = root / f"{stamp}_seed{seed}"
    suffix = 1
    while path.exists():
        path = root / f"{stamp}_seed{seed}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def _polytope_vertices(normals: Tensor, b: Tensor) -> list[tuple[float, float]]:
    rows = normals[:, :2].detach().cpu().tolist()
    offsets = b.detach().cpu().tolist()
    vertices: list[tuple[float, float]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ax, ay = rows[i]
            bx, by = rows[j]
            determinant = ax * by - ay * bx
            if abs(determinant) < 1.0e-8:
                continue
            x = (offsets[i] * by - ay * offsets[j]) / determinant
            y = (ax * offsets[j] - offsets[i] * bx) / determinant
            if all(
                nx * x + ny * y <= limit + 1.0e-6
                for (nx, ny), limit in zip(rows, offsets)
            ):
                vertices.append((x, y))
    if not vertices:
        return []
    center = (
        sum(point[0] for point in vertices) / len(vertices),
        sum(point[1] for point in vertices) / len(vertices),
    )
    return sorted(
        vertices,
        key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]),
    )


def write_loss_plot(
    history: list[dict], path: Path, *, solver_cost_weight: float, stopped: bool = False
) -> None:
    values = [float(item["loss"]) for item in history]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.plot(
        range(len(values)), values, color="#1f5fbf", linewidth=1.8, label="total loss"
    )
    polytope_values = [item.get("polytope_obstacle") for item in history]
    if all(value is not None for value in polytope_values):
        axis.plot(
            range(len(polytope_values)),
            [
                POLYTOPE_OBSTACLE_WEIGHT * float(cast(float, value))
                for value in polytope_values
            ],
            color="#d46a1f",
            linewidth=1.2,
            label="weighted polytope obstacle loss",
        )
    if solver_cost_weight > 0.0:
        axis.plot(
            range(len(history)),
            [float(item["weighted_solver_cost"]) for item in history],
            color="#7a5195",
            linewidth=1.2,
            label="weighted native solver cost",
        )
    axis.set_title(
        "Strict backward rejected before the first update"
        if stopped and not history
        else "CPU polytope training loss"
    )
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.grid(True, alpha=0.25)
    if all(value is not None for value in polytope_values):
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_trajectory_plot(
    scenes: SceneBatch, decoded: DecodedAction, samples: Tensor, path: Path
) -> None:
    start = scenes.start_xy[0].detach().cpu().tolist()
    goal = scenes.goal_xy[0].detach().cpu().tolist()
    centers = scenes.circle_centers_xy[0].detach().cpu().tolist()
    radii = scenes.circle_radii[0].detach().cpu().tolist()
    trajectory = samples[0].reshape(-1, 3).detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(9, 6), dpi=150)
    for center, radius in zip(centers, radii):
        axis.add_patch(
            Circle(
                center,
                radius,
                facecolor="#e85b55",
                edgecolor="#9f2424",
                alpha=0.55,
            )
        )
    for piece in range(PIECE_COUNT):
        vertices = _polytope_vertices(FACE_DIRECTIONS, decoded.corridor_b[0, piece])
        if len(vertices) >= 2:
            axis.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    fill=False,
                    edgecolor="#5096bd",
                    linewidth=1.0,
                )
            )
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#1428a0",
        linewidth=2.2,
        label="optimized trajectory",
    )
    axis.scatter(*start, color="#169646", s=55, zorder=4, label="start")
    axis.scatter(*goal, color="#9b2c9b", s=55, zorder=4, label="goal")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Planar Hard Minkowski trajectory and learned corridors")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _final_report(
    result: Result,
    metrics: dict[str, Tensor],
    valid: Tensor,
    scenes: SceneBatch,
    run_dir: Path,
) -> dict:
    summary = {
        "forward_status": [
            int(value) for value in result.status.detach().cpu().tolist()
        ],
        "optimizer_status": [
            int(value)
            for value in result.diagnostics.optimizer_status.detach().cpu().tolist()
        ],
        "valid_lanes": int(valid.sum()),
        "goal_error": float(metrics["goal"][valid].mean().sqrt().detach()),
        "minimum_clearance": float(metrics["minimum_clearance"][valid].min().detach()),
        "polytope_obstacle_loss": float(
            metrics["polytope_obstacle"][valid].mean().detach()
        ),
        "polytope_penetration": float(
            metrics["polytope_penetration"][valid].max().detach()
        ),
        "polytope_minimum_exclusion": float(
            metrics["polytope_minimum_exclusion"][valid].min().detach()
        ),
        "task_loss": float(metrics["task_loss"][valid].mean().detach()),
        "trajectory_loss": float(metrics["trajectory_loss"][valid].mean().detach()),
        "direct_loss": float(metrics["direct_loss"][valid].mean().detach()),
        "solver_cost": float(metrics["solver_cost"][valid].mean().detach()),
        "weighted_solver_cost": float(
            metrics["weighted_solver_cost"][valid].mean().detach()
        ),
        "total_duration": float(result.durations[0].sum().detach()),
        "piece_durations": [
            float(value) for value in result.durations[0].detach().cpu().tolist()
        ],
        "run_dir": str(run_dir),
    }
    summary.update(_solve_means(result, valid))
    return summary


def run_training(
    *,
    steps: int = 300,
    batch_size: int = 32,
    seed: int = 7,
    output_dir: str | Path | None = None,
    log_every: int = 10,
    learning_rate: float = LEARNING_RATE,
    solver_cost_weight: float = SOLVER_COST_WEIGHT,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
    native_optimizer: str = NATIVE_OPTIMIZER,
    gradient_tolerance: float = GRADIENT_TOLERANCE,
    solver_cost_transform: str = "linear",
    solver_cost_scale: float = 100.0,
    validation_size: int = 1,
    validation_seed: int | None = None,
    checkpoint_every: int = 0,
    endpoint_mode: str = "predicted_terminal",
    backward: p.Implicit | None = None,
    minimum_wall_seconds: float = 0.0,
    constraints: str = "hinge",
    barrier_truncated: bool = True,
) -> dict:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if not math.isfinite(minimum_wall_seconds) or minimum_wall_seconds < 0:
        raise ValueError("minimum wall seconds must be finite and non-negative")
    if solver_cost_weight < 0.0:
        raise ValueError("solver cost weight must be non-negative")
    if maximum_iterations < 1:
        raise ValueError("iterations must be positive")
    if validation_size < 1 or checkpoint_every < 0:
        raise ValueError(
            "validation size must be positive and checkpoint interval non-negative"
        )
    transform_solver_cost(torch.tensor(0.0), solver_cost_transform, solver_cost_scale)
    if endpoint_mode not in ("fixed_goal", "predicted_terminal"):
        raise ValueError(f"unknown endpoint mode: {endpoint_mode}")
    print(f"constraints: {constraints}, barrier_truncated: {barrier_truncated}")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    policy = PolytopePolicy().to(dtype=torch.float64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    root = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / "cpu_polytope_training"
    )
    run_dir = _new_run_dir(root, seed)
    training_problem = prepare_problem(
        batch_size,
        maximum_iterations=maximum_iterations,
        optimizer=native_optimizer,
        gradient_tolerance=gradient_tolerance,
        backward=backward,
        constraints=constraints,
        barrier_truncated=barrier_truncated,
    )
    validation_problem = prepare_problem(
        validation_size,
        maximum_iterations=maximum_iterations,
        optimizer=native_optimizer,
        gradient_tolerance=gradient_tolerance,
        backward=backward,
        constraints=constraints,
        barrier_truncated=barrier_truncated,
    )
    backward_policy = training_problem.plan.spec.backward
    settings = {
        "plan": training_problem.plan.explain(),
        "constraints": constraints,
        "barrier_truncated": barrier_truncated,
        "action_dim": ACTION_DIM,
        "piece_count": PIECE_COUNT,
        "face_count": FACE_COUNT,
        "steps": steps,
        "minimum_wall_seconds": minimum_wall_seconds,
        "batch_size": batch_size,
        "seed": seed,
        "learning_rate": learning_rate,
        "solver_cost_weight": solver_cost_weight,
        "solver_cost_transform": solver_cost_transform,
        "solver_cost_scale": solver_cost_scale,
        "native_optimizer": native_optimizer,
        "maximum_iterations": maximum_iterations,
        "gradient_tolerance": gradient_tolerance,
        "validation_size": validation_size,
        "validation_seed": seed + 1 if validation_seed is None else validation_seed,
        "checkpoint_every": checkpoint_every,
        "endpoint_mode": endpoint_mode,
        "backward": {
            "curvature": {"kind": "exact"},
            "damping": {
                "kind": type(backward_policy.damping).__name__,
                **asdict(backward_policy.damping),
            },
            "response": {
                "kind": type(backward_policy.response).__name__,
                **asdict(backward_policy.response),
            },
            "checks": asdict(backward_policy.checks),
        },
        "total_duration_range": [TOTAL_DURATION_MIN, TOTAL_DURATION_MAX],
        "tail_velocity_max": TAIL_VELOCITY_MAX,
        "tail_acceleration_max": TAIL_ACCELERATION_MAX,
        "polytope_obstacle_weight": POLYTOPE_OBSTACLE_WEIGHT,
        "polytope_obstacle_sample_count": POLYTOPE_OBSTACLE_SAMPLE_COUNT,
        "polytope_obstacle_margin": POLYTOPE_OBSTACLE_MARGIN,
        "polytope_face_temperature": POLYTOPE_FACE_TEMPERATURE,
        "polytope_membership_sharpness": POLYTOPE_MEMBERSHIP_SHARPNESS,
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    validation = sample_scenes(
        validation_size,
        generator=torch.Generator(device="cpu").manual_seed(
            settings["validation_seed"]
        ),
    )

    def evaluate():
        with torch.no_grad():
            decoded = decode_action(
                policy(validation.features()), validation, endpoint_mode=endpoint_mode
            )
            result = validation_problem.solver.solve(
                validation_problem.bind(validation, decoded)
            )
            metrics = trajectory_metrics(
                result,
                validation,
                decoded,
                solver_cost_weight=solver_cost_weight,
                solver_cost_transform=solver_cost_transform,
                solver_cost_scale=solver_cost_scale,
            )
            valid = _valid_lanes(result, metrics)
            if not bool(valid.any()):
                raise RuntimeError("all validation solver lanes failed")
            summary = _final_report(result, metrics, valid, validation, run_dir)
            dense = sample_trajectory(
                result.coefficients, result.durations, samples=128
            )
            clearance = analytic_esdf(dense.flatten(1, 2)[..., :2], validation).amin(-1)
            summary.update(
                {
                    "validation_size": validation_size,
                    "validation_failure_fraction": float((~valid).double().mean()),
                    "sampled_collision_fraction": float(
                        ((clearance < 0.0) & valid).double().sum() / valid.sum()
                    ),
                    "sampled_safe_fraction": float(
                        ((clearance >= CLEARANCE_MARGIN) & valid).double().mean()
                    ),
                    "dense_clearance_mean": float(clearance[valid].mean()),
                    "dense_clearance_p05": float(
                        torch.quantile(clearance[valid], 0.05)
                    ),
                    "dense_clearance_min": float(clearance[valid].min()),
                    "solver_cost_median": float(result.cost[valid].median()),
                    "solver_cost_p95": float(torch.quantile(result.cost[valid], 0.95)),
                }
            )
            lanes = {
                "valid": valid.tolist(),
                "task_loss": metrics["task_loss"].tolist(),
                "solver_cost": result.cost.tolist(),
                "dense_minimum_clearance": clearance.tolist(),
                "polytope_penetration": metrics["polytope_penetration"].tolist(),
            }
        return summary, decoded, metrics, lanes

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "step": step,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scene_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "settings": settings,
            },
            run_dir / f"checkpoint_{step:06d}.pt",
        )

    history: list[dict] = []
    evaluations: list[dict] = []
    rejection = None
    outcome = "completed"
    if checkpoint_every:
        save_checkpoint(0)
        evaluations.append({"step": 0, "wall_seconds": 0.0, **evaluate()[0]})
    evaluation_seconds = 0.0
    training_started = time.perf_counter()
    step = 0
    while step < steps or time.perf_counter() - training_started < minimum_wall_seconds:
        scenes = sample_scenes(batch_size, generator=generator)
        action = policy(scenes.features())
        decoded = decode_action(action, scenes, endpoint_mode=endpoint_mode)
        result, state = training_problem.solver.solve_with_state(
            training_problem.bind(scenes, decoded)
        )
        metrics = trajectory_metrics(
            result,
            scenes,
            decoded,
            solver_cost_weight=solver_cost_weight,
            solver_cost_transform=solver_cost_transform,
            solver_cost_scale=solver_cost_scale,
        )
        loss, valid = _loss_from_result(result, metrics)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(policy.parameters())
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        raw_gradient_norm = torch.nn.utils.get_total_norm(gradients)
        if not bool(torch.isfinite(raw_gradient_norm)):
            # NaN solver lanes may already have reached shared network grads.
            # Diagnose only this failed step and never let Adam consume them.
            diagnostics = _diagnose_backward(
                training_problem.solver,
                result,
                state,
                scenes,
                decoded,
                solver_cost_weight=solver_cost_weight,
                solver_cost_transform=solver_cost_transform,
                solver_cost_scale=solver_cost_scale,
            )
            native_rejected = bool(
                (diagnostics.status != int(BackwardStatus.APPLIED)).any()
            )
            outcome = (
                "strict_backward_rejected"
                if native_rejected
                else "nonfinite_training_gradient"
            )
            reason = (
                "native_backward_rejected"
                if native_rejected
                else "nonfinite_parameter_gradient"
                if any(
                    not bool(torch.isfinite(gradient).all()) for gradient in gradients
                )
                else "nonfinite_gradient_norm"
            )
            rejection = {
                "attempted_step": step,
                "completed_steps": len(history),
                "reason": reason,
                "raw_gradient_norm": None,
            }
            for field in fields(diagnostics):
                value = getattr(diagnostics, field.name).detach().cpu()
                rejection[field.name] = [
                    number
                    if not value.is_floating_point() or math.isfinite(number)
                    else None
                    for number in value.tolist()
                ]
            rejection["status_names"] = [
                BackwardStatus(int(code)).name for code in diagnostics.status
            ]
            rejection["attempted_loss"] = float(loss.detach())
            rejection_file = (
                "strict_rejection.json"
                if native_rejected
                else "gradient_rejection.json"
            )
            (run_dir / rejection_file).write_text(
                json.dumps(rejection, allow_nan=False, indent=2) + "\n"
            )
            save_checkpoint(step)
            print(
                f"training stopped before step {step + 1}: {reason}, "
                f"{rejection['status_names']}"
            )
            break
        torch.nn.utils.clip_grads_with_norm_(
            parameters, max_norm=10.0, total_norm=raw_gradient_norm
        )
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "failure_rate": float((~valid).to(torch.float64).mean()),
            "valid_lanes": int(valid.sum()),
            "raw_gradient_norm": float(raw_gradient_norm),
            "training_seconds": time.perf_counter()
            - training_started
            - evaluation_seconds,
            **_metric_means(metrics, valid),
            **_solve_means(result, valid),
        }
        history.append(record)
        if step == 0 or (step + 1) % max(1, log_every) == 0 or step + 1 == steps:
            print(
                f"step {step + 1:04d}/{steps} loss={record['loss']:.6f} "
                f"goal={record['goal']:.5f} esdf={record['esdf']:.5f} "
                f"polytope={record['polytope_obstacle']:.5f} "
                f"native={record['solver_cost']:.3f} "
                f"valid={record['valid_lanes']}/{batch_size}"
            )
            (run_dir / "progress.json").write_text(json.dumps(record) + "\n")
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            evaluation_started = time.perf_counter()
            save_checkpoint(step + 1)
            evaluation = evaluate()[0]
            evaluations.append(
                {
                    "step": step + 1,
                    **evaluation,
                    "wall_seconds": time.perf_counter() - training_started,
                }
            )
            (run_dir / "training_history.json").write_text(
                json.dumps({"history": history, "evaluations": evaluations}) + "\n"
            )
            evaluation_seconds += time.perf_counter() - evaluation_started
        step += 1
    training_seconds = time.perf_counter() - training_started - evaluation_seconds
    final, validation_decoded, validation_metrics, validation_lanes = evaluate()
    if checkpoint_every and len(history) % checkpoint_every:
        save_checkpoint(len(history))
        evaluations.append(
            {
                "step": len(history),
                **final,
                "wall_seconds": time.perf_counter() - training_started,
            }
        )
    final["training_seconds"] = training_seconds
    final["wall_seconds"] = time.perf_counter() - training_started
    (run_dir / "validation_lanes.json").write_text(
        json.dumps(validation_lanes, indent=2) + "\n"
    )
    write_loss_plot(
        history,
        run_dir / "loss_curve.png",
        solver_cost_weight=solver_cost_weight,
        stopped=rejection is not None,
    )
    write_trajectory_plot(
        validation,
        validation_decoded,
        validation_metrics["samples"],
        run_dir / "trajectory_result.png",
    )
    (run_dir / "training_history.json").write_text(
        json.dumps(
            {
                "history": history,
                "evaluations": evaluations,
                "final": final,
                "outcome": outcome,
                "rejection": rejection,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"completed updates={len(history)} goal_error={final['goal_error']:.6f} "
        f"minimum_clearance={final['minimum_clearance']:.6f} "
        f"polytope_penetration={final['polytope_penetration']:.6f}"
    )
    print(f"run directory: {run_dir}")
    print(f"training time: {training_seconds:.2f}s")
    return {
        "run_dir": run_dir,
        "history": history,
        "final": final,
        "outcome": outcome,
        "rejection": rejection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--solver-cost-weight", type=float, default=SOLVER_COST_WEIGHT)
    parser.add_argument(
        "--solver-cost-transform", choices=("linear", "log1p", "sqrt"), default="linear"
    )
    parser.add_argument("--solver-cost-scale", type=float, default=100.0)
    parser.add_argument("--validation-size", type=int, default=1)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--endpoint-mode",
        choices=("fixed_goal", "predicted_terminal"),
        default="predicted_terminal",
    )
    parser.add_argument("--maximum-iterations", type=int, default=MAXIMUM_ITERATIONS)
    parser.add_argument(
        "--native-optimizer",
        choices=("bfgs", "lbfgs"),
        default=NATIVE_OPTIMIZER.name.lower(),
    )
    parser.add_argument("--gradient-tolerance", type=float, default=GRADIENT_TOLERANCE)
    add_constraint_argument(parser)
    parser.add_argument(
        "--barrier-untruncated",
        action="store_true",
        help="ablation: use the relaxed barrier without its activation cutoff",
    )
    args = parser.parse_args()
    if args.barrier_untruncated and args.constraints != "relaxed-barrier":
        parser.error("--barrier-untruncated requires --constraints relaxed-barrier")
    run_training(
        constraints=args.constraints,
        barrier_truncated=not args.barrier_untruncated,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        log_every=args.log_every,
        learning_rate=args.learning_rate,
        solver_cost_weight=args.solver_cost_weight,
        solver_cost_transform=args.solver_cost_transform,
        solver_cost_scale=args.solver_cost_scale,
        validation_size=args.validation_size,
        validation_seed=args.validation_seed,
        checkpoint_every=args.checkpoint_every,
        endpoint_mode=args.endpoint_mode,
        maximum_iterations=args.maximum_iterations,
        native_optimizer=args.native_optimizer,
        gradient_tolerance=args.gradient_tolerance,
        backward=p.Implicit(
            checks=p.BackwardChecks(
                stationarity_tolerance=None,
                hessian_symmetry_tolerance=None,
                hessian_pivot_tolerance=None,
                adjoint_residual_tolerance=None,
            ),
            damping=p.MinimumShift(),
        ),
    )


if __name__ == "__main__":
    main()
