"""Train the certified Unicycle gate task through the public NumPy API.

The network controls 32 bounded corridor expansions and three physical caps.
Training composes NumPy ``solve_with_state``/``pullback`` calls with a Torch
network and Adam.  The command line can reproduce the frozen 1020 A/RS runs,
freeze the 1021 comparison design, execute its paired runs, and analyze them.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses as dc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cpu_gate_training as gate
import numpy as np
import torch

from pivopt import ForwardStatus
from pivopt import numpy as cpu
from pivopt import problem as p

RESULT_FIELDS = ("x", "waypoints", "coefficients", "durations", "cost")
DIAGNOSTIC_FIELDS = (
    "response_displacement",
    "response_residual_ratio",
    "response_correction_steps",
)
CHECKPOINT_UPDATES = tuple(range(0, 1601, 200))
ENDPOINTS = ("auc_0_1600", "utility_at_1600", "utility_at_time")
TIME_BUDGET_SECONDS = 21.78
TEST_SEED = 290002


@dataclass(frozen=True)
class Method:
    tag: str
    label: str
    first: str
    second: str | None = None
    switch_at: int | None = None
    task_scale: float = 1.0

    def mode(self, successful_updates: int) -> str:
        if self.second is None or successful_updates < self.switch_at:
            return self.first
        return self.second

    @property
    def modes(self) -> tuple[str, ...]:
        return (self.first,) if self.second is None else (self.first, self.second)


METHODS = {
    method.tag: method
    for method in (
        Method("A", "A", "A"),
        Method("RS", "RS", "RS"),
        Method("A-k0.8", "A×0.8", "A", task_scale=0.8),
        Method("A-k1.25", "A×1.25", "A", task_scale=1.25),
        Method("A-k1.5", "A×1.5", "A", task_scale=1.5),
        Method("RS-to-A-200", "RS→A@200", "RS", "A", 200),
        Method("RS-to-A-400", "RS→A@400", "RS", "A", 400),
        Method("RS-to-A-800", "RS→A@800", "RS", "A", 800),
        Method("A-to-RS-200", "A→RS@200", "A", "RS", 200),
        Method("A-to-RS-400", "A→RS@400", "A", "RS", 400),
        Method("A-to-RS-800", "A→RS@800", "A", "RS", 800),
    )
}


def clean(value):
    if dc.is_dataclass(value):
        return {
            field.name: clean(getattr(value, field.name)) for field in dc.fields(value)
        }
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    return sha256(Path(__file__).resolve())


class LearningPolicy(torch.nn.Module):
    def __init__(self, features: int, outputs: int, *, geometry: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(features, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, outputs),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[:geometry] = -1.5
        self.double()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class GateBatch:
    features: torch.Tensor
    scenes: gate.SceneBatch
    anchors: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    total: torch.Tensor


def backward_checks() -> p.BackwardChecks:
    return p.BackwardChecks(
        stationarity_tolerance=None,
        hessian_symmetry_tolerance=None,
        hessian_pivot_tolerance=None,
        adjoint_residual_tolerance=None,
    )


def response(mode: str):
    if mode == "A":
        return p.LinearResponse()
    if mode == "RS":
        return p.FiniteResponse(
            tilt=10.0,
            side=p.ResponseSide.PLUS,
            correction=p.Resolve(max_iterations=256),
        )
    raise ValueError(f"unknown response mode {mode!r}")


class CertifiedGateTask:
    geometry = 8 * 4
    outputs = geometry + 3
    width_min = 0.04
    width_max = 0.20
    time_factor = 1.6

    def __init__(self, batch_size: int, modes: tuple[str, ...] = ("A", "RS")):
        self.batch_size = batch_size
        setup = gate.prepare_problem(
            batch_size, task=gate.SGATE3, maximum_iterations=256
        )
        self.features = gate.SGATE3.feature_dim
        self.plans = {}
        for mode in dict.fromkeys(modes):
            backward = p.Implicit(
                damping=p.MinimumShift(),
                response=response(mode),
                checks=backward_checks(),
            )
            spec = setup.plan.spec.with_backward(backward)
            self.plans[mode] = p.build_plan(spec, setup.plan.shape)

    def sample(self, generator: torch.Generator) -> GateBatch:
        batch_size = self.batch_size

        def uniform(*shape):
            return torch.rand(
                (batch_size, *shape), generator=generator, dtype=torch.float64
            )

        signs = torch.where(uniform() < 0.5, -1.0, 1.0)
        centers = torch.empty((batch_size, 3, 2), dtype=torch.float64)
        centers[..., 0] = torch.tensor([2.25, 4.5, 6.75])
        centers[..., 1] = (
            signs[:, None] * torch.tensor([1.0, -1.0, 1.0]) * (0.3 + 0.25 * uniform(3))
        )
        widths = 0.65 + 0.15 * uniform(3)
        angles = torch.zeros((batch_size, 3), dtype=torch.float64)
        thickness = torch.full_like(widths, 0.15)
        wall_centers, wall_rotations, wall_extents = gate.build_wall_geometry(
            gate.SGATE3, centers, angles, widths, thickness
        )
        start = torch.zeros((batch_size, 2), dtype=torch.float64)
        goal = start.clone()
        goal[:, 0] = 9
        zero = torch.zeros(batch_size, dtype=torch.float64)
        empty = torch.empty((batch_size, 0, 2), dtype=torch.float64)
        scenes = gate.SceneBatch(
            start,
            goal,
            zero,
            zero,
            centers,
            angles,
            widths,
            thickness,
            0.6 + 0.3 * uniform(),
            uniform(),
            wall_centers,
            wall_rotations,
            wall_extents,
            empty,
            empty,
        )
        anchors = torch.empty((batch_size, 9, 2), dtype=torch.float64)
        anchors[..., 0] = torch.tensor([0.0, 1.5, 3.0, 3.75, 5.25, 6.0, 7.5, 8.25, 9.0])
        anchors[..., 1] = torch.stack(
            (
                zero,
                centers[:, 0, 1],
                centers[:, 0, 1],
                centers[:, 1, 1],
                centers[:, 1, 1],
                centers[:, 2, 1],
                centers[:, 2, 1],
                zero,
                zero,
            ),
            dim=1,
        )
        lower = torch.minimum(anchors[:, :-1], anchors[:, 1:])
        upper = torch.maximum(anchors[:, :-1], anchors[:, 1:])
        total = self.time_factor * scenes.task_distance / scenes.speed_command
        return GateBatch(scenes.features(), scenes, anchors, lower, upper, total)

    def decode(self, raw: torch.Tensor, batch: GateBatch):
        width = self.width_min + (self.width_max - self.width_min) * raw[
            :, : self.geometry
        ].sigmoid().reshape(self.batch_size, gate.SGATE3.pieces, gate.FACES)
        caps = raw.new_tensor(gate.PHYSICAL_CAPS) * (
            0.25 + 0.75 * raw[:, self.geometry :].sigmoid()
        )
        normals = raw.new_tensor([[1, 0], [-1, 0], [0, 1], [0, -1]]).expand(
            self.batch_size, gate.SGATE3.pieces, -1, -1
        )
        support = (
            torch.stack(
                (
                    batch.upper[..., 0],
                    -batch.lower[..., 0],
                    batch.upper[..., 1],
                    -batch.lower[..., 1],
                ),
                dim=-1,
            )
            + width
        )
        corridor = torch.cat((normals, -support[..., None]), dim=-1)
        return corridor, caps, width

    def bindings(self, mode: str, raw: torch.Tensor, batch: GateBatch):
        corridor, caps, _ = self.decode(raw, batch)
        head = torch.from_numpy(
            p.unicycle_state(
                batch.scenes.start.numpy(),
                yaw=batch.scenes.start_yaw.numpy(),
                acceleration=gate.REST_ACCELERATION,
            )
        )
        tail = torch.from_numpy(
            p.unicycle_state(
                batch.scenes.goal.numpy(),
                yaw=batch.scenes.goal_yaw.numpy(),
                acceleration=-gate.REST_ACCELERATION,
            )
        )
        plan = self.plans[mode]
        initial = plan.initial_guess(
            waypoints=batch.anchors[:, 1:-1].numpy(),
            durations=np.broadcast_to(
                (batch.total / 8)[:, None].numpy(), (self.batch_size, 8)
            ).copy(),
        )
        return {
            "head": head,
            "tail": tail,
            "corridor": corridor,
            "total_duration": batch.total,
            "initial": torch.from_numpy(initial),
            **{
                name: caps[:, index].contiguous()
                for index, name in enumerate(gate.CAP_NAMES)
            },
        }

    def inputs(self, mode: str, bindings) -> p.Inputs:
        return self.plans[mode].inputs(
            **{name: value.detach().numpy().copy() for name, value in bindings.items()}
        )

    def losses(self, result, raw: torch.Tensor, batch: GateBatch):
        samples = {
            order: gate.sample_derivative(
                result.coefficients, result.durations, order, intervals=32
            )
            for order in (0, 1, 2, 3)
        }

        def average(value):
            return (torch.trapezoid(value, dx=1 / 32, dim=-1) * result.durations).sum(
                -1
            ) / result.durations.sum(-1)

        speed = torch.linalg.vector_norm(samples[1], dim=-1)
        tracking = average(
            (
                (speed - batch.scenes.speed_command[:, None, None])
                / batch.scenes.speed_command[:, None, None]
            ).square()
        )
        acceleration = average((samples[2] / gate.ACCELERATION_SCALE).square().sum(-1))
        jerk = average((samples[3] / gate.JERK_SCALE).square().sum(-1))
        smoothness = 10**batch.scenes.smoothness
        opening = gate.trajectory_gate_loss(samples[0], batch.scenes)
        _, caps, width = self.decode(raw, batch)
        direct = 0.001 * (caps / raw.new_tensor(gate.PHYSICAL_CAPS)).square().mean(
            -1
        ) + 0.0001 * (width / self.width_max).square().mean((1, 2))
        task = (
            tracking
            + 0.03 * smoothness * acceleration
            + 0.02 * smoothness * jerk
            + 10 * opening
        )
        return task, direct

    def certify(self, batch: GateBatch):
        low = (batch.lower - self.width_max).numpy()
        high = (batch.upper + self.width_max).numpy()
        centers = batch.scenes.wall_centers.numpy()
        extents = batch.scenes.wall_half_extents.numpy() + gate.BODY_RADIUS
        walls_low = centers - extents
        walls_high = centers + extents
        separation = np.maximum(
            walls_low[:, None] - high[:, :, None],
            low[:, :, None] - walls_high[:, None],
        )
        margin = separation.max(axis=-1)
        return {
            "all_boxes_free": bool((margin > 0).all()),
            "minimum_separation": float(margin.min()),
        }


def pack_batch(batch: GateBatch) -> dict[str, np.ndarray]:
    arrays = {
        name: getattr(batch, name).numpy()
        for name in ("features", "anchors", "lower", "upper", "total")
    }
    arrays.update(
        {
            "scene." + field.name: getattr(batch.scenes, field.name).numpy()
            for field in dc.fields(batch.scenes)
        }
    )
    return arrays


def save_batch(path: Path, batch: GateBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **pack_batch(batch))
    temporary.replace(path)


def read_batch(path: Path) -> GateBatch:
    with np.load(path) as archive:
        values = {
            name: torch.from_numpy(archive[name].copy()) for name in archive.files
        }
    scenes = gate.SceneBatch(
        **{
            name.removeprefix("scene."): value
            for name, value in values.items()
            if name.startswith("scene.")
        }
    )
    return GateBatch(
        scenes=scenes,
        **{
            name: value
            for name, value in values.items()
            if not name.startswith("scene.")
        },
    )


def torch_result(result, *, gradients: bool):
    values = {
        name: torch.from_numpy(np.asarray(getattr(result, name)).copy()).requires_grad_(
            gradients
        )
        for name in RESULT_FIELDS
    }
    return (
        SimpleNamespace(
            **values,
            status=torch.from_numpy(np.asarray(result.status).copy()),
        ),
        values,
    )


def result_seeds(gradients, values) -> p.ResultSeeds:
    arrays = {}
    for name, gradient in zip(RESULT_FIELDS, gradients, strict=True):
        value = torch.zeros_like(values[name]) if gradient is None else gradient
        arrays[name] = value.detach().numpy().copy()
    return p.ResultSeeds(**arrays)


def finite_mean(value) -> float | None:
    array = np.asarray(value)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def evaluate(
    task: CertifiedGateTask,
    model: LearningPolicy,
    batch: GateBatch,
    *,
    mode: str,
):
    with torch.no_grad():
        raw = model(batch.features)
        bindings = task.bindings(mode, raw, batch)
        solver = cpu.build_solver(task.plans[mode], dtype=np.float64)
        actual = solver.solve(task.inputs(mode, bindings))
        record, _ = torch_result(actual, gradients=False)
        task_loss, direct = task.losses(record, raw, batch)
        available = (
            actual.status == int(ForwardStatus.RESULT_AVAILABLE)
        ) & np.isfinite(task_loss.numpy())
        traversal = gate.gate_traversal(
            record, batch.scenes, collision_method="continuous"
        )
        motion = gate.exact_motion(record, intervals=128)
        omega = torch.where(
            motion["moving"], motion["yaw_rate"], motion["low_speed_yaw_rate"]
        )
        wheels = torch.stack(
            (
                motion["speed"] - 0.5 * gate.TRACK_WIDTH * omega,
                motion["speed"] + 0.5 * gate.TRACK_WIDTH * omega,
            ),
            dim=-1,
        )

        def maximum(value):
            return value.abs().flatten(1).amax(-1).numpy()

        maxima = {
            "speed": maximum(motion["speed"]),
            "centripetal": maximum(motion["centripetal"]),
            "forward_acceleration": maximum(motion["tangential"]),
            "wheel": maximum(wheels),
        }
        physical = np.ones(task.batch_size, dtype=bool)
        for name, limit in zip(
            ("speed", "centripetal", "forward_acceleration", "wheel"),
            (*gate.PHYSICAL_CAPS, gate.WHEEL_LIMIT),
            strict=True,
        ):
            physical &= np.isfinite(maxima[name]) & (maxima[name] <= 1.01 * limit)
        geometry = np.asarray(traversal["gate_success"], dtype=bool)
        feasible = available & geometry & physical
        corridor, caps, _ = task.decode(raw, batch)
        violations = gate.constraint_violations(
            record, SimpleNamespace(corridor=corridor, caps=caps), intervals=128
        )
        utility = np.where(feasible, np.exp(-task_loss.numpy()), 0.0)
        return {
            "mean_utility": float(utility.mean()),
            "mean_task": finite_mean(task_loss.numpy()),
            "feasible_fraction": float(feasible.mean()),
            "numerical_failure_fraction": float((~available).mean()),
            "utility": utility,
            "task": task_loss.numpy(),
            "direct": direct.numpy(),
            "feasible": feasible,
            "status": actual.status,
            "continuous_collision": traversal["continuous_collision"],
            "gate_success": geometry,
            "sampled_physical_feasible": physical,
            "sampled_physical_maxima": maxima,
            "solver_constraint_maxima": {
                name: value.flatten(1).amax(-1).numpy()
                for name, value in violations.items()
            },
            "corridor_certificate": task.certify(batch),
        }


def training_step(
    task: CertifiedGateTask,
    solver: cpu.Solver,
    model: LearningPolicy,
    optimizer: torch.optim.Optimizer,
    batch: GateBatch,
    *,
    mode: str,
    task_scale: float,
):
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    raw = model(batch.features)
    bindings = task.bindings(mode, raw, batch)
    result, state = solver.solve_with_state(task.inputs(mode, bindings))
    if np.any(result.status != int(ForwardStatus.RESULT_AVAILABLE)):
        return {
            "updated": False,
            "reason": "forward_failure",
            "forward_status": result.status,
            "mode": mode,
            "wall_seconds": time.perf_counter() - started,
        }
    readout, leaf_values = torch_result(result, gradients=True)
    task_loss, direct = task.losses(readout, raw, batch)
    loss = task_scale * task_loss + direct
    if not bool(torch.isfinite(loss).all()):
        return {
            "updated": False,
            "reason": "nonfinite_outer_task",
            "mode": mode,
            "wall_seconds": time.perf_counter() - started,
        }
    gradients = torch.autograd.grad(
        loss.sum(),
        (*[leaf_values[name] for name in RESULT_FIELDS], raw),
        allow_unused=True,
        retain_graph=True,
    )
    direct_raw = torch.zeros_like(raw) if gradients[-1] is None else gradients[-1]
    pulled = solver.pullback(state, result_seeds(gradients[:-1], leaf_values))
    if np.any(pulled.diagnostics.status != 0):
        return {
            "updated": False,
            "reason": "backward_failure",
            "status": pulled.diagnostics.status,
            "cause_domain": pulled.diagnostics.cause_domain,
            "cause_code": pulled.diagnostics.cause_code,
            "mode": mode,
            "wall_seconds": time.perf_counter() - started,
        }
    active = [(name, value) for name, value in bindings.items() if value.requires_grad]
    response_raw = torch.autograd.grad(
        tuple(value for _, value in active),
        raw,
        grad_outputs=tuple(
            torch.from_numpy(pulled.parameters[name]) for name, _ in active
        ),
        retain_graph=True,
    )[0]
    raw.backward((response_raw + direct_raw) / task.batch_size)
    parameters = [value.grad for value in model.parameters() if value.grad is not None]
    gradient_norm = torch.nn.utils.get_total_norm(parameters)
    if not bool(torch.isfinite(gradient_norm)):
        return {
            "updated": False,
            "reason": "nonfinite_network_gradient",
            "mode": mode,
            "wall_seconds": time.perf_counter() - started,
        }
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    response_norm = torch.linalg.vector_norm(response_raw, dim=1)
    direct_norm = torch.linalg.vector_norm(direct_raw, dim=1)
    diagnostics = {
        name: np.asarray(getattr(pulled.diagnostics, name)).copy()
        for name in DIAGNOSTIC_FIELDS
    }
    return {
        "updated": True,
        "mode": mode,
        "task_scale": task_scale,
        "task": float(task_loss.detach().mean()),
        "direct_loss": float(direct.detach().mean()),
        "loss": float(loss.detach().mean()),
        "gradient_norm": float(gradient_norm),
        "response_raw_norm": float(response_norm.mean()),
        "direct_raw_norm": float(direct_norm.mean()),
        "effective_response_fraction": float((response_norm > 0).double().mean()),
        "forward_objective_evaluations": result.diagnostics.objective_evaluations,
        "response_displacement": finite_mean(pulled.diagnostics.response_displacement),
        "wall_seconds": time.perf_counter() - started,
        "diagnostic_arrays": diagnostics,
    }


def model_digest(model: LearningPolicy) -> str:
    values = torch.cat(
        [parameter.detach().flatten() for parameter in model.parameters()]
    ).numpy()
    return array_hash(values)


def save_checkpoint(
    path: Path,
    model: LearningPolicy,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    seed: int,
    updates: int,
    attempts: int,
) -> None:
    torch.save(
        {
            "policy": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "updates": updates,
            "attempts": attempts,
            "seed": seed,
        },
        path,
    )


def evaluate_checkpoints(
    destination: Path,
    method: Method,
    task: CertifiedGateTask,
    model: LearningPolicy,
    test_batch: GateBatch,
    *,
    time_checkpoint: Path | None,
):
    evaluation_started = time.perf_counter()
    curve = []
    for updates in CHECKPOINT_UPDATES:
        checkpoint = destination / f"checkpoint-{updates:06d}.pt"
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(state["policy"])
        mode = method.mode(max(0, updates - 1))
        metrics = evaluate(task, model, test_batch, mode=mode)
        write_json(destination / f"test-{updates:06d}.json", metrics)
        curve.append(
            {
                "updates": updates,
                **{
                    name: metrics[name]
                    for name in (
                        "mean_utility",
                        "mean_task",
                        "feasible_fraction",
                        "numerical_failure_fraction",
                    )
                },
            }
        )
    timed = None
    if time_checkpoint is not None and time_checkpoint.exists():
        state = torch.load(time_checkpoint, weights_only=True)
        model.load_state_dict(state["policy"])
        mode = method.mode(max(0, int(state["updates"]) - 1))
        metrics = evaluate(task, model, test_batch, mode=mode)
        write_json(destination / "test-time-budget.json", metrics)
        timed = {
            "updates": int(state["updates"]),
            **{
                name: metrics[name]
                for name in (
                    "mean_utility",
                    "mean_task",
                    "feasible_fraction",
                    "numerical_failure_fraction",
                )
            },
        }
    by_update = {row["updates"]: row for row in curve}
    required = all(update in by_update for update in CHECKPOINT_UPDATES)
    if required:
        x = np.array(CHECKPOINT_UPDATES)
        y = np.array([by_update[update]["mean_utility"] for update in x])
        auc = float(np.trapezoid(y, x) / 1600)
        final = float(by_update[1600]["mean_utility"])
    else:
        auc = final = 0.0
    endpoints = [auc, final, 0.0 if timed is None else timed["mean_utility"]]
    return {
        "curve": curve,
        "timed": timed,
        "endpoints": endpoints,
        "required_update_budget_complete": required,
        "required_time_budget_complete": timed is not None,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
    }


def train_one(
    output: Path,
    method: Method,
    seed: int,
    *,
    test_batch: GateBatch,
    validation_batch: GateBatch,
    successful_updates: int = 1600,
    time_budget_seconds: float | None = TIME_BUDGET_SECONDS,
    maximum_successful_updates: int = 4800,
    maximum_attempts: int = 9600,
    consecutive_failure_limit: int = 20,
    validation_frequency: int = 100,
    checkpoint_frequency: int = 200,
):
    destination = output / "runs" / f"{method.tag}-seed{seed}"
    summary_path = destination / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    if destination.exists():
        raise RuntimeError(f"partial run retained for inspection: {destination}")

    setup_started = time.perf_counter()
    torch.manual_seed(seed)
    task = CertifiedGateTask(16, method.modes)
    solvers = {
        mode: cpu.build_solver(task.plans[mode], dtype=np.float64, backward=True)
        for mode in method.modes
    }
    model = LearningPolicy(task.features, task.outputs, geometry=task.geometry)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 100000)
    destination.mkdir(parents=True)
    initial_digest = model_digest(model)
    torch.save({"policy": model.state_dict(), "seed": seed}, destination / "initial.pt")
    setup_seconds = time.perf_counter() - setup_started
    started = setup_started
    successful = attempts = failures = consecutive = 0
    validation = []
    diagnostics = {name: [] for name in DIAGNOSTIC_FIELDS}
    available = {
        "policy": copy.deepcopy(model.state_dict()),
        "updates": 0,
        "attempt": 0,
        "model_available_seconds": setup_seconds,
    }
    timed = None
    update_budget_seconds = None

    validation_task = CertifiedGateTask(64, method.modes)
    test_task = CertifiedGateTask(256, method.modes)

    def checkpoint() -> None:
        save_checkpoint(
            destination / f"checkpoint-{successful:06d}.pt",
            model,
            optimizer,
            generator,
            seed=seed,
            updates=successful,
            attempts=attempts,
        )

    def assess() -> None:
        mode = method.mode(max(0, successful - 1))
        metrics = evaluate(validation_task, model, validation_batch, mode=mode)
        row = {
            "updates": successful,
            "attempts": attempts,
            "elapsed_seconds": time.perf_counter() - started,
            **{
                name: metrics[name]
                for name in (
                    "mean_utility",
                    "mean_task",
                    "feasible_fraction",
                    "numerical_failure_fraction",
                )
            },
        }
        validation.append(row)
        write_json(destination / f"validation-{successful:06d}.json", metrics)

    def capture_time() -> None:
        nonlocal timed
        if (
            time_budget_seconds is not None
            and timed is None
            and time.perf_counter() - started >= time_budget_seconds
        ):
            timed = available
            torch.save(timed, destination / "time-budget.pt")
            write_json(
                destination / "time-budget.json",
                {name: value for name, value in timed.items() if name != "policy"}
                | {
                    "budget_seconds": time_budget_seconds,
                    "gap_seconds": time_budget_seconds
                    - timed["model_available_seconds"],
                    "captured_wall_seconds": time.perf_counter() - started,
                },
            )

    checkpoint()
    assess()
    reason = None
    while True:
        capture_time()
        time_complete = time_budget_seconds is None or timed is not None
        if successful >= successful_updates and time_complete:
            reason = "required_budgets_complete"
            break
        if successful >= maximum_successful_updates:
            reason = "successful_update_cap"
            break
        if attempts >= maximum_attempts:
            reason = "attempt_cap"
            break
        if consecutive >= consecutive_failure_limit:
            reason = "consecutive_failure_limit"
            break
        data_started = time.perf_counter()
        batch = task.sample(generator)
        certificate = task.certify(batch)
        if not certificate["all_boxes_free"]:
            raise RuntimeError("unsafe decoder envelope")
        data_seconds = time.perf_counter() - data_started
        mode = method.mode(successful)
        record = training_step(
            task,
            solvers[mode],
            model,
            optimizer,
            batch,
            mode=mode,
            task_scale=method.task_scale,
        )
        attempts += 1
        if record["updated"]:
            successful += 1
            consecutive = 0
            for name, values in record.pop("diagnostic_arrays").items():
                if mode == "RS":
                    diagnostics[name].append(values)
        else:
            failures += 1
            consecutive += 1
        ready = time.perf_counter() - started
        if (
            record["updated"]
            and time_budget_seconds is not None
            and ready <= time_budget_seconds
        ):
            available = {
                "policy": copy.deepcopy(model.state_dict()),
                "updates": successful,
                "attempt": attempts,
                "model_available_seconds": ready,
            }
        record.update(
            {
                "attempt": attempts,
                "updates": successful,
                "data_seconds": data_seconds,
                "elapsed_seconds": ready,
                "features_sha256": array_hash(batch.features.numpy()),
            }
        )
        with (destination / "curve.jsonl").open("a") as stream:
            stream.write(json.dumps(clean(record), allow_nan=False) + "\n")
        if record["updated"] and successful % validation_frequency == 0:
            assess()
        if record["updated"] and successful % checkpoint_frequency == 0:
            checkpoint()
        if successful == successful_updates and update_budget_seconds is None:
            update_budget_seconds = time.perf_counter() - started
        if attempts % 400 == 0:
            print(
                method.tag,
                seed,
                successful,
                round(time.perf_counter() - started, 3),
                flush=True,
            )
    capture_time()
    save_checkpoint(
        destination / "final.pt",
        model,
        optimizer,
        generator,
        seed=seed,
        updates=successful,
        attempts=attempts,
    )
    training_wall_seconds = time.perf_counter() - started
    np.savez_compressed(
        destination / "response-diagnostics.npz",
        **{
            name: np.concatenate(values) if values else np.empty(0)
            for name, values in diagnostics.items()
        },
    )
    evaluation = evaluate_checkpoints(
        destination,
        method,
        test_task,
        model,
        test_batch,
        time_checkpoint=(destination / "time-budget.pt") if timed else None,
    )
    summary = {
        "method": dc.asdict(method),
        "seed": seed,
        "updated": successful,
        "attempts": attempts,
        "failed_batches": failures,
        "termination": reason,
        "setup_seconds": setup_seconds,
        "training_wall_seconds": training_wall_seconds,
        "evaluation_seconds": evaluation["evaluation_seconds"],
        "complete_wall_seconds": time.perf_counter() - started,
        "update_budget_seconds": update_budget_seconds,
        "time_budget_seconds": time_budget_seconds,
        "time_checkpoint_updates": None if timed is None else timed["updates"],
        "initial_model_tensor_sha256": initial_digest,
        "source_sha256": source_hash(),
        "validation": validation,
        "test_curve": evaluation["curve"],
        "timed_test": evaluation["timed"],
        "endpoints": evaluation["endpoints"],
        "update_budget_complete": evaluation["required_update_budget_complete"],
        "time_budget_complete": evaluation["required_time_budget_complete"],
    }
    write_json(summary_path, summary)
    print(
        "completed",
        method.tag,
        seed,
        successful,
        failures,
        summary["endpoints"],
        flush=True,
    )
    return summary


def freeze_datasets(output: Path, seeds: list[int]) -> dict[str, str]:
    datasets = output / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    task = CertifiedGateTask(256, ("A",))
    test_path = datasets / "gate-test.npz"
    if not test_path.exists():
        save_batch(
            test_path,
            task.sample(torch.Generator().manual_seed(TEST_SEED)),
        )
    paths = {"test": str(test_path)}
    for seed in seeds:
        path = datasets / f"gate-validation-seed{seed}.npz"
        if not path.exists():
            validation_task = CertifiedGateTask(64, ("A",))
            save_batch(
                path,
                validation_task.sample(torch.Generator().manual_seed(seed + 200000)),
            )
        paths[str(seed)] = str(path)
    return paths


def freeze_design(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    design_path = output / "formal-design.json"
    if design_path.exists():
        raise FileExistsError("formal design is immutable")
    seeds = list(range(20001, 20011))
    datasets = freeze_datasets(output, seeds)
    design = {
        "kind": "certified-gate-perturbation-followup",
        "source_sha256": source_hash(),
        "methods": [dc.asdict(method) for method in METHODS.values()],
        "seeds": seeds,
        "batch_size": 16,
        "successful_updates": 1600,
        "maximum_successful_updates": 4800,
        "maximum_attempts": 9600,
        "consecutive_failure_limit": 20,
        "validation_batch_size": 64,
        "validation_frequency": 100,
        "checkpoint_frequency": 200,
        "test_batch_size": 256,
        "test_seed": TEST_SEED,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "learning_rate": 0.01,
        "dtype": "float64",
        "curvature": "exact",
        "damping": "MinimumShift",
        "backward_checks": {
            "stationarity_tolerance": None,
            "hessian_symmetry_tolerance": None,
            "hessian_pivot_tolerance": None,
            "adjoint_residual_tolerance": None,
        },
        "method_order": "rotate the listed methods by seed index; runs remain serial",
        "statistics": {
            "replicates": 10000,
            "seed": 20261021,
            "interval": "two-sided simultaneous max-t within each experiment family",
            "retention_ratio": "ratio of paired mean effects; shared-seed bootstrap and simultaneous max-t interval",
        },
        "datasets": {
            name: {"path": path, "sha256": sha256(Path(path))}
            for name, path in datasets.items()
        },
    }
    write_json(design_path, design)
    print(design_path)


def load_design(output: Path):
    design = json.loads((output / "formal-design.json").read_text())
    if design["source_sha256"] != source_hash():
        raise RuntimeError("the frozen design does not match this driver source")
    for dataset in design["datasets"].values():
        path = Path(dataset["path"])
        if sha256(path) != dataset["sha256"]:
            raise RuntimeError(f"frozen dataset changed: {path}")
    return design


def train_design(
    output: Path,
    *,
    selected_seeds: set[int] | None = None,
    selected_methods: set[str] | None = None,
) -> None:
    design = load_design(output)
    seeds = design["seeds"]
    test_batch = read_batch(Path(design["datasets"]["test"]["path"]))
    methods = list(METHODS.values())
    for index, seed in enumerate(seeds):
        if selected_seeds is not None and seed not in selected_seeds:
            continue
        offset = index % len(methods)
        ordered = methods[offset:] + methods[:offset]
        validation = read_batch(Path(design["datasets"][str(seed)]["path"]))
        for method in ordered:
            if selected_methods is not None and method.tag not in selected_methods:
                continue
            train_one(
                output,
                method,
                seed,
                test_batch=test_batch,
                validation_batch=validation,
                successful_updates=design["successful_updates"],
                time_budget_seconds=design["time_budget_seconds"],
                maximum_successful_updates=design["maximum_successful_updates"],
                maximum_attempts=design["maximum_attempts"],
                consecutive_failure_limit=design["consecutive_failure_limit"],
                validation_frequency=design["validation_frequency"],
                checkpoint_frequency=design["checkpoint_frequency"],
            )


def equal_arrays(left: GateBatch, right: GateBatch) -> bool:
    for name in ("features", "anchors", "lower", "upper", "total"):
        if not torch.equal(getattr(left, name), getattr(right, name)):
            return False
    return all(
        torch.equal(getattr(left.scenes, field.name), getattr(right.scenes, field.name))
        for field in dc.fields(left.scenes)
    )


def compare_state_dicts(actual: Path, expected: Path) -> bool:
    left = torch.load(actual, weights_only=True)["policy"]
    right = torch.load(expected, weights_only=True)["policy"]
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def deterministic_curve(path: Path):
    keys = (
        "updated",
        "reason",
        "task",
        "direct_loss",
        "loss",
        "gradient_norm",
        "response_raw_norm",
        "direct_raw_norm",
        "effective_response_fraction",
        "forward_objective_evaluations",
        "response_displacement",
        "attempt",
        "updates",
        "features_sha256",
    )
    return [
        {key: row.get(key) for key in keys if key in row}
        for row in map(json.loads, path.read_text().splitlines())
    ]


def reproduce(output: Path, historical: Path) -> None:
    if output.exists():
        raise FileExistsError(f"reproduction output already exists: {output}")
    output.mkdir(parents=True)
    seeds = [19001, 19002]
    paths = freeze_datasets(output, seeds)
    old_datasets = historical / "datasets"
    dataset_checks = {
        "test": equal_arrays(
            read_batch(Path(paths["test"])), read_batch(old_datasets / "gate-test.npz")
        )
    }
    for seed in seeds:
        dataset_checks[str(seed)] = equal_arrays(
            read_batch(Path(paths[str(seed)])),
            read_batch(old_datasets / f"gate-validation-seed{seed}.npz"),
        )
    if not all(dataset_checks.values()):
        raise AssertionError("regenerated frozen datasets differ")
    test_batch = read_batch(Path(paths["test"]))
    rows = []
    for seed in seeds:
        validation = read_batch(Path(paths[str(seed)]))
        for tag in ("A", "RS"):
            method = METHODS[tag]
            current = train_one(
                output,
                method,
                seed,
                test_batch=test_batch,
                validation_batch=validation,
                time_budget_seconds=None,
            )
            current_dir = output / "runs" / f"{tag}-seed{seed}"
            old_dir = historical / "formal" / f"gate-{tag}-seed{seed}"
            checkpoints = {
                str(update): compare_state_dicts(
                    current_dir / f"checkpoint-{update:06d}.pt",
                    old_dir / f"checkpoint-{update:06d}.pt",
                )
                for update in CHECKPOINT_UPDATES
            }
            current_curve = deterministic_curve(current_dir / "curve.jsonl")
            old_curve = deterministic_curve(old_dir / "curve.jsonl")[:1600]
            curve_exact = current_curve == old_curve
            old_summary = json.loads((old_dir / "summary.json").read_text())
            validation_exact = [
                row["mean_utility"] for row in current["validation"]
            ] == [
                row["mean_utility"]
                for row in old_summary["validation"]
                if row["updates"] <= 1600
            ]
            old_test = json.loads(
                (
                    historical
                    / "evaluations"
                    / f"gate-{tag}-seed{seed}"
                    / "updates-001600.json"
                ).read_text()
            )["mean_utility"]
            new_test = current["endpoints"][1]
            utility_difference = abs(new_test - old_test)
            row = {
                "seed": seed,
                "method": tag,
                "checkpoint_tensors_bitwise": checkpoints,
                "training_curve_exact": curve_exact,
                "validation_utility_exact": validation_exact,
                "historical_final_test_utility": old_test,
                "current_final_test_utility": new_test,
                "final_test_utility_difference": utility_difference,
            }
            rows.append(row)
            if not (
                all(checkpoints.values())
                and curve_exact
                and validation_exact
                and utility_difference < 1e-9
            ):
                write_json(output / "reproduction.json", {"rows": rows})
                raise AssertionError(f"frozen reproduction differs: {tag} seed {seed}")
    write_json(
        output / "reproduction.json",
        {
            "datasets_bitwise": dataset_checks,
            "rows": rows,
            "accepted": True,
            "criterion": "checkpoint tensors, deterministic training curve and validation utility exact; final test utility difference < 1e-9",
            "source_sha256": source_hash(),
        },
    )
    print(json.dumps(clean(rows), indent=2))


def max_t_intervals(
    values: np.ndarray,
    *,
    names: list[dict],
    bootstrap_indices: np.ndarray,
):
    from scipy.stats import t

    means = values.mean(axis=1)
    standard_errors = values.std(axis=1, ddof=1) / math.sqrt(values.shape[1])
    bootstrap = values[:, bootstrap_indices].mean(axis=2)
    varying = standard_errors > 0
    standardized = np.zeros_like(bootstrap)
    standardized[varying] = (
        bootstrap[varying] - means[varying, None]
    ) / standard_errors[varying, None]
    critical = float(np.quantile(np.abs(standardized[varying]).max(axis=0), 0.95))
    rows = []
    count = len(values)
    for index, name in enumerate(names):
        interval = (
            [
                float(means[index] - critical * standard_errors[index]),
                float(means[index] + critical * standard_errors[index]),
            ]
            if varying[index]
            else [float(means[index]), float(means[index])]
        )
        sensitivity = (
            float(
                means[index]
                - t.ppf(1 - 0.05 / count, values.shape[1] - 1) * standard_errors[index]
            )
            if varying[index]
            else float(means[index])
        )
        rows.append(
            name
            | {
                "paired_seed_effects": values[index],
                "estimate": float(means[index]),
                "standard_error": float(standard_errors[index]),
                "simultaneous_two_sided_95": interval,
                "bonferroni_paired_t_sensitivity_lower": sensitivity,
            }
        )
    return rows, critical


def ratio_intervals(
    numerators: list[np.ndarray],
    denominator: np.ndarray,
    *,
    names: list[dict],
    bootstrap_indices: np.ndarray,
):
    denominator_bootstrap = denominator[bootstrap_indices].mean(axis=1)
    estimates = []
    bootstraps = []
    for numerator in numerators:
        estimates.append(float(numerator.mean() / denominator.mean()))
        sampled = numerator[bootstrap_indices].mean(axis=1)
        bootstraps.append(sampled / denominator_bootstrap)
    bootstrap = np.stack(bootstraps)
    estimates = np.asarray(estimates)
    standard_errors = bootstrap.std(axis=1, ddof=1)
    varying = np.isfinite(standard_errors) & (standard_errors > 0)
    standardized = np.zeros_like(bootstrap)
    standardized[varying] = (
        bootstrap[varying] - estimates[varying, None]
    ) / standard_errors[varying, None]
    critical = float(np.quantile(np.abs(standardized[varying]).max(axis=0), 0.95))
    rows = []
    for index, name in enumerate(names):
        rows.append(
            name
            | {
                "estimate": float(estimates[index]),
                "bootstrap_standard_error": float(standard_errors[index]),
                "simultaneous_two_sided_95": [
                    float(estimates[index] - critical * standard_errors[index]),
                    float(estimates[index] + critical * standard_errors[index]),
                ],
            }
        )
    return rows, critical


def diagnostic_summary(paths: list[Path]):
    result = {}
    for field in DIAGNOSTIC_FIELDS:
        arrays = []
        for path in paths:
            with np.load(path) as archive:
                values = archive[field]
                arrays.append(values[np.isfinite(values)])
        values = np.concatenate(arrays) if arrays else np.empty(0)
        result[field] = {
            "count": int(values.size),
            "quantiles": dict(
                zip(
                    ("min", "p05", "p25", "median", "p75", "p95", "max"),
                    np.quantile(values, (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1))
                    if values.size
                    else [None] * 7,
                    strict=True,
                )
            ),
        }
    return result


def forest_plot(path: Path, rows: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['comparison']} · {row['endpoint']}" for row in rows]
    estimates = 100 * np.array([row["estimate"] for row in rows])
    intervals = 100 * np.array([row["simultaneous_two_sided_95"] for row in rows])
    positions = np.arange(len(rows))[::-1]
    figure, axis = plt.subplots(figsize=(10, max(5, 0.33 * len(rows))))
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates)),
        fmt="o",
        capsize=3,
        color="#1f5c99",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("paired utility effect (percentage points)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def analyze(output: Path) -> None:
    design = load_design(output)
    seeds = design["seeds"]
    summaries = {}
    run_rows = []
    for tag, method in METHODS.items():
        diagnostics = []
        for seed in seeds:
            path = output / "runs" / f"{tag}-seed{seed}"
            summary = json.loads((path / "summary.json").read_text())
            summaries[(tag, seed)] = np.asarray(summary["endpoints"], dtype=float)
            diagnostics.append(path / "response-diagnostics.npz")
            run_rows.append(
                {
                    "method": tag,
                    "label": method.label,
                    "seed": seed,
                    "endpoints": summary["endpoints"],
                    "training_wall_seconds": summary["training_wall_seconds"],
                    "complete_wall_seconds": summary["complete_wall_seconds"],
                    "updates": summary["updated"],
                    "attempts": summary["attempts"],
                    "failed_batches": summary["failed_batches"],
                    "final_feasible_fraction": summary["test_curve"][-1][
                        "feasible_fraction"
                    ],
                }
            )
    generator = np.random.default_rng(design["statistics"]["seed"])
    bootstrap_indices = generator.integers(
        0,
        len(seeds),
        (design["statistics"]["replicates"], len(seeds)),
    )

    def differences(candidate: str, reference: str):
        return np.stack(
            [
                summaries[(candidate, seed)] - summaries[(reference, seed)]
                for seed in seeds
            ]
        )

    magnitude_names = []
    magnitude_values = []
    magnitude_pairs = [("RS", "A")]
    magnitude_pairs += [(f"A-k{k}", "A") for k in ("0.8", "1.25", "1.5")]
    magnitude_pairs += [("RS", f"A-k{k}") for k in ("0.8", "1.25", "1.5")]
    for candidate, reference in magnitude_pairs:
        values = differences(candidate, reference)
        for column, endpoint in enumerate(ENDPOINTS):
            magnitude_names.append(
                {
                    "comparison": f"{candidate} - {reference}",
                    "candidate": candidate,
                    "reference": reference,
                    "endpoint": endpoint,
                }
            )
            magnitude_values.append(values[:, column])
    magnitude_rows, magnitude_critical = max_t_intervals(
        np.stack(magnitude_values),
        names=magnitude_names,
        bootstrap_indices=bootstrap_indices,
    )

    switch_tags = [
        tag for tag in METHODS if tag.startswith("RS-to-A") or tag.startswith("A-to-RS")
    ]
    switch_names = []
    switch_values = []
    base = differences("RS", "A")
    for column, endpoint in enumerate(ENDPOINTS):
        switch_names.append(
            {
                "comparison": "RS - A",
                "candidate": "RS",
                "reference": "A",
                "endpoint": endpoint,
            }
        )
        switch_values.append(base[:, column])
    for tag in switch_tags:
        values = differences(tag, "A")
        for column, endpoint in enumerate(ENDPOINTS):
            switch_names.append(
                {
                    "comparison": f"{tag} - A",
                    "candidate": tag,
                    "reference": "A",
                    "endpoint": endpoint,
                }
            )
            switch_values.append(values[:, column])
    switch_rows, switch_critical = max_t_intervals(
        np.stack(switch_values),
        names=switch_names,
        bootstrap_indices=bootstrap_indices,
    )
    base_rows = {
        row["endpoint"]: row for row in switch_rows if row["comparison"] == "RS - A"
    }
    ratio_names = []
    ratio_numerators = []
    ratio_denominators = []
    for tag in switch_tags:
        values = differences(tag, "A")
        for column, endpoint in enumerate(ENDPOINTS):
            if (
                base_rows[endpoint]["simultaneous_two_sided_95"][0]
                <= 0
                <= base_rows[endpoint]["simultaneous_two_sided_95"][1]
            ):
                continue
            ratio_names.append(
                {
                    "comparison": f"retention({tag})",
                    "method": tag,
                    "endpoint": endpoint,
                }
            )
            ratio_numerators.append(values[:, column])
            ratio_denominators.append(base[:, column])
    ratio_rows = []
    ratio_critical = None
    if ratio_names:
        # Every endpoint uses its matching RS-A denominator.  Compute all
        # shared draws first, then one simultaneous maximum over the ratios.
        estimates, bootstraps = [], []
        for numerator, denominator in zip(
            ratio_numerators, ratio_denominators, strict=True
        ):
            estimates.append(float(numerator.mean() / denominator.mean()))
            bootstraps.append(
                numerator[bootstrap_indices].mean(axis=1)
                / denominator[bootstrap_indices].mean(axis=1)
            )
        estimates = np.asarray(estimates)
        bootstraps = np.stack(bootstraps)
        standard_errors = bootstraps.std(axis=1, ddof=1)
        standardized = (bootstraps - estimates[:, None]) / standard_errors[:, None]
        ratio_critical = float(np.quantile(np.abs(standardized).max(axis=0), 0.95))
        ratio_rows = [
            name
            | {
                "estimate": float(estimates[index]),
                "bootstrap_standard_error": float(standard_errors[index]),
                "simultaneous_two_sided_95": [
                    float(estimates[index] - ratio_critical * standard_errors[index]),
                    float(estimates[index] + ratio_critical * standard_errors[index]),
                ],
            }
            for index, name in enumerate(ratio_names)
        ]

    diagnostic_methods = {}
    for tag in METHODS:
        paths = [
            output / "runs" / f"{tag}-seed{seed}" / "response-diagnostics.npz"
            for seed in seeds
        ]
        diagnostic_methods[tag] = diagnostic_summary(paths)
    report = {
        "design_sha256": sha256(output / "formal-design.json"),
        "source_sha256": source_hash(),
        "endpoints": ENDPOINTS,
        "seeds": seeds,
        "runs": run_rows,
        "magnitude_family": {
            "critical": magnitude_critical,
            "comparisons": magnitude_rows,
        },
        "switch_family": {
            "critical": switch_critical,
            "comparisons": switch_rows,
            "retention_ratio_critical": ratio_critical,
            "retention_ratios": ratio_rows,
        },
        "response_diagnostics": diagnostic_methods,
    }
    write_json(output / "analysis.json", report)
    forest_plot(
        output / "forest-magnitude",
        magnitude_rows,
        "Certified gate: response magnitude comparisons",
    )
    forest_plot(
        output / "forest-switch",
        [row for row in switch_rows if row["comparison"] != "RS - A"],
        "Certified gate: response switching comparisons",
    )
    print(output / "analysis.json")


def parse_set(values, cast=str):
    return None if not values else {cast(value) for value in values.split(",")}


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reproduce_parser = commands.add_parser("reproduce")
    reproduce_parser.add_argument("--output", type=Path, required=True)
    reproduce_parser.add_argument("--historical", type=Path, required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--seeds")
    train_parser.add_argument("--methods")
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reproduce":
        reproduce(args.output.resolve(), args.historical.resolve())
    elif args.command == "freeze":
        freeze_design(args.output.resolve())
    elif args.command == "train":
        train_design(
            args.output.resolve(),
            selected_seeds=parse_set(args.seeds, int),
            selected_methods=parse_set(args.methods),
        )
    else:
        analyze(args.output.resolve())


if __name__ == "__main__":
    main()
