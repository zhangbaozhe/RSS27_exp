"""Development calibration and paired policy-gradient probes for gate stage 0.

``calibrate`` uses seed 1000 only and follows the G4 rule of
``agent_docs/unicycle_gate_testbed_1018.md`` (3.6). ``gate`` uses seed 2000 and
identical scene queries and policy parameters across profiles, and reports
every state with its stationarity (section 4). It never implements a perturbed
backward (stage 1).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cpu_gate_training as T
import numpy as np
import torch
from torch.func import functional_call

from pivopt import BackwardStatus, ForwardStatus, OptimizerStatus
from pivopt import problem as p

PROFILES = ("hinge", "relaxed-barrier")
RHO_FACTORS = (0.1, 0.3, 1.0, 3.0)
DESCENT_STEPS = (0.01, 0.03, 0.1)
DIRECTIONAL_STEPS = (1e-5, 1e-4, 1e-3, 1e-2)
INSTANCE_SEEDS = (1000, 1001, 1002)
INSTANCE_LEARNING_RATES = (3e-3, 1e-2, 3e-2)
INSTANCE_CAPSULE_RADII = (0.08, 0.04)
INSTANCE_A_STEPS = 2000


def action_groups(task: T.TaskConfig) -> dict[str, slice]:
    return {
        "full": slice(0, task.action_dim),
        "geometry": slice(0, task.geometry_dim),
        "caps": task.caps,
        "time": slice(task.time, task.time + 1),
    }


ACTION_GROUPS = action_groups(T.GATE2)
# Violation tolerance of 3.6: 0.02 m for the corridor, 5% of each physical cap.
VIOLATION_TOLERANCE = {
    "corridor": 0.02,
    "speed_cap": 0.05 * T.PHYSICAL_CAPS[0],
    "centripetal_cap": 0.05 * T.PHYSICAL_CAPS[1],
    "forward_acceleration_cap": 0.05 * T.PHYSICAL_CAPS[2],
    "wheel_speeds": 0.05 * T.WHEEL_LIMIT,
}
# 10% of each constraint's own scale; the corridor's is below 3 delta.
SCALE_ACTIVATION = {
    "corridor": 0.0,
    "speed_cap": 0.12,
    "centripetal_cap": 0.15,
    "forward_acceleration_cap": 0.2,
    "wheel_speeds": 0.12,
}
MINIMUM_CONVERGED_FRACTION = 0.8
# Q28 reference total time, as a multiple of the D_task / 1.2 lower bound.
REFERENCE_TIME_FACTOR = 2.0
REFERENCE_CAP_FACTOR = 1.0
ATTRIBUTION_INTERVALS = T.DENSE_INTERVALS


@dataclass(frozen=True)
class ReferenceCondition:
    cap_factors: tuple[float, float, float]
    time_factor: float


S1_REFERENCES = {
    "corridor": ReferenceCondition((1.0, 1.0, 1.0), 1.5),
    "speed_cap": ReferenceCondition((0.625, 1.0, 1.0), 2.0),
    "centripetal_cap": ReferenceCondition((1.0, 0.625, 1.0), 1.5),
    "forward_acceleration_cap": ReferenceCondition((1.0, 1.0, 0.625), 1.5),
    "wheel_speeds": ReferenceCondition((1.0, 1.0, 1.0), 1.5),
}


def load_policy(
    checkpoint: str | Path, task: T.TaskConfig | None = None
) -> tuple[T.GatePolicy, dict]:
    payload = torch.load(checkpoint, weights_only=True)
    checkpoint_task = payload.get(
        "task", payload.get("settings", {}).get("task", "gate2")
    )
    if task is None:
        task = T.TASKS[checkpoint_task]
    if checkpoint_task != task.name:
        raise ValueError("checkpoint task does not match requested task")
    policy = T.GatePolicy(task).double()
    policy.load_state_dict(payload["policy"])
    return policy, payload


def two_figures(value: float) -> float:
    return float(f"{value:.2g}")


def profile_for_width(
    name: str,
    width: float,
    rho: float,
    *,
    corridor_activation: float | None = None,
) -> T.Calibration:
    """kappa=3 rule: delta=w/4.5, mu=rho*delta, activation>=3 delta."""
    delta = two_figures(width / 4.5)
    scale_activation = (
        corridor_activation
        if name == "corridor" and corridor_activation is not None
        else SCALE_ACTIVATION[name]
    )
    activation = max(scale_activation, two_figures(3 * delta))
    return T.Calibration(float(rho), width, two_figures(rho * delta), delta, activation)


def calibrations_for(widths, rhos, *, corridor_activation=None) -> tuple:
    return tuple(
        profile_for_width(
            name,
            widths[name],
            rhos[name],
            corridor_activation=corridor_activation,
        )
        for name in T.CONSTRAINT_NAMES
    )


def converged(result) -> torch.Tensor:
    return (result.status == int(ForwardStatus.RESULT_AVAILABLE)) & (
        result.diagnostics.optimizer_status == int(OptimizerStatus.GRADIENT_CONVERGENCE)
    )


def reference_action(
    decoded: T.DecodedAction,
    scenes,
    condition: ReferenceCondition | None = None,
) -> T.DecodedAction:
    """Warm-up geometry with fixed feasible D/T (G4 user decision).

    The untrained D/T heads give infeasible (time, cap) pairs at rest
    boundaries, so stiffness calibration uses physical caps and twice the
    total-time lower bound instead; training itself is unchanged.
    """
    if condition is None:
        condition = ReferenceCondition(
            (REFERENCE_CAP_FACTOR,) * len(T.PHYSICAL_CAPS), REFERENCE_TIME_FACTOR
        )
    caps = (
        decoded.caps.new_tensor(T.PHYSICAL_CAPS)
        * decoded.caps.new_tensor(condition.cap_factors)
    ).expand_as(decoded.caps)
    lower = scenes.task_distance / T.PHYSICAL_CAPS[0]
    return replace(
        decoded, caps=caps.clone(), total_duration=condition.time_factor * lower
    )


def calibrate_rhos(
    policy,
    scenes,
    grid,
    *,
    output: Path,
    cycles=8,
    task: T.TaskConfig = T.GATE2,
) -> dict:
    """3.6: smallest grid rho meeting the violation tolerance, then convergence.

    Violations use converged 2048-iteration references; convergence uses the
    training budget. On a conflict the corridor width doubles, then every cap
    width doubles, then the budget rises to 512. Every attempt is recorded.
    """
    grid = tuple(sorted(set(float(value) for value in grid)))
    with torch.no_grad():
        decoded = reference_action(
            T.decode_action(policy(scenes.features()), scenes), scenes
        )
    widths = {
        name: value.width
        for name, value in zip(
            T.CONSTRAINT_NAMES, T.task_calibrations(task), strict=True
        )
    }
    trace, cache = [], {}

    def solve(rhos, iterations):
        setup = T.prepare_problem(
            scenes.batch_size,
            maximum_iterations=iterations,
            calibrations=calibrations_for(
                widths, rhos, corridor_activation=task.corridor_activation
            ),
            task=task,
        )
        with torch.no_grad():
            return setup.solver.solve(setup.bind(scenes, decoded))

    def p95(rhos):
        key = (tuple(widths.values()), tuple(rhos.values()))
        if key in cache:
            return cache[key]
        result = solve(rhos, T.REFERENCE_ITERATIONS)
        valid = result.status == int(ForwardStatus.RESULT_AVAILABLE)
        violations = T.constraint_violations(result, decoded)
        cache[key] = (
            valid,
            {
                name: float(
                    torch.quantile(
                        violation[valid].flatten(1).amax(-1).clamp_min(0), 0.95
                    )
                )
                if bool(valid.any())
                else math.inf
                for name, violation in violations.items()
            },
        )
        return cache[key]

    def sweep():
        rhos = dict.fromkeys(T.CONSTRAINT_NAMES, grid[0])
        for cycle in range(cycles):
            previous = dict(rhos)
            for name in T.CONSTRAINT_NAMES:
                for rho in grid:
                    candidate = {**rhos, name: rho}
                    valid, values = p95(candidate)
                    passed = (
                        bool(valid.all())
                        and values[name] <= (VIOLATION_TOLERANCE[name])
                    )
                    trace.append(
                        {
                            "cycle": cycle,
                            "constraint": name,
                            "widths": dict(widths),
                            "rhos": candidate,
                            "p95": values,
                            "passed": passed,
                        }
                    )
                    T.save_json(output / "rho_progress.json", {"trace": trace})
                    if passed:
                        rhos[name] = rho
                        break
                else:
                    return None
            if rhos == previous:
                return rhos
        return None

    attempts = []
    budget = T.MAXIMUM_ITERATIONS
    plan = ("initial", "corridor_width", "cap_widths", "budget_512")
    for step in plan:
        if step == "corridor_width":
            widths["corridor"] *= 2
        elif step == "cap_widths":
            for name in T.CONSTRAINT_NAMES[1:]:
                widths[name] *= 2
        elif step == "budget_512":
            budget = 512
        rhos = sweep()
        if rhos is None:
            attempts.append({"step": step, "widths": dict(widths), "outcome": "no_rho"})
            continue
        result = solve(rhos, budget)
        fraction = float(converged(result).double().mean())
        attempts.append(
            {
                "step": step,
                "widths": dict(widths),
                "rhos": rhos,
                "budget": budget,
                "converged_fraction": fraction,
                "stationarity": result.diagnostics.stationarity.tolist(),
                "optimizer_status": result.diagnostics.optimizer_status.tolist(),
            }
        )
        if fraction >= MINIMUM_CONVERGED_FRACTION:
            report = {
                "outcome": "calibrated",
                "grid": grid,
                "widths": widths,
                "rhos": rhos,
                "training_budget": budget,
                "calibrations": [
                    asdict(value)
                    for value in calibrations_for(
                        widths, rhos, corridor_activation=task.corridor_activation
                    )
                ],
                "violation_tolerance": VIOLATION_TOLERANCE,
                "attempts": attempts,
                "trace": trace,
            }
            T.save_json(output / "rho_calibration.json", report)
            return report
    report = {"outcome": "not_calibrated", "attempts": attempts, "trace": trace}
    T.save_json(output / "rho_calibration.json", report)
    return report


class RobustRhoCalibrator:
    """Select shared profile weights on the task's per-constraint references."""

    def __init__(
        self,
        policy,
        scenes,
        grid,
        *,
        widths,
        corridor_activation,
        task,
        output,
    ):
        self.policy = policy
        self.scenes = scenes
        self.grid = tuple(sorted(set(float(value) for value in grid)))
        self.widths = dict(widths)
        self.corridor_activation = corridor_activation
        self.task = task
        self.output = output
        with torch.no_grad():
            self.decoded = T.decode_action(policy(scenes.features()), scenes)
        self.cache = {}
        self.trace = []

    def calibrations(self, rhos):
        return calibrations_for(
            self.widths,
            rhos,
            corridor_activation=self.corridor_activation,
        )

    def solve(self, mode, rhos, reference, iterations):
        key = (
            mode,
            tuple(rhos[name] for name in T.CONSTRAINT_NAMES),
            reference.cap_factors,
            reference.time_factor,
            iterations,
        )
        if key not in self.cache:
            decoded = reference_action(self.decoded, self.scenes, reference)
            setup = T.prepare_problem(
                self.scenes.batch_size,
                constraints=mode,
                maximum_iterations=iterations,
                calibrations=self.calibrations(rhos),
                task=self.task,
            )
            with torch.no_grad():
                result = setup.solver.solve(setup.bind(self.scenes, decoded))
            self.cache[key] = (decoded, result)
        return self.cache[key]

    def measure(self, name, rhos):
        reference = S1_REFERENCES[name]
        profiles = {}
        for mode in PROFILES:
            decoded, result = self.solve(mode, rhos, reference, T.REFERENCE_ITERATIONS)
            valid = result.status == int(ForwardStatus.RESULT_AVAILABLE)
            violation = T.constraint_violations(result, decoded)[name]
            lane_maximum = violation.flatten(1).amax(-1).clamp_min(0)
            profiles[mode] = {
                "valid_fraction": float(valid.double().mean()),
                "p95_max_violation": float(torch.quantile(lane_maximum[valid], 0.95))
                if bool(valid.any())
                else math.inf,
                "violation_fraction": float((violation[valid] > 0).double().mean())
                if bool(valid.any())
                else 1.0,
            }
        return profiles

    def run(self, *, cycles=4):
        rhos = dict.fromkeys(T.CONSTRAINT_NAMES, self.grid[0])
        inactive = set()
        for cycle in range(cycles):
            previous = dict(rhos)
            for name in T.CONSTRAINT_NAMES:
                if name in (
                    "centripetal_cap",
                    "forward_acceleration_cap",
                ):
                    weak = self.measure(name, {**rhos, name: self.grid[0]})
                    if weak["hinge"]["violation_fraction"] == 0:
                        rhos[name] = rhos["speed_cap"]
                        inactive.add(name)
                        self.trace.append(
                            {
                                "cycle": cycle,
                                "constraint": name,
                                "outcome": "inactive_use_speed_rho",
                                "rhos": dict(rhos),
                                "profiles": weak,
                            }
                        )
                        continue
                for rho in self.grid:
                    candidate = {**rhos, name: rho}
                    profiles = self.measure(name, candidate)
                    passed = all(
                        values["valid_fraction"] == 1.0
                        and values["p95_max_violation"] <= VIOLATION_TOLERANCE[name]
                        for values in profiles.values()
                    )
                    self.trace.append(
                        {
                            "cycle": cycle,
                            "constraint": name,
                            "rho": rho,
                            "rhos": candidate,
                            "profiles": profiles,
                            "passed": passed,
                        }
                    )
                    T.save_json(
                        self.output / "rho_progress.json",
                        {
                            "widths": self.widths,
                            "corridor_activation": self.corridor_activation,
                            "trace": self.trace,
                        },
                    )
                    if passed:
                        rhos[name] = rho
                        break
                else:
                    return {
                        "outcome": "no_rho",
                        "widths": self.widths,
                        "corridor_activation": self.corridor_activation,
                        "trace": self.trace,
                    }
            if rhos == previous:
                return {
                    "outcome": "selected",
                    "widths": self.widths,
                    "rhos": rhos,
                    "inactive_constraints": sorted(inactive),
                    "corridor_activation": self.corridor_activation,
                    "calibrations": [
                        asdict(value) for value in self.calibrations(rhos)
                    ],
                    "trace": self.trace,
                }
        return {
            "outcome": "no_fixed_point",
            "widths": self.widths,
            "rhos": rhos,
            "corridor_activation": self.corridor_activation,
            "trace": self.trace,
        }


def robust_checkpoint(policy, scenes, calibrations, *, budget, task) -> dict:
    reference = S1_REFERENCES["corridor"]
    with torch.no_grad():
        decoded = reference_action(
            T.decode_action(policy(scenes.features()), scenes), scenes, reference
        )
    profiles = {}
    for mode in PROFILES:
        reference_setup = T.prepare_problem(
            scenes.batch_size,
            constraints=mode,
            maximum_iterations=T.REFERENCE_ITERATIONS,
            calibrations=calibrations,
            task=task,
        )
        budget_setup = T.prepare_problem(
            scenes.batch_size,
            constraints=mode,
            maximum_iterations=budget,
            calibrations=calibrations,
            task=task,
        )
        with torch.no_grad():
            reference_result = reference_setup.solver.solve(
                reference_setup.bind(scenes, decoded)
            )
            budget_result = budget_setup.solver.solve(
                budget_setup.bind(scenes, decoded)
            )
        reference_valid = reference_result.status == int(ForwardStatus.RESULT_AVAILABLE)
        violation = T.constraint_violations(reference_result, decoded)["corridor"]
        profiles[mode] = {
            "reference_valid_fraction": float(reference_valid.double().mean()),
            "dense_violation_fraction": float(
                (violation[reference_valid].amax(-1) > 0).double().mean()
            )
            if bool(reference_valid.any())
            else 1.0,
            "training_converged_fraction": float(
                converged(budget_result).double().mean()
            ),
            "reference_optimizer_status": reference_result.diagnostics.optimizer_status.tolist(),
            "training_optimizer_status": budget_result.diagnostics.optimizer_status.tolist(),
            "training_stationarity": budget_result.diagnostics.stationarity.tolist(),
        }

    barrier_pinch = pinch_acceptance(
        policy,
        scenes,
        calibrations,
        budget=budget,
        task=task,
        constraints="relaxed-barrier",
        reference=reference,
    )
    active_fraction = barrier_pinch["corridor_active_sample_fraction"]
    targets = {
        "activation_fraction": 0.20 <= active_fraction <= 0.40,
        "reference_violation_fraction": all(
            values["dense_violation_fraction"] < 0.05 for values in profiles.values()
        ),
        "training_convergence": all(
            values["training_converged_fraction"] >= 0.90
            for values in profiles.values()
        ),
        "pinch": barrier_pinch["passed"],
    }
    return {
        "passed": all(targets.values()),
        "targets": targets,
        "profiles": profiles,
        "barrier_activation_fraction": active_fraction,
        "pinch": barrier_pinch,
        "budget": budget,
    }


def calibrate_robust_rhos(policy, scenes, grid, *, output, task) -> dict:
    base = T.task_calibrations(task)
    widths = {
        name: value.width for name, value in zip(T.CONSTRAINT_NAMES, base, strict=True)
    }
    budget = T.MAXIMUM_ITERATIONS
    attempts = []
    fallback = ("initial", "corridor_width", "cap_widths", "budget_512")
    for step in fallback:
        if step == "corridor_width":
            widths["corridor"] *= 2
        elif step == "cap_widths":
            for name in T.CONSTRAINT_NAMES[1:]:
                widths[name] *= 2
        elif step == "budget_512":
            budget = 512

        activation = task.corridor_activation
        tried = set()
        for _ in range(5):
            activation = float(np.clip(round(activation, 2), 0.06, 0.10))
            if activation in tried:
                break
            tried.add(activation)
            attempt_output = output / f"{step}_activation_{activation:.2f}"
            attempt_output.mkdir(parents=True, exist_ok=True)
            selector = RobustRhoCalibrator(
                policy,
                scenes,
                grid,
                widths=widths,
                corridor_activation=activation,
                task=task,
                output=attempt_output,
            )
            selected = selector.run()
            attempt = {
                "step": step,
                "budget": budget,
                "widths": dict(widths),
                "corridor_activation": activation,
                "rho": selected,
            }
            if selected["outcome"] != "selected":
                attempts.append(attempt)
                break
            calibrations = tuple(
                T.Calibration(**value) for value in selected["calibrations"]
            )
            checkpoint = robust_checkpoint(
                policy, scenes, calibrations, budget=budget, task=task
            )
            attempt["checkpoint"] = checkpoint
            attempts.append(attempt)
            T.save_json(
                output / "robust_calibration_progress.json", {"attempts": attempts}
            )
            if checkpoint["passed"]:
                report = {
                    "outcome": "calibrated",
                    "grid": list(selector.grid),
                    "widths": dict(widths),
                    "rhos": selected["rhos"],
                    "inactive_constraints": selected["inactive_constraints"],
                    "corridor_activation": activation,
                    "training_budget": budget,
                    "calibrations": selected["calibrations"],
                    "checkpoint": checkpoint,
                    "attempts": attempts,
                    "references": {
                        name: asdict(value) for name, value in S1_REFERENCES.items()
                    },
                }
                T.save_json(output / "rho_calibration.json", report)
                return report
            active = checkpoint["barrier_activation_fraction"]
            if active < 0.20:
                activation = min(0.10, activation * 0.25 / max(active, 1e-6))
                continue
            if active > 0.30:
                activation = max(0.06, activation * 0.25 / active)
                continue
            break

        last = attempts[-1]
        checkpoint = last.get("checkpoint")
        if checkpoint and checkpoint["targets"]["training_convergence"]:
            break

    report = {"outcome": "not_calibrated", "attempts": attempts}
    T.save_json(output / "rho_calibration.json", report)
    return report


def pinch_acceptance(
    policy,
    scenes,
    calibrations,
    *,
    budget,
    task: T.TaskConfig = T.GATE2,
    constraints: str = "hinge",
    reference: ReferenceCondition | None = None,
) -> dict:
    """3.2: gates are pinch points, the corridor is not a tube everywhere."""
    setup = T.prepare_problem(
        scenes.batch_size,
        constraints=constraints,
        maximum_iterations=budget,
        calibrations=calibrations,
        task=task,
    )
    summary, _ = T.evaluate(
        policy,
        setup,
        scenes,
        weights=T.LOSS_WEIGHTS,
        corridor_activation=calibrations[0].activation,
        transform=lambda decoded: reference_action(decoded, scenes, reference),
    )
    return {
        "gate_corridor_active_scene_fraction": summary[
            "gate_corridor_active_scene_fraction"
        ],
        "corridor_active_sample_fraction": summary["corridor_active_sample_fraction"],
        "gate_corridor_narrow_half_extent_median": summary[
            "gate_corridor_narrow_half_extent_median"
        ],
        "narrow_half_extent_median": summary["narrow_half_extent_median"],
        "passed": summary["gate_corridor_active_scene_fraction"] >= 0.5
        and summary["corridor_active_sample_fraction"] <= 0.3,
        "budget": budget,
        "reference": {
            "geometry": "warm-up",
            "cap_factors": (
                reference.cap_factors
                if reference is not None
                else (REFERENCE_CAP_FACTOR,) * len(T.PHYSICAL_CAPS)
            ),
            "time_factor": (
                reference.time_factor
                if reference is not None
                else REFERENCE_TIME_FACTOR
            ),
        },
    }


def calibrate_weights(
    policy,
    scenes,
    calibrations,
    *,
    iterations,
    task: T.TaskConfig = T.GATE2,
) -> dict:
    midpoint = replace(scenes, smoothness=torch.full_like(scenes.smoothness, 0.5))
    setup = T.prepare_problem(
        scenes.batch_size,
        maximum_iterations=iterations,
        calibrations=calibrations,
        task=task,
    )
    with torch.no_grad():
        decoded = T.decode_action(policy(midpoint.features()), midpoint)
        result = setup.solver.solve(setup.bind(midpoint, decoded))
        losses = T.trajectory_losses(result, decoded, midpoint, weights=T.LossWeights())
        valid = T.valid_lanes(result, losses)
        if not bool(valid.all()):
            raise RuntimeError(
                "outer-weight calibration requires all development lanes available"
            )
        median = {
            name: float(torch.quantile(losses[name], 0.5))
            for name in (
                "pace",
                "acceleration",
                "jerk",
                "clearance_loss",
                "gate",
                "geometry",
            )
        }
        if min(median["acceleration"], median["jerk"]) <= 0:
            raise RuntimeError(
                f"cannot balance an identically zero development contribution: {median}"
            )
        # A successful safety loss may be exactly zero. Its multiplier must
        # not disappear or diverge merely because the current checkpoint is
        # safe; balance only performance and keep the safety units fixed.
        target = max(median["pace"], 0.01)
        a = target / (2 * math.sqrt(10) * median["acceleration"])
        j = target / (2 * math.sqrt(10) * median["jerk"])
        correction = target / float(
            torch.quantile(
                math.sqrt(10) * (a * losses["acceleration"] + j * losses["jerk"]), 0.5
            )
        )
        weights = T.LossWeights(
            pace=1.0,
            acceleration_low=a * correction,
            jerk_low=j * correction,
            clearance=1.0,
            geometry=1.0,
            gate=1.0,
        )
        weighted = T.trajectory_losses(result, decoded, midpoint, weights=weights)
        contributions = {
            name: float(torch.quantile(weighted[name], 0.5))
            for name in (
                "weighted_pace",
                "weighted_smoothness",
                "weighted_clearance",
                "weighted_gate",
                "weighted_geometry",
            )
        }
    return {
        "weights": asdict(weights),
        "raw_medians": median,
        "weighted_medians": contributions,
        "basis": f"hinge, {task.name} warm-up checkpoint, c_smooth=0.5, seed=1000",
        "performance_target": target,
        "safety_rule": "gate and geometry multipliers stay at one; clearance follows the task contract and may be evaluation-only",
    }


def learning_rate_trial(
    policy_checkpoint,
    *,
    learning_rate,
    mode,
    weights,
    calibrations,
    steps,
    scene_count,
    budget,
    task: T.TaskConfig = T.GATE2,
) -> dict:
    policy, _ = load_policy(policy_checkpoint, task)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(1000)
    setup = T.prepare_problem(
        scene_count,
        constraints=mode,
        calibrations=calibrations,
        maximum_iterations=budget,
        task=task,
    )
    outcome = "completed"
    completed = 0
    for step in range(steps):
        scenes = T.sample_scenes(scene_count, generator=generator, task=task)
        decoded = T.decode_action(policy(scenes.features()), scenes)
        result = setup.solver.solve(setup.bind(scenes, decoded))
        losses = T.trajectory_losses(result, decoded, scenes, weights=weights)
        if not bool(T.valid_lanes(result, losses).any()):
            outcome = "all_lanes_failed"
            break
        loss, _ = T.batch_loss(result, losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in policy.parameters()
            if parameter.grad is not None
        ]
        norm = torch.nn.utils.get_total_norm(gradients)
        if not bool(torch.isfinite(norm)):
            outcome = "nonfinite_parameter_gradient"
            break
        torch.nn.utils.clip_grads_with_norm_(list(policy.parameters()), 10, norm)
        optimizer.step()
        completed = step + 1
    development = T.sample_scenes(
        scene_count,
        generator=torch.Generator().manual_seed(1000),
        task=task,
    )
    with torch.no_grad():
        decoded = T.decode_action(policy(development.features()), development)
        result = setup.solver.solve(setup.bind(development, decoded))
        losses = T.trajectory_losses(result, decoded, development, weights=weights)
        if bool(T.valid_lanes(result, losses).any()):
            score, valid = T.batch_loss(result, losses)
            value, failures = float(score), float((~valid).double().mean())
        else:
            value, failures = None, 1.0
        crossings = T.gate_traversal(result, development)
    return {
        "profile": mode,
        "learning_rate": learning_rate,
        "steps": completed,
        "outcome": outcome,
        "score": value,
        "failure_fraction": failures,
        "gate_success_fraction": float(np.mean(crossings["gate_success"])),
    }


def run_calibrate(
    *,
    warmup_checkpoint: Path,
    output: Path,
    scene_count=32,
    iterations=T.REFERENCE_ITERATIONS,
    rho_grid=None,
    learning_rate_steps=64,
    task: T.TaskConfig = T.GATE2,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    policy, payload = load_policy(warmup_checkpoint, task)
    scenes = T.sample_scenes(
        scene_count,
        generator=torch.Generator().manual_seed(1000),
        task=task,
    )
    if rho_grid is None:
        rho_grid = [
            mantissa * 10.0**exponent
            for exponent in range(-1, 9)
            for mantissa in (1, 3)
        ]
    rho = (
        calibrate_rhos(policy, scenes, rho_grid, output=output, task=task)
        if task is T.GATE2
        else calibrate_robust_rhos(policy, scenes, rho_grid, output=output, task=task)
    )
    if rho["outcome"] != "calibrated":
        return rho
    calibrations = tuple(T.Calibration(**value) for value in rho["calibrations"])
    budget = rho["training_budget"]
    pinch = (
        rho["checkpoint"]["pinch"]
        if task is T.SGATE3
        else pinch_acceptance(policy, scenes, calibrations, budget=budget, task=task)
    )
    balanced = calibrate_weights(
        policy, scenes, calibrations, iterations=iterations, task=task
    )
    weights = T.LossWeights(**balanced["weights"])
    trials = [
        learning_rate_trial(
            warmup_checkpoint,
            learning_rate=lr,
            mode=mode,
            weights=weights,
            calibrations=calibrations,
            steps=learning_rate_steps,
            scene_count=scene_count,
            budget=budget,
            task=task,
        )
        for lr in (5e-5, 2e-4)
        for mode in PROFILES
    ]
    scores = {}
    for lr in (5e-5, 2e-4):
        paired = [row for row in trials if row["learning_rate"] == lr]
        if all(
            row["outcome"] == "completed" and row["score"] is not None for row in paired
        ):
            scores[lr] = (
                -min(row["gate_success_fraction"] for row in paired),
                max(row["failure_fraction"] for row in paired),
                float(np.mean([row["score"] for row in paired])),
            )
    if not scores:
        raise RuntimeError(
            "both learning rates failed; inspect development diagnostics"
        )
    mechanisms = (
        plot_mechanisms(calibrations, output / "figures") if task is T.SGATE3 else None
    )
    report = {
        "outcome": "calibrated",
        "task": task.name,
        "seed": 1000,
        "scenes": scene_count,
        "reference_iterations": iterations,
        "warmup_checkpoint": str(warmup_checkpoint),
        "warmup_steps": payload["warmup_steps"],
        "rho": rho,
        "calibrations": [asdict(value) for value in calibrations],
        "training_budget": budget,
        "pinch": pinch,
        **balanced,
        "learning_rate_trials": trials,
        "learning_rate_trial_steps": learning_rate_steps,
        "learning_rate": min(scores, key=scores.get),
        "learning_rate_selection": "maximize the worse profile's development gate success, then failure fraction, then mean outer loss",
        "mechanism_figures": None if mechanisms is None else mechanisms["figures"],
    }
    T.save_json(output / "calibration.json", report)
    return report


def flat_parameters(policy):
    return tuple(policy.parameters())


def flattened_vjp(outputs, parameters, seed=None):
    gradients = torch.autograd.grad(
        outputs, parameters, grad_outputs=seed, retain_graph=True, allow_unused=True
    )
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    ).detach()


def profile_violation_partial(violation, calibration, mode):
    """Independent scalar-profile partial for node anticipation diagnostics.

    This is the objective's local partial, not a per-node decomposition of
    the outer implicit VJP. Native input cotangents are reported separately.
    """
    if mode == "hinge":
        ratio = (violation / calibration.width).clamp(0, 1)
        return calibration.hinge_weight * ratio.square() * (3 - 2 * ratio)
    slack = -violation
    delta = calibration.delta
    safe = slack.clamp_min(delta)
    if calibration.activation is None:
        partial = torch.where(slack > delta, 1 / safe, (2 - slack / delta) / delta)
    else:
        activation = calibration.activation
        t = (activation - safe) / activation
        logarithm = torch.log(safe / activation)
        unrelaxed = -2 * t * logarithm / activation + t.square() / safe
        td = 1 - delta / activation
        ld = math.log(delta / activation)
        first = 2 * td * ld / activation - td**2 / delta
        second = (
            -2 * ld / activation**2 + 4 * td / (activation * delta) + td**2 / delta**2
        )
        relaxed = -first + second * (delta - slack)
        partial = torch.where(
            slack >= activation, 0, torch.where(slack > delta, unrelaxed, relaxed)
        )
    return calibration.barrier_weight * partial


def profile_penalty(slack, calibration, mode):
    """Scalar profile primal matching the native hinge/barrier definitions."""
    violation = -slack
    if mode == "hinge":
        width = calibration.width
        ratio = (violation / width).clamp(0, 1)
        middle = (width - 0.5 * violation) * ratio.pow(3)
        basis = torch.where(
            violation <= 0,
            torch.zeros_like(violation),
            torch.where(violation >= width, violation - 0.5 * width, middle),
        )
        return calibration.hinge_weight * basis

    delta = calibration.delta
    activation = calibration.activation
    if activation is None:
        boundary_value, first, second = -math.log(delta), -1.0, 1.0
        unrelaxed = -torch.log(slack)
    else:
        ratio = delta / activation
        t_delta = (activation - delta) / activation
        logarithm = math.log(delta / activation)
        boundary_value = -(t_delta**2) * logarithm
        first = 2 * t_delta * ratio * logarithm - t_delta**2
        second = -2 * logarithm * ratio**2 + 4 * t_delta * ratio + t_delta**2
        safe = slack.clamp_min(torch.finfo(slack.dtype).tiny)
        t = (activation - safe) / activation
        unrelaxed = -t.square() * torch.log(safe / activation)
    q = (slack - delta) / delta
    relaxed = boundary_value + first * q + 0.5 * second * q.square()
    basis = torch.where(slack > delta, unrelaxed, relaxed)
    if activation is not None:
        basis = torch.where(slack >= activation, torch.zeros_like(slack), basis)
    return calibration.barrier_weight * basis


def native_profile_samples(calibration, mode, slacks) -> dict:
    """Read a fixed endpoint penalty through the public native CPU solver."""
    from pivopt import numpy as cpu

    slacks = np.asarray(slacks, dtype=np.float64)
    step = 1e-6
    query = np.concatenate((slacks, slacks + step, slacks - step))
    batch = len(query)
    trajectory = p.MincoSpec(
        p.Space(position=2),
        derivative_order=3,
        pieces=2,
        head_state=p.Input("head"),
        tail_state=p.Input("tail"),
        time=p.FixedTotalTime(total_duration=1.0, minimum_piece_duration=0.1),
        waypoints=p.DirectWaypoints(),
    )
    spec = p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=1),
        objective=(
            p.Energy("energy"),
            T.term(
                p.AtEnd,
                "face",
                lambda profile: p.Inside(
                    trajectory.position, p.Input("rows"), profile=profile
                ),
                calibration,
                mode,
            ),
        ),
        forward=p.BFGS(
            initial_guess=p.Input("initial"),
            settings=p.OptimizerSettings(
                maximum_iterations=8, gradient_tolerance=1e-9, past=0
            ),
        ),
        backward=p.ForwardOnly(),
    )
    plan = p.build_plan(
        spec,
        p.PlanShape(
            batch_size=batch,
            input_shapes=(p.InputShape(p.InputRef("rows"), (batch, 2, 1, 3)),),
        ),
    )
    rows = np.zeros((batch, 2, 1, 3))
    rows[..., 0] = 1
    rows[..., 2] = -query[:, None, None]
    inputs = plan.inputs(
        head=np.zeros((batch, 2, 3)),
        tail=np.zeros((batch, 2, 3)),
        rows=rows,
        initial=plan.initial_guess(
            waypoints=np.zeros((batch, 1, 2)),
            durations=np.full((batch, 2), 0.5),
        ),
    )
    result = cpu.build_solver(plan, dtype=np.float64).solve(inputs)
    if not np.all(result.status == int(ForwardStatus.RESULT_AVAILABLE)):
        raise RuntimeError("native mechanism leaf failed")
    count = len(slacks)
    return {
        "slack": slacks.tolist(),
        "penalty": result.cost[:count].tolist(),
        "absolute_slack_derivative": (
            np.abs(result.cost[count : 2 * count] - result.cost[2 * count :])
            / (2 * step)
        ).tolist(),
    }


def toy_sensitivity(calibration, mode, boundaries) -> dict:
    """Implicit sensitivity of a planar toy whose only motion is along x."""
    from pivopt.torch import build_solver

    boundaries = torch.as_tensor(boundaries, dtype=torch.float64).requires_grad_()
    batch = len(boundaries)
    trajectory = p.MincoSpec(
        p.Space(position=2),
        derivative_order=3,
        pieces=2,
        head_state=p.Input("head"),
        tail_state=p.Input("tail"),
        time=p.FixedTotalTime(total_duration=2.0, minimum_piece_duration=0.1),
        waypoints=p.DirectWaypoints(),
    )
    spec = p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=16),
        objective=(
            p.Energy("energy", weights={"position": 0.001}),
            p.Integral(
                "target",
                p.Quadratic(trajectory.position, target=(0.6, 0.0)),
                weight=1.0,
            ),
            T.term(
                p.Integral,
                "face",
                lambda profile: p.Inside(
                    trajectory.position,
                    p.Input("rows", differentiate=True),
                    profile=profile,
                ),
                calibration,
                mode,
            ),
        ),
        forward=p.LBFGS(
            initial_guess=p.Input("initial"),
            memory_size=8,
            settings=p.OptimizerSettings(
                maximum_iterations=512, gradient_tolerance=1e-7, past=0
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
            batch_size=batch,
            input_shapes=(p.InputShape(p.InputRef("rows"), (batch, 2, 1, 3)),),
        ),
    )
    rows = torch.zeros((batch, 2, 1, 3), dtype=torch.float64)
    rows[..., 0] = 1
    rows[..., 2] = -boundaries[:, None, None]
    boundary_state = torch.zeros((batch, 2, 3), dtype=torch.float64)
    initial = torch.from_numpy(
        plan.initial_guess(
            waypoints=np.tile((0.2, 0.0), (batch, 1, 1)),
            durations=np.full((batch, 2), 1.0),
        )
    )
    result = build_solver(plan, dtype=torch.float64, backward=True).solve(
        plan.inputs(
            head=boundary_state,
            tail=boundary_state.clone(),
            rows=rows,
            initial=initial,
        )
    )
    if not bool((result.status == int(ForwardStatus.RESULT_AVAILABLE)).all()):
        raise RuntimeError("implicit-sensitivity toy failed")
    coefficients = result.coefficients[:, 0, :, 0]
    duration = result.durations[:, 0]
    waypoint = sum(
        coefficients[:, degree] * duration**degree
        for degree in range(coefficients.shape[1])
    )
    sensitivity = torch.autograd.grad(waypoint.sum(), boundaries)[0]
    return {
        "boundary": boundaries.detach().tolist(),
        "waypoint": waypoint.detach().tolist(),
        "nominal_slack": (boundaries - waypoint).detach().tolist(),
        "absolute_dz_db": sensitivity.detach().abs().tolist(),
        "stationarity": result.diagnostics.stationarity.tolist(),
        "optimizer_status": result.diagnostics.optimizer_status.tolist(),
    }


def plot_mechanisms(calibrations, output: Path) -> dict:
    """Render the predeclared scalar-profile and implicit-sensitivity figures."""
    import os

    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibration = calibrations[0]
    slack = torch.linspace(-0.02, 0.12, 701, dtype=torch.float64).requires_grad_()
    selected = np.asarray((-0.02, 0.0, 0.01, 0.03, 0.05, 0.08, 0.12))
    colors = {"hinge": "#2563eb", "relaxed-barrier": "#d97706"}
    curves, native = {}, {}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for mode in PROFILES:
        penalty = profile_penalty(slack, calibration, mode)
        derivative = torch.autograd.grad(penalty.sum(), slack, retain_graph=True)[
            0
        ].abs()
        curves[mode] = {
            "slack": slack.detach().tolist(),
            "penalty": penalty.detach().tolist(),
            "absolute_slack_derivative": derivative.detach().tolist(),
        }
        native[mode] = native_profile_samples(calibration, mode, selected)
        expected_slack = torch.as_tensor(selected, dtype=torch.float64).requires_grad_()
        expected_penalty = profile_penalty(expected_slack, calibration, mode)
        expected_derivative = torch.autograd.grad(
            expected_penalty.sum(), expected_slack
        )[0].abs()
        np.testing.assert_allclose(
            native[mode]["penalty"], expected_penalty.detach(), rtol=2e-10, atol=2e-10
        )
        np.testing.assert_allclose(
            native[mode]["absolute_slack_derivative"],
            expected_derivative.detach(),
            rtol=2e-5,
            atol=2e-5,
        )
        axes[0].plot(slack.detach(), penalty.detach(), color=colors[mode], label=mode)
        axes[1].plot(
            slack.detach(), derivative.detach(), color=colors[mode], label=mode
        )
        axes[0].scatter(
            selected, native[mode]["penalty"], color=colors[mode], marker="x", zorder=4
        )
        axes[1].scatter(
            selected,
            native[mode]["absolute_slack_derivative"],
            color=colors[mode],
            marker="x",
            zorder=4,
        )
    axes[0].set(xlabel="Slack (m)", ylabel="Penalty", title="Profile value")
    axes[1].set(
        xlabel="Slack (m)",
        ylabel="|d penalty / d slack|",
        title="Profile gradient magnitude",
    )
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[1].set_yscale("symlog", linthresh=1.0)
    for axis in axes:
        axis.axvline(0, color="#64748b", linestyle=":")
        axis.axvline(calibration.activation, color="#64748b", linestyle="--")
        axis.grid(alpha=0.2)
        axis.legend()
    profiles_path = output / "mechanism_profiles.png"
    figure.savefig(profiles_path, dpi=180)
    plt.close(figure)

    boundaries = np.linspace(0.10, 0.60, 51)
    toy = {mode: toy_sensitivity(calibration, mode, boundaries) for mode in PROFILES}
    figure, axis = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    for mode in PROFILES:
        axis.plot(
            toy[mode]["nominal_slack"],
            toy[mode]["absolute_dz_db"],
            color=colors[mode],
            label=mode,
        )
    axis.axvline(0, color="#64748b", linestyle=":")
    axis.axvline(calibration.activation, color="#64748b", linestyle="--")
    axis.set(
        xlabel="Optimized nominal slack (m)",
        ylabel="|dz* / db|",
        title="Native implicit sensitivity in the 1D motion toy",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    toy_path = output / "mechanism_implicit_sensitivity.png"
    figure.savefig(toy_path, dpi=180)
    plt.close(figure)
    report = {
        "calibration": asdict(calibration),
        "profiles": curves,
        "native_crosscheck": native,
        "toy": toy,
        "figures": {
            "profiles": str(profiles_path.resolve()),
            "implicit_sensitivity": str(toy_path.resolve()),
        },
    }
    T.save_json(output / "mechanisms.json", report)
    return report


def policy_paths(
    policy,
    scenes,
    setup,
    *,
    weights=T.LOSS_WEIGHTS,
    train_clearance: bool | None = None,
    geometry_loss: str = "current",
    capsule_radius: float | None = None,
) -> dict:
    """Per-lane network and raw-action VJPs of the unaveraged lane loss.

    Result leaves separate the explicit readout/geometry paths from the
    implicit response. A lane is eligible when its reference is available and
    its Pullback is applied with finite cotangents; convergence is reported
    separately, not required (section 4).
    """
    action = policy(scenes.features())
    decoded = T.decode_action(action, scenes, capsule_radius=capsule_radius)
    with torch.no_grad():
        result, state = setup.solver.solve_with_state(setup.bind(scenes, decoded))
    names = p.ResultSeeds._fields
    leaves = tuple(
        getattr(result, name).detach().requires_grad_(True) for name in names
    )
    output = replace(result, **dict(zip(names, leaves, strict=True)))
    losses = T.trajectory_losses(
        output,
        decoded,
        scenes,
        weights=weights,
        train_clearance=train_clearance,
        geometry_loss=geometry_loss,
    )
    available = T.valid_lanes(output, losses)
    if not bool(available.any()):
        raise RuntimeError("all probe references failed")
    seeds = torch.autograd.grad(
        losses["task"][available].sum(), leaves, retain_graph=True, allow_unused=True
    )
    pulled = setup.solver.pullback(
        state,
        p.ResultSeeds(
            **{
                name: None if seed is None else seed.detach().contiguous()
                for name, seed in zip(names, seeds, strict=True)
            }
        ),
    )
    eligible = available & (pulled.diagnostics.status == int(BackwardStatus.APPLIED))
    sources = {
        "corridor": decoded.corridor,
        "total_duration": decoded.total_duration,
        **{name: decoded.caps[:, index] for index, name in enumerate(T.CAP_NAMES)},
    }
    cotangents = {}
    for name in T.SOLVER_PATHS:
        seed = pulled.parameters[name]
        finite = torch.isfinite(seed.reshape(seed.shape[0], -1)).all(-1)
        eligible &= finite
        cotangents[name] = torch.where(
            finite.reshape(-1, *([1] * (seed.dim() - 1))), seed, torch.zeros_like(seed)
        )
    direct = weights.geometry * losses["geometry"] + weights.pace * losses["pace"]
    lane_scalar = sum(
        (sources[name] * cotangents[name]).reshape(scenes.batch_size, -1).sum(-1)
        for name in T.SOLVER_PATHS
    ) + torch.where(eligible, direct, torch.zeros_like(direct))
    # Lanes are independent, so the gradient of the lane sum is per lane.
    action_gradient = torch.autograd.grad(lane_scalar.sum(), action, retain_graph=True)[
        0
    ].detach()
    parameters = flat_parameters(policy)
    vectors = {name: [] for name in (*T.SOLVER_PATHS, "direct_geometry", "direct_time")}
    for lane in range(scenes.batch_size):
        for name in T.SOLVER_PATHS:
            vectors[name].append(
                flattened_vjp(sources[name][lane], parameters, cotangents[name][lane])
            )
        vectors["direct_geometry"].append(
            flattened_vjp(weights.geometry * losses["geometry"][lane], parameters)
        )
        vectors["direct_time"].append(
            flattened_vjp(weights.pace * losses["pace"][lane], parameters)
        )
    vectors = {name: torch.stack(value) for name, value in vectors.items()}
    vectors["constraints"] = sum(vectors[name] for name in ("corridor", *T.CAP_NAMES))
    vectors["solver"] = sum(vectors[name] for name in T.SOLVER_PATHS)
    vectors["task_total"] = vectors["solver"] + vectors["direct_time"]
    vectors["total"] = vectors["task_total"] + vectors["direct_geometry"]
    return {
        "vectors": vectors,
        "action": action.detach(),
        "action_gradient": action_gradient,
        "eligible": eligible.detach(),
        "converged": converged(result),
        "stationarity": result.diagnostics.stationarity.detach(),
        "result": result,
        "decoded": decoded,
        "losses": {name: value.detach() for name, value in losses.items()},
        "input_cotangents": {
            name: value.detach() for name, value in pulled.parameters.items()
        },
        "backward_status": pulled.diagnostics.status.tolist(),
    }


@torch.no_grad()
def anticipation_metrics(
    result,
    decoded,
    input_cotangents,
    eligible,
    mode,
    calibrations=None,
):
    """Reduce anticipation on the same paired-converged lane set."""
    violations = T.constraint_violations(result, decoded, intervals=16)
    anticipation = {}
    if calibrations is None:
        calibrations = T.CALIBRATIONS
    for name, calibration in zip(T.CONSTRAINT_NAMES, calibrations, strict=True):
        slack = -violations[name].detach()
        near = (slack > 0) & (slack < calibration.activation)
        local_partial = profile_violation_partial(
            violations[name].detach(), calibration, mode
        )
        if name == "corridor":
            # One native offset cotangent per face, accumulated over its nodes.
            near_faces = near.any(2) & (slack > 0).all(2)
            nonzero_offset = input_cotangents[name][..., 2].abs() > 1e-12
            nonzero = input_cotangents[name].norm(dim=-1) > 1e-12
            mask = near_faces & eligible[:, None, None]
            anticipation[name] = {
                "near_feasible_faces": int(mask.sum()),
                "near_feasible_nodes": int(
                    (near & eligible[:, None, None, None]).sum()
                ),
                "faces_with_native_outer_gradient": int((mask & nonzero).sum()),
                "faces_with_native_offset_gradient": int((mask & nonzero_offset).sum()),
                "aggregation": "native row cotangent (normal and offset) accumulated over quadrature nodes",
                "near_nodes_with_objective_profile_partial": int(
                    (
                        near
                        & eligible[:, None, None, None]
                        & (local_partial.abs() > 1e-12)
                    ).sum()
                ),
            }
        elif name in T.CAP_NAMES:
            mask = near & eligible[:, None, None]
            nonzero = input_cotangents[name].abs() > 1e-12
            anticipation[name] = {
                "near_feasible_nodes": int(mask.sum()),
                "nodes_in_lanes_with_native_cap_gradient": int(
                    (mask & nonzero[:, None, None]).sum()
                ),
                "aggregation": "native uniform-cap cotangent accumulated over all nodes; not per-node attribution",
                "near_nodes_with_objective_profile_partial": int(
                    (mask & (local_partial.abs() > 1e-12)).sum()
                ),
            }
    return anticipation


def lane_cosine(left, right):
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return torch.where(
        denominator > 1e-24,
        (left * right).sum(-1) / denominator,
        torch.full_like(denominator, float("nan")),
    )


COMPONENTS = (
    "task",
    "weighted_pace",
    "weighted_smoothness",
    "weighted_clearance",
    "weighted_gate",
    "weighted_geometry",
    "total",
)


def solve_actions(actions, scenes, setup, weights):
    with torch.no_grad():
        decoded = T.decode_action(actions, scenes)
        result = setup.solver.solve(setup.bind(scenes, decoded))
        losses = T.trajectory_losses(result, decoded, scenes, weights=weights)
    return result, losses


def directional_check(policy, scenes, paths, setup, *, weights, steps) -> list:
    """Central differences of the lane loss along its own unit policy gradient.

    The first-order prediction along u = g/|g| is |g|. Each step re-solves all
    lanes from the same initial-guess rule with the reference budget.
    """
    names, values = zip(*policy.named_parameters(), strict=True)
    gradient = paths["vectors"]["total"]
    norm = gradient.norm(dim=-1)
    usable = paths["eligible"] & (norm > 1e-12)
    unit = gradient / norm.clamp_min(1e-300)[:, None]
    features = scenes.features()
    rows = []
    for eta in steps:
        totals, flags = [], []
        for sign in (1.0, -1.0):
            actions = []
            with torch.no_grad():
                for lane in range(scenes.batch_size):
                    shifted, offset = {}, 0
                    for name, value in zip(names, values, strict=True):
                        shifted[name] = value.detach() + sign * eta * unit[lane][
                            offset : offset + value.numel()
                        ].reshape_as(value)
                        offset += value.numel()
                    actions.append(
                        functional_call(policy, shifted, (features[lane : lane + 1],))
                    )
            result, losses = solve_actions(torch.cat(actions), scenes, setup, weights)
            totals.append(losses["total"])
            flags.append(T.valid_lanes(result, losses) & converged(result))
        difference = (totals[0] - totals[1]) / (2 * eta)
        relative = (difference - norm).abs() / norm.clamp_min(1e-300)
        for lane in range(scenes.batch_size):
            rows.append(
                {
                    "lane": lane,
                    "eta": eta,
                    "usable": bool(usable[lane]),
                    "both_converged": bool(flags[0][lane] & flags[1][lane]),
                    "predicted": float(norm[lane]),
                    "difference": float(difference[lane]),
                    "relative_error": float(relative[lane]),
                }
            )
    return rows


def summarize_directional(rows, tolerance=0.1) -> dict:
    """A lane has a stable interval when two adjacent steps both meet tolerance."""
    by_lane = {}
    for row in rows:
        if row["usable"]:
            by_lane.setdefault(row["lane"], []).append(row)
    stable = 0
    for lane_rows in by_lane.values():
        errors = [
            row["relative_error"] for row in sorted(lane_rows, key=lambda r: r["eta"])
        ]
        stable += any(
            left <= tolerance and right <= tolerance
            for left, right in zip(errors[:-1], errors[1:], strict=False)
        )
    per_step = {}
    for row in rows:
        if row["usable"] and math.isfinite(row["relative_error"]):
            per_step.setdefault(row["eta"], []).append(row["relative_error"])
    return {
        "lanes": len(by_lane),
        "stable_interval_fraction": stable / len(by_lane) if by_lane else None,
        "median_relative_error": {
            str(eta): float(np.median(values)) for eta, values in per_step.items()
        },
        "tolerance": tolerance,
    }


def action_descent(scenes, paths, setup, *, weights, steps) -> list:
    """RSS27 M-02: unit raw-action steps per G/D/T group with S_a = I."""
    action = paths["action"]
    gradient = paths["action_gradient"]
    baseline = paths["losses"]
    rows = []
    for group, columns in action_groups(T.task_for_scenes(scenes)).items():
        projected = torch.zeros_like(gradient)
        projected[:, columns] = gradient[:, columns]
        norm = projected.norm(dim=-1)
        usable = paths["eligible"] & (norm > 1e-12)
        unit = projected / norm.clamp_min(1e-300)[:, None]
        for eta in steps:
            result, losses = solve_actions(action - eta * unit, scenes, setup, weights)
            available = T.valid_lanes(result, losses)
            done = converged(result)
            for lane in range(scenes.batch_size):
                rows.append(
                    {
                        "group": group,
                        "eta": eta,
                        "lane": lane,
                        "usable": bool(usable[lane]),
                        "converged": bool(done[lane]),
                        "delta": {
                            name: float(losses[name][lane] - baseline[name][lane])
                            if bool(available[lane])
                            else None
                            for name in COMPONENTS
                        },
                    }
                )
    return rows


def summarize_descent(rows) -> dict:
    summary = {}
    for row in rows:
        if not row["usable"] or row["delta"]["total"] is None:
            continue
        key = f"{row['group']}@{row['eta']}"
        entry = summary.setdefault(key, {"lanes": 0, "decreased": 0})
        entry["lanes"] += 1
        entry["decreased"] += row["delta"]["total"] < 0
    for entry in summary.values():
        entry["fraction"] = entry["decreased"] / entry["lanes"]
    return summary


def cliff_scan(
    policy,
    scenes,
    *,
    iterations,
    eligible,
    factors=RHO_FACTORS,
    base_calibrations=None,
) -> list:
    task = T.task_for_scenes(scenes)
    if base_calibrations is None:
        base_calibrations = T.task_calibrations(task)
    with torch.no_grad():
        decoded = T.decode_action(policy(scenes.features()), scenes)
    rows = []
    for group in ("corridor", "limits"):
        for factor in factors:
            calibrations = tuple(
                replace(
                    value,
                    hinge_weight=value.hinge_weight
                    * (
                        factor if (name == "corridor") == (group == "corridor") else 1.0
                    ),
                )
                for name, value in zip(
                    T.CONSTRAINT_NAMES, base_calibrations, strict=True
                )
            )
            setup = T.prepare_problem(
                scenes.batch_size,
                maximum_iterations=iterations,
                calibrations=calibrations,
                task=task,
            )
            with torch.no_grad():
                result = setup.solver.solve(setup.bind(scenes, decoded))
                valid = eligible & converged(result)
                violations = T.constraint_violations(result, decoded)
                summary = {}
                for name, value in zip(T.CONSTRAINT_NAMES, calibrations, strict=True):
                    data = violations[name][valid]
                    summary[name] = {
                        "rho": value.hinge_weight,
                        "p95_max_violation": float(
                            torch.quantile(data.flatten(1).amax(-1).clamp_min(0), 0.95)
                        )
                        if bool(valid.any())
                        else None,
                        "linear_hinge_fraction": float(
                            (data >= value.width).double().mean()
                        )
                        if bool(valid.any())
                        else None,
                    }
            rows.append(
                {
                    "group": group,
                    "factor": factor,
                    "converged_lanes": int(valid.sum()),
                    "lane_mask": valid.tolist(),
                    "constraints": summary,
                    "stationarity": result.diagnostics.stationarity.tolist(),
                    "optimizer_status": result.diagnostics.optimizer_status.tolist(),
                }
            )
    return rows


def reevaluate(run_dir: Path, *, iterations=T.REFERENCE_ITERATIONS) -> dict:
    """Final policy on its validation scenes with a converged-budget forward."""
    config = json.loads((run_dir / "config.json").read_text())
    task = T.TASKS[config.get("task", "gate2")]
    policy, _ = load_policy(run_dir / f"checkpoint_{config['steps']:06d}.pt", task)
    calibrations = tuple(T.Calibration(**value) for value in config["calibrations"])
    setup = T.prepare_problem(
        config["validation_size"],
        constraints=config["constraints"],
        maximum_iterations=iterations,
        calibrations=calibrations,
        task=task,
        corridor_margin=config.get("corridor_margin", 0.0),
    )
    scenes = T.sample_scenes(
        config["validation_size"],
        generator=torch.Generator().manual_seed(config["validation_seed"]),
        task=task,
        geometry_loss=config.get("geometry_loss", "current"),
        capsule_radius=config.get("capsule_radius", task.capsule_radius),
    )
    summary, _ = T.evaluate(
        policy,
        setup,
        scenes,
        weights=T.LossWeights(**config["weights"]),
        corridor_activation=calibrations[0].activation,
    )
    return summary


def measure_barrier_margin(run_dirs, *, output: Path | None = None) -> dict:
    """Q9: mean per-seed median gate-corridor slack, rounded to one cm."""
    rows = []
    pooled = []
    for run_dir in map(Path, run_dirs):
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        if (
            config.get("task") != "sgate3"
            or config["constraints"] != "relaxed-barrier"
            or config.get("corridor_margin", 0) != 0
            or summary["outcome"] != "completed"
            or summary["completed_steps"] != config["steps"]
        ):
            raise ValueError(f"not a completed sgate3 barrier run: {run_dir}")
        task = T.SGATE3
        policy, _ = load_policy(
            run_dir / f"checkpoint_{summary['completed_steps']:06d}.pt", task
        )
        calibrations = tuple(T.Calibration(**value) for value in config["calibrations"])
        scenes = T.sample_scenes(
            config["validation_size"],
            generator=torch.Generator().manual_seed(config["validation_seed"]),
            task=task,
        )
        setup = T.prepare_problem(
            scenes.batch_size,
            constraints="relaxed-barrier",
            maximum_iterations=config["maximum_iterations"],
            calibrations=calibrations,
            task=task,
        )
        with torch.no_grad():
            decoded = T.decode_action(policy(scenes.features()), scenes)
            result = setup.solver.solve(setup.bind(scenes, decoded))
        slack = -T.constraint_violations(result, decoded)["corridor"].amax(-1)
        pieces = T.gate_pieces(result, T.gate_traversal(result, scenes))
        values = [
            float(slack[lane, piece].min())
            for lane, lane_pieces in enumerate(pieces)
            for piece in lane_pieces
            if piece is not None
        ]
        if not values:
            raise RuntimeError(f"no gate-containing corridors in {run_dir}")
        pooled.extend(values)
        rows.append(
            {
                "seed": config["seed"],
                "run": str(run_dir),
                "gate_corridors": len(values),
                "minimum_slack_median": float(np.median(values)),
                "minimum_slack_p05": float(np.quantile(values, 0.05)),
                "minimum_slack_min": float(np.min(values)),
            }
        )
    seeds = [row["seed"] for row in rows]
    if seeds != list(range(42, 52)):
        raise ValueError(f"margin protocol requires ordered seeds 42..51, got {seeds}")
    seed_median_mean = float(np.mean([row["minimum_slack_median"] for row in rows]))
    report = {
        "definition": "per seed, median over every validation gate-containing corridor of its dense trajectory minimum slack; arithmetic mean over seeds; round to nearest 0.01 m",
        "seeds": rows,
        "seed_median_mean": seed_median_mean,
        "corridor_margin": round(seed_median_mean, 2),
        "pooled_gate_corridors": len(pooled),
        "pooled_median": float(np.median(pooled)),
    }
    if output is not None:
        T.save_json(output, report)
    return report


def profile_difference(rows, name) -> dict | None:
    grouped = {
        mode: np.asarray(
            [row[name] for row in rows if row["profile"] == mode and name in row],
            dtype=float,
        )
        for mode in PROFILES
    }
    if not all(len(value) >= 2 for value in grouped.values()):
        return None
    means = {mode: float(value.mean()) for mode, value in grouped.items()}
    deviations = {mode: float(value.std(ddof=1)) for mode, value in grouped.items()}
    difference = means[PROFILES[1]] - means[PROFILES[0]]
    return {
        "means": means,
        "seed_sd": deviations,
        "barrier_minus_hinge": difference,
        "exceeds_seed_sd": abs(difference) > max(deviations.values()),
    }


def training_comparison(run_dirs, *, reevaluation_iterations=T.REFERENCE_ITERATIONS):
    """Criterion 2: the predeclared task difference exceeds within-profile seed SD.

    The same difference must persist when every final policy is re-evaluated
    with a converged-budget forward, so it cannot come from endpoint quality.
    """
    rows, reevaluated = [], []
    for path in map(Path, run_dirs):
        config = json.loads((path / "config.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        rows.append(
            {**summary, "seed": config["seed"], "profile": config["constraints"]}
        )
        reevaluated.append(
            {
                **reevaluate(path, iterations=reevaluation_iterations),
                "seed": config["seed"],
                "profile": config["constraints"],
            }
        )
    complete = len(rows) == 6 and all(
        row["outcome"] == "completed" and row["completed_steps"] == 2000 for row in rows
    )
    names = (
        "task",
        "gate_success_fraction",
        "clearance_p05",
        "clearance_min",
        "collision_fraction",
        "pace_error",
        "acceleration",
        "jerk",
    )
    metrics = {name: profile_difference(rows, name) for name in names}
    converged_metrics = {name: profile_difference(reevaluated, name) for name in names}
    task, converged_task = metrics["task"], converged_metrics["task"]
    criterion_2 = bool(
        complete
        and task
        and converged_task
        and task["exceeds_seed_sd"]
        and converged_task["exceeds_seed_sd"]
        and task["barrier_minus_hinge"] * converged_task["barrier_minus_hinge"] > 0
    )
    admissible = complete and all(
        np.mean([row["gate_success_fraction"] for row in rows if row["profile"] == m])
        >= 0.95
        and min(row["gate_success_fraction"] for row in rows if row["profile"] == m)
        >= 0.90
        for m in PROFILES
    )
    return {
        "protocol_complete": complete,
        "criterion_2": criterion_2,
        "physical_admissibility": admissible,
        "primary_metric": "task",
        "rule": "profile-mean task difference exceeds both seed SDs, at the training budget and with a converged-budget re-evaluation",
        "metrics": metrics,
        "converged_reevaluation": converged_metrics,
        "runs": rows,
        "reevaluated_runs": reevaluated,
    }


def run_gate(
    *,
    warmup_checkpoint: Path,
    trained_checkpoint: Path,
    output: Path,
    scene_count=32,
    iterations=T.REFERENCE_ITERATIONS,
    weights=None,
    step_sizes=DESCENT_STEPS,
    directional_steps=DIRECTIONAL_STEPS,
    rho_factors=RHO_FACTORS,
    run_dirs=(),
    task: T.TaskConfig = T.GATE2,
    calibrations=None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if calibrations is None:
        calibrations = T.task_calibrations(task)
    if weights is None:
        weights = T.task_weights(task)
    base = T.sample_scenes(
        scene_count,
        generator=torch.Generator().manual_seed(2000),
        task=task,
    )
    scenes = base.select(torch.arange(scene_count).repeat_interleave(2))
    scenes = replace(
        scenes,
        smoothness=torch.tensor((0.2, 0.8), dtype=torch.float64).repeat(scene_count),
    )
    setups = {
        mode: T.prepare_problem(
            scenes.batch_size,
            constraints=mode,
            maximum_iterations=iterations,
            task=task,
            calibrations=calibrations,
        )
        for mode in PROFILES
    }
    pool, sources, directional, descent, cliffs = [], [], [], [], []
    for source, checkpoint in (
        ("warmup", warmup_checkpoint),
        ("trained", trained_checkpoint),
    ):
        policy, payload = load_policy(checkpoint, task)
        values = {
            mode: policy_paths(policy, scenes, setups[mode], weights=weights)
            for mode in PROFILES
        }
        hinge, barrier = (values[mode] for mode in PROFILES)
        eligible = hinge["eligible"] & barrier["eligible"]
        both_converged = hinge["converged"] & barrier["converged"]
        cosines = {
            name: lane_cosine(hinge["vectors"][name], barrier["vectors"][name])
            for name in hinge["vectors"]
        }
        shares = {
            mode: value["vectors"]["constraints"].norm(dim=-1)
            / value["vectors"]["total"].norm(dim=-1).clamp_min(1e-300)
            for mode, value in values.items()
        }
        eligible &= torch.isfinite(cosines["total"])
        for lane in range(scenes.batch_size):
            pool.append(
                {
                    "source": source,
                    "scene": lane // 2,
                    "intention": lane % 2,
                    "eligible": bool(eligible[lane]),
                    "converged": bool(both_converged[lane]),
                    "stationarity": {
                        mode: float(value["stationarity"][lane])
                        for mode, value in values.items()
                    },
                    "cosines": {
                        name: float(value[lane]) for name, value in cosines.items()
                    },
                    "constraint_share": {
                        mode: float(value[lane]) for mode, value in shares.items()
                    },
                    "gradient_norms": {
                        mode: {
                            name: float(vector[lane].norm())
                            for name, vector in value["vectors"].items()
                        }
                        for mode, value in values.items()
                    },
                }
            )
        for mode, value in values.items():
            result = value["result"]
            sources.append(
                {
                    "source": source,
                    "profile": mode,
                    "checkpoint": str(checkpoint),
                    "step": payload.get("step", 0),
                    "forward_status": result.status.tolist(),
                    "optimizer_status": result.diagnostics.optimizer_status.tolist(),
                    "stationarity": result.diagnostics.stationarity.tolist(),
                    "backward_status": value["backward_status"],
                    "anticipation": anticipation_metrics(
                        result,
                        value["decoded"],
                        value["input_cotangents"],
                        eligible,
                        mode,
                        calibrations,
                    ),
                }
            )
            rows = directional_check(
                policy,
                scenes,
                value,
                setups[mode],
                weights=weights,
                steps=directional_steps,
            )
            directional.append(
                {
                    "source": source,
                    "profile": mode,
                    "summary": summarize_directional(rows),
                    "rows": rows,
                }
            )
            rows = action_descent(
                scenes, value, setups[mode], weights=weights, steps=step_sizes
            )
            descent.append(
                {
                    "source": source,
                    "profile": mode,
                    "summary": summarize_descent(rows),
                    "rows": rows,
                }
            )
        cliffs.extend(
            {"source": source, **row}
            for row in cliff_scan(
                policy,
                scenes,
                iterations=iterations,
                eligible=eligible,
                factors=rho_factors,
                base_calibrations=calibrations,
            )
        )
        T.save_json(
            output / "gate_progress.json",
            {"pool": pool, "sources": sources},
        )

    def criterion(rows):
        if not rows:
            return None
        cosine = float(np.median([row["cosines"]["total"] for row in rows]))
        share = float(
            np.median([row["constraint_share"]["relaxed-barrier"] for row in rows])
        )
        return {
            "states": len(rows),
            "median_total_cosine": cosine,
            "median_barrier_constraint_share": share,
            "passed": cosine < 0.9 and share >= 0.25,
        }

    eligible_rows = [row for row in pool if row["eligible"]]
    criterion_1 = criterion(eligible_rows)
    converged_subset = criterion([row for row in eligible_rows if row["converged"]])
    comparison = training_comparison(run_dirs) if run_dirs else None
    passed = bool(
        comparison
        and comparison["protocol_complete"]
        and comparison["physical_admissibility"]
        and ((criterion_1 and criterion_1["passed"]) or comparison["criterion_2"])
    )
    decision_role = (
        "stage-zero acceptance criterion"
        if task is T.GATE2
        else "secondary analysis without an acceptance criterion"
    )
    report = {
        "outcome": ("passed" if passed else "not_passed")
        if task is T.GATE2
        else "analysis_complete",
        "decision_role": decision_role,
        "task": task.name,
        "seed": 2000,
        "reference_iterations": iterations,
        "states": len(pool),
        "eligible_states": len(eligible_rows),
        "criterion_1": criterion_1,
        "criterion_1_converged_subset": converged_subset,
        "criterion_1_definition": "L_total policy gradient including direct geometry/time; constraint share is barrier corridor plus three caps; all eligible states, convergence not required",
        "training_comparison": comparison,
        "directional_summary": [
            {key: entry[key] for key in ("source", "profile", "summary")}
            for entry in directional
        ],
        "descent_summary": [
            {key: entry[key] for key in ("source", "profile", "summary")}
            for entry in descent
        ],
        "weights": asdict(weights),
        "pool": pool,
        "sources": sources,
        "directional": directional,
        "finite_descent": descent,
        "cliff_scan": cliffs,
    }
    T.save_json(output / "gate.json", report)
    return report


def frozen_run_directories(runs_root: Path) -> list[Path]:
    runs = sorted(
        path.parent
        for path in runs_root.glob("*_seed*/config.json")
        if (path.parent / "summary.json").is_file()
        and (path.parent / "validation_lanes.json").is_file()
    )
    if not runs:
        raise ValueError(f"no completed frozen runs under {runs_root}")
    return runs


def load_frozen_evaluation(run: Path):
    """Replay one final policy with exactly its saved validation contract."""
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    saved = json.loads((run / "validation_lanes.json").read_text())
    task = T.TASKS[config["task"]]
    checkpoint = run / f"checkpoint_{summary['completed_steps']:06d}.pt"
    policy, payload = load_policy(checkpoint, task)
    if payload["step"] != summary["completed_steps"]:
        raise ValueError(f"checkpoint step does not match summary: {run}")
    scenes = T.sample_scenes(
        config["validation_size"],
        generator=torch.Generator().manual_seed(config["validation_seed"]),
        task=task,
    )
    setup = T.prepare_problem(
        scenes.batch_size,
        constraints=config["constraints"],
        maximum_iterations=config["maximum_iterations"],
        calibrations=tuple(T.Calibration(**value) for value in config["calibrations"]),
        task=task,
        corridor_margin=config["corridor_margin"],
    )
    weights = T.LossWeights(**config["weights"])
    with torch.no_grad():
        action = policy(scenes.features())
        decoded = T.decode_action(
            action,
            scenes,
            capsule_radius=config.get("capsule_radius", task.capsule_radius),
        )
        result = setup.solver.solve(setup.bind(scenes, decoded))
        losses = T.trajectory_losses(result, decoded, scenes, weights=weights)
        valid = T.valid_lanes(result, losses)
        traversal = T.gate_traversal(result, scenes)
        dense_minimum = losses["dense_clearance"].flatten(1).amin(-1)
    if valid.tolist() != saved["valid"]:
        raise ValueError(f"valid-lane replay differs: {run}")
    if traversal["continuous_collision"] != saved["continuous_collision"]:
        raise ValueError(f"continuous-collision replay differs: {run}")
    if not torch.equal(
        dense_minimum,
        torch.tensor(saved["dense_minimum_clearance"], dtype=torch.float64),
    ):
        raise ValueError(f"dense-clearance replay differs: {run}")
    return {
        "run": run,
        "config": config,
        "summary": summary,
        "saved": saved,
        "task": task,
        "policy": policy,
        "scenes": scenes,
        "setup": setup,
        "weights": weights,
        "action": action,
        "decoded": decoded,
        "result": result,
        "losses": losses,
    }


def wall_name(index: int, gates: int) -> str:
    if index < 2 * gates:
        return f"gate_{index // 2}_side_{index % 2}"
    return ("lower_boundary", "upper_boundary")[index - 2 * gates]


def finite_quantiles(values) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    quantiles = np.quantile(values, (0, 0.05, 0.5, 0.95, 1))
    return dict(
        zip(
            ("minimum", "p05", "median", "p95", "maximum"),
            quantiles.tolist(),
            strict=True,
        )
    )


def signed_summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        **finite_quantiles(values),
        "positive_fraction": float(np.mean(values > 0)) if len(values) else 0,
        "negative_fraction": float(np.mean(values < 0)) if len(values) else 0,
    }


def summarize_attribution(rows: list[dict]) -> dict:
    widths = np.asarray([row["minimum_cspace_half_width"] for row in rows])
    low, high = np.quantile(widths, (1 / 3, 2 / 3))
    for row in rows:
        width = row["minimum_cspace_half_width"]
        row["opening_tertile"] = (
            "narrow" if width <= low else "middle" if width <= high else "wide"
        )
    groups = {}
    for profile in ("hinge", "barrier", "hinge-margin"):
        selected = [row for row in rows if row["profile"] == profile]
        if not selected:
            continue
        collisions = [row for row in selected if row["continuous_collision"]]
        unsafe_boxes = [
            (depth, query)
            for row in selected
            for depth, query in zip(
                row["box_intrusion"], row["unsafe_query_inside"], strict=True
            )
            if depth > T.VALIDITY_BOX_PENETRATION
        ]
        diagnostic_boxes = [
            (depth, query)
            for row in selected
            for depth, query in zip(
                row["box_intrusion"], row["unsafe_query_inside"], strict=True
            )
            if depth > T.VALIDITY_CORRIDOR_TOLERANCE
        ]
        classification = {
            name: sum(row["classification"] == name for row in collisions)
            for name in ("inside_own_box", "outside_own_box", "dense_not_found")
        }
        groups[profile] = {
            "lanes": len(selected),
            "collision_lanes": len(collisions),
            "continuous_collision_fraction": len(collisions) / len(selected),
            "collision_classification_counts": classification,
            "collision_classification_fractions": {
                name: count / len(collisions) if collisions else 0
                for name, count in classification.items()
            },
            "lane_with_unsafe_box_fraction": np.mean(
                [row["unsafe_box_count"] > 0 for row in selected]
            ),
            "unsafe_box_fraction": len(unsafe_boxes)
            / sum(len(row["box_intrusion"]) for row in selected),
            "unsafe_box_with_query_fraction": np.mean(
                [query for _, query in unsafe_boxes]
            )
            if unsafe_boxes
            else 0,
            "lane_with_overlapping_box_fraction_1mm": np.mean(
                [
                    any(
                        depth > T.VALIDITY_CORRIDOR_TOLERANCE
                        for depth in row["box_intrusion"]
                    )
                    for row in selected
                ]
            ),
            "overlapping_box_fraction_1mm": len(diagnostic_boxes)
            / sum(len(row["box_intrusion"]) for row in selected),
            "overlapping_box_with_query_fraction_1mm": np.mean(
                [query for _, query in diagnostic_boxes]
            )
            if diagnostic_boxes
            else 0,
            "seed_collision_fraction": np.mean(
                [row["seed_collision"] for row in selected]
            ),
            "seed_collision_share_of_collision_lanes": np.mean(
                [row["seed_collision"] for row in collisions]
            )
            if collisions
            else 0,
            "collision_penetration": finite_quantiles(
                [row["penetration_depth"] for row in collisions]
            ),
            "collision_deeper_than_5cm_fraction": np.mean(
                [row["penetration_depth"] > 0.05 for row in collisions]
            )
            if collisions
            else 0,
            "v1": np.mean(
                [
                    depth > T.VALIDITY_BOX_PENETRATION
                    for row in selected
                    for depth in row["box_intrusion"]
                ]
            ),
            "v2": np.mean([row["seed_collision"] for row in selected]),
            "v3": np.mean(
                [row["classification"] == "inside_own_box" for row in selected]
            ),
            "fill_coverage": float(np.mean([row["fill_coverage"] for row in selected])),
            "lateral_half_extent": finite_quantiles(
                [value for row in selected for value in row["lateral_half_extent"]]
            ),
            "collision_by_opening_tertile": {
                tertile: float(
                    np.mean(
                        [
                            row["continuous_collision"]
                            for row in selected
                            if row["opening_tertile"] == tertile
                        ]
                    )
                )
                for tertile in ("narrow", "middle", "wide")
            },
        }
    return {
        "opening_tertile_boundaries": [float(low), float(high)],
        "profiles": groups,
    }


def plot_attribution(summary: dict, output: Path) -> Path:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = list(summary["profiles"])
    labels = {"hinge": "hinge", "barrier": "barrier", "hinge-margin": "hinge+margin"}
    colors = {
        "inside_own_box": "#dc2626",
        "outside_own_box": "#f59e0b",
        "dense_not_found": "#64748b",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    bottom = np.zeros(len(profiles))
    for name, color in colors.items():
        values = np.asarray(
            [
                summary["profiles"][profile]["collision_classification_fractions"][name]
                for profile in profiles
            ]
        )
        axes[0].bar(
            range(len(profiles)),
            values,
            bottom=bottom,
            label=name.replace("_", " "),
            color=color,
        )
        bottom += values
    axes[0].set(
        xticks=range(len(profiles)),
        xticklabels=[labels[value] for value in profiles],
        ylim=(0, 1),
        ylabel="Share of collision lanes",
        title="Deepest collision attribution",
    )
    axes[0].legend(fontsize=8)
    x = np.arange(3)
    width = 0.25
    for index, profile in enumerate(profiles):
        values = summary["profiles"][profile]["collision_by_opening_tertile"]
        axes[1].bar(
            x + (index - 1) * width,
            [values[name] for name in ("narrow", "middle", "wide")],
            width,
            label=labels[profile],
        )
    axes[1].set(
        xticks=x,
        xticklabels=("narrow", "middle", "wide"),
        ylim=(0, 1),
        ylabel="Continuous collision fraction",
        title="Collision by minimum C-space opening",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    path = output / "frozen_collision_attribution.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run_attribution(runs_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in frozen_run_directories(runs_root):
        evaluation = load_frozen_evaluation(run)
        scenes = evaluation["scenes"]
        decoded = evaluation["decoded"]
        validity = T.gate_validity(decoded, evaluation["result"], scenes)
        coverage = T.geometry_losses(decoded, scenes)["fill_coverage"]
        profile = run.name.rsplit("_seed", 1)[0]
        seed = evaluation["config"]["seed"]
        for lane in range(scenes.batch_size):
            collision = bool(validity["continuous_collision"][lane])
            if not collision:
                classification = "no_collision"
            elif bool(validity["dense_missed"][lane]):
                classification = "dense_not_found"
            elif bool(validity["inside_collision"][lane]):
                classification = "inside_own_box"
            else:
                classification = "outside_own_box"
            deepest_wall = int(validity["deepest_wall"][lane])
            rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "lane": lane,
                    "classification": classification,
                    "continuous_collision": collision,
                    "deepest_piece": int(validity["deepest_piece"][lane]),
                    "deepest_wall": wall_name(deepest_wall, evaluation["task"].gates),
                    "deepest_face_value": float(validity["deepest_face_value"][lane]),
                    "penetration_depth": float(
                        max(0, -validity["dense_minimum_clearance"][lane])
                    ),
                    "unsafe_box_count": int(validity["unsafe_box"][lane].sum()),
                    "box_intrusion": validity["box_intrusion"][lane].tolist(),
                    "unsafe_query_inside": validity["unsafe_query_inside"][
                        lane
                    ].tolist(),
                    "seed_collision": bool(validity["seed_collision"][lane]),
                    "seed_intrusion": float(validity["seed_intrusion"][lane]),
                    "minimum_cspace_half_width": float(
                        scenes.gate_half_widths[lane].min() - T.BODY_RADIUS
                    ),
                    "fill_coverage": float(coverage[lane]),
                    "lateral_half_extent": decoded.half_extents[lane, :, 1].tolist(),
                }
            )
    summary = summarize_attribution(rows)
    figure = plot_attribution(summary, output)
    report = {
        "runs_root": str(runs_root.resolve()),
        "runs": len(frozen_run_directories(runs_root)),
        "replay_verified": True,
        "intervals_per_piece": ATTRIBUTION_INTERVALS,
        **summary,
        "rows": rows,
        "figure": str(figure.resolve()),
    }
    T.save_json(output / "attribution.json", report)
    return report


def component_outward_gradient(action, task, value):
    gradient = torch.autograd.grad(value.sum(), action, retain_graph=True)[0]
    slack_action = action[:, task.slack].reshape(-1, task.pieces, T.FACES)
    db_draw = T.LENGTH_SCALE * torch.sigmoid(slack_action + T.SLACK_BIAS)
    return (-gradient[:, task.slack].reshape_as(db_draw) / db_draw).detach()


def constraint_node_violations(result, decoded, *, corridor_margin=0.0):
    samples = T.drive_samples(result.coefficients, result.durations, intervals=16)
    corridor_rows = T.corridor_with_margin(decoded.corridor, corridor_margin)
    corridor = (
        torch.einsum("bkfd,bkqd->bkqf", corridor_rows[..., :2], samples["position"])
        + corridor_rows[:, :, None, :, 2]
    ).transpose(-1, -2)

    def norm_violation(value, limit):
        return (value.square() - limit.square()) / (2 * limit)

    caps = {
        name: norm_violation(samples[name], decoded.caps[:, index, None, None])
        for index, name in enumerate(T.CAP_NAMES)
    }
    wheel = norm_violation(
        samples["wheel_speeds"],
        samples["wheel_speeds"].new_tensor(T.WHEEL_LIMIT),
    )
    return {"corridor": corridor, **caps, "wheel_speeds": wheel}


def classify_profile_nodes(violation, calibration, mode):
    """0=inactive, 1=transition/anticipation, 2=saturated/quadratic."""
    if mode == "hinge":
        return torch.where(
            violation <= 0,
            torch.zeros_like(violation, dtype=torch.int8),
            torch.where(
                violation < calibration.width,
                torch.ones_like(violation, dtype=torch.int8),
                torch.full_like(violation, 2, dtype=torch.int8),
            ),
        )
    slack = -violation
    if calibration.activation is None:
        return torch.where(
            slack > calibration.delta,
            torch.ones_like(violation, dtype=torch.int8),
            torch.full_like(violation, 2, dtype=torch.int8),
        )
    return torch.where(
        slack >= calibration.activation,
        torch.zeros_like(violation, dtype=torch.int8),
        torch.where(
            slack > calibration.delta,
            torch.ones_like(violation, dtype=torch.int8),
            torch.full_like(violation, 2, dtype=torch.int8),
        ),
    )


def input_regime(codes, mode):
    active_first = bool((codes == 1).any())
    active_second = bool((codes == 2).any())
    if active_first and active_second:
        return "mixed_active"
    if active_first:
        return "transition_only" if mode == "hinge" else "anticipation_only"
    if active_second:
        return "saturated_only" if mode == "hinge" else "quadratic_only"
    return "inactive_only"


def pressure_setup(evaluation, damping):
    original = evaluation["setup"]
    spec = replace(
        original.plan.spec,
        backward=replace(
            original.plan.spec.backward,
            damping=damping,
        ),
    )
    plan = p.build_plan(spec, original.plan.shape)
    return T.TrajectorySetup(
        plan,
        T.build_solver(plan, dtype=torch.float64, backward=True),
        original.task,
        original.corridor_margin,
    )


def public_input_pressures(parameters, decoded):
    """Convert public input cotangents to the declared relaxation signs."""
    corridor = parameters["corridor"]
    normal = decoded.corridor[..., :2]
    orientation = (corridor[..., :2] * T.perpendicular(normal)).sum(-1)
    return {
        # The public row stores c=-b, hence dL/dc=-dL/db.
        "corridor_offset": corridor[..., 2].detach(),
        "corridor_orientation": orientation.detach(),
        # Increasing a cap relaxes it, so relaxation pressure is -dL/dcap.
        **{name: (-parameters[name]).detach() for name in T.CAP_NAMES},
    }


def task_pressures(evaluation, *, setup=None, split=True) -> dict:
    """Parameter-level task pressures and the native pullback diagnostics."""
    scenes = evaluation["scenes"]
    decoded = evaluation["decoded"]
    setup = evaluation["setup"] if setup is None else setup
    with torch.no_grad():
        result, state = setup.solver.solve_with_state(setup.bind(scenes, decoded))
    names = p.ResultSeeds._fields
    leaves = tuple(
        getattr(result, name).detach().requires_grad_(True) for name in names
    )
    output = replace(result, **dict(zip(names, leaves, strict=True)))
    train_clearance = evaluation["config"]["loss_definition"]["train_clearance"]
    losses = T.trajectory_losses(
        output,
        decoded,
        scenes,
        weights=evaluation["weights"],
        train_clearance=train_clearance,
        geometry_loss=evaluation["config"].get("geometry_loss", "current"),
    )
    available = T.valid_lanes(output, losses)
    weights = evaluation["weights"]
    acceleration_weight = weights.acceleration_low * torch.exp(
        math.log(10) * scenes.smoothness
    )
    jerk_weight = weights.jerk_low * torch.exp(math.log(10) * scenes.smoothness)
    components = {
        "acceleration": acceleration_weight * losses["acceleration"],
        "jerk": jerk_weight * losses["jerk"],
        "gate": weights.gate * losses["gate"],
        "clearance": weights.clearance * losses["clearance_loss"]
        if train_clearance
        else 0 * losses["clearance_loss"],
    }

    def pull(value):
        seeds = torch.autograd.grad(
            value[available].sum(), leaves, retain_graph=True, allow_unused=True
        )
        return setup.solver.pullback(
            state,
            p.ResultSeeds(
                **{
                    name: None if seed is None else seed.detach().contiguous()
                    for name, seed in zip(names, seeds, strict=True)
                }
            ),
        )

    total = pull(losses["task"])
    pulled = {"total": total}
    if split:
        pulled.update({name: pull(value) for name, value in components.items()})
    applied = available & (total.diagnostics.status == int(BackwardStatus.APPLIED))
    for value in total.parameters.values():
        applied &= torch.isfinite(value.reshape(value.shape[0], -1)).all(-1)

    pressure = {
        name: public_input_pressures(value.parameters, decoded)
        for name, value in pulled.items()
    }
    if split:
        pressure["pace"] = {
            name: torch.zeros_like(value) for name, value in pressure["total"].items()
        }
        summed = {
            name: sum(
                pressure[component][name]
                for component in (
                    "pace",
                    "acceleration",
                    "jerk",
                    "gate",
                    "clearance",
                )
            )
            for name in pressure["total"]
        }
        absolute_error = max(
            float((summed[name] - pressure["total"][name]).abs().max())
            for name in summed
        )
        scale = max(
            1.0,
            max(float(value.abs().max()) for value in pressure["total"].values()),
        )
        relative_error = absolute_error / scale
    else:
        absolute_error = None
        relative_error = None
    diagnostics = total.diagnostics
    return {
        "result": result,
        "pressure": pressure,
        "applied": applied.detach(),
        "linearity_max_absolute_error": absolute_error,
        "linearity_max_relative_error": relative_error,
        "diagnostics": {
            "status": diagnostics.status.detach(),
            "stationarity": diagnostics.stationarity.detach(),
            "hessian_damping_shift": diagnostics.hessian_damping_shift.detach(),
            "minimum_relative_cholesky_pivot": diagnostics.minimum_relative_cholesky_pivot.detach(),
            "adjoint_residual_ratio": diagnostics.adjoint_residual_ratio.detach(),
        },
        "raw_total_parameters": {
            name: value.detach() for name, value in total.parameters.items()
        },
    }


def pressure_vector(pressure: dict[str, torch.Tensor]) -> torch.Tensor:
    batch = pressure["corridor_offset"].shape[0]
    return torch.cat(
        (
            pressure["corridor_offset"].reshape(batch, -1),
            pressure["corridor_orientation"].reshape(batch, -1),
            *(pressure[name].reshape(batch, -1) for name in T.CAP_NAMES),
        ),
        dim=1,
    )


def pressure_input_rows(evaluation, measured, node_codes, profile, seed) -> list[dict]:
    components = ("pace", "acceleration", "jerk", "gate", "clearance", "total")
    mode = evaluation["config"]["constraints"]
    mode = "barrier" if mode == "relaxed-barrier" else "hinge"
    task = evaluation["task"]
    applied = measured["applied"]
    crossing_pieces = T.gate_pieces(
        measured["result"], T.gate_traversal(measured["result"], evaluation["scenes"])
    )
    near_gate = torch.zeros(
        (evaluation["scenes"].batch_size, task.pieces), dtype=torch.bool
    )
    for lane, pieces in enumerate(crossing_pieces):
        for piece in pieces:
            if piece is not None:
                near_gate[lane, piece] = True
    rows = []
    corridor_codes = node_codes["corridor"]
    for lane, piece, face in (
        applied[:, None, None].expand_as(corridor_codes[..., 0]).nonzero().tolist()
    ):
        codes = corridor_codes[lane, piece, face]
        rows.append(
            {
                "profile": profile,
                "seed": seed,
                "lane": lane,
                "constraint": "corridor",
                "piece": piece,
                "face": face,
                "near_gate": bool(near_gate[lane, piece]),
                "regime": input_regime(codes, mode),
                "node_fractions": [
                    float((codes == index).double().mean()) for index in range(3)
                ],
                "outward": {
                    name: float(
                        measured["pressure"][name]["corridor_offset"][lane, piece, face]
                    )
                    for name in components
                },
                "orientation": {
                    name: float(
                        measured["pressure"][name]["corridor_orientation"][
                            lane, piece, face
                        ]
                    )
                    for name in components
                },
            }
        )
    for name in T.CAP_NAMES:
        codes_by_lane = node_codes[name].flatten(1)
        for lane in applied.nonzero().flatten().tolist():
            codes = codes_by_lane[lane]
            rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "lane": lane,
                    "constraint": name,
                    "piece": None,
                    "face": None,
                    "near_gate": None,
                    "regime": input_regime(codes, mode),
                    "node_fractions": [
                        float((codes == index).double().mean()) for index in range(3)
                    ],
                    "outward": {
                        component: float(measured["pressure"][component][name][lane])
                        for component in components
                    },
                    "orientation": None,
                }
            )
    return rows


def summarize_pressure(
    rows: list[dict],
    room_rows: list[dict],
    input_rows: list[dict],
    node_rows: list[dict],
    fidelity_rows: list[dict],
    lm_rows: list[dict],
    share_rows: list[dict],
) -> dict:
    profiles = {}
    preferred_components = (
        "task_implicit",
        "unsafe",
        "overrun",
        "underfill",
        "exclude",
        "seed",
        "fill",
        "size",
        "room",
    )
    for profile in ("hinge", "barrier", "hinge-margin"):
        selected = [row for row in rows if row["profile"] == profile]
        selected_room = [row for row in room_rows if row["profile"] == profile]
        selected_inputs = [row for row in input_rows if row["profile"] == profile]
        selected_nodes = [row for row in node_rows if row["profile"] == profile]
        selected_fidelity_all = [
            row for row in fidelity_rows if row["profile"] == profile
        ]
        selected_fidelity = [row for row in selected_fidelity_all if row["applied"]]
        selected_lm = [row for row in lm_rows if row["profile"] == profile]
        selected_share = [row for row in share_rows if row["profile"] == profile]
        if not selected:
            continue
        components = tuple(
            name
            for name in preferred_components
            if any(name in row["components"] for row in selected)
        )
        node_intervals = {}
        for constraint in ("corridor", *T.CAP_NAMES, "wheel_speeds"):
            counts = np.asarray(
                [
                    row["counts"]
                    for row in selected_nodes
                    if row["constraint"] == constraint
                ]
            ).sum(0)
            node_intervals[constraint] = {
                "counts": counts.tolist(),
                "fractions": (counts / counts.sum()).tolist(),
                "second_active_share": float(counts[2] / counts[1:].sum())
                if counts[1:].sum()
                else 0,
            }
        input_pressure = {}
        for constraint in ("corridor", *T.CAP_NAMES):
            constraint_rows = [
                row for row in selected_inputs if row["constraint"] == constraint
            ]
            input_pressure[constraint] = {}
            for regime in sorted({row["regime"] for row in constraint_rows}):
                regime_rows = [
                    row for row in constraint_rows if row["regime"] == regime
                ]
                total = [row["outward"]["total"] for row in regime_rows]
                value = {
                    "inputs": len(regime_rows),
                    "total_outward": signed_summary(total),
                    "outward_fraction": float(np.mean(np.asarray(total) > 0)),
                    "inward_fraction": float(np.mean(np.asarray(total) < 0)),
                    "components": {
                        name: signed_summary(
                            [row["outward"][name] for row in regime_rows]
                        )
                        for name in (
                            "pace",
                            "acceleration",
                            "jerk",
                            "gate",
                            "clearance",
                        )
                    },
                }
                if constraint == "corridor":
                    value["orientation_absolute"] = finite_quantiles(
                        [abs(row["orientation"]["total"]) for row in regime_rows]
                    )
                    value["near_gate_fraction"] = float(
                        np.mean([row["near_gate"] for row in regime_rows])
                    )
                    value["location_components"] = {
                        location: {
                            "inputs": len(location_rows),
                            "components": {
                                name: signed_summary(
                                    [row["outward"][name] for row in location_rows]
                                )
                                for name in (
                                    "acceleration",
                                    "jerk",
                                    "gate",
                                    "clearance",
                                    "total",
                                )
                            },
                        }
                        for location, location_rows in (
                            (
                                "near_gate",
                                [row for row in regime_rows if row["near_gate"]],
                            ),
                            (
                                "away_from_gate",
                                [row for row in regime_rows if not row["near_gate"]],
                            ),
                        )
                    }
                input_pressure[constraint][regime] = value
        saturated_corridor = [
            row
            for row in selected_inputs
            if row["constraint"] == "corridor" and row["regime"] == "saturated_only"
        ]
        profiles[profile] = {
            "faces": len(selected),
            "outward_pressure": {
                name: {
                    **finite_quantiles([row["components"][name] for row in selected]),
                    "positive_fraction": float(
                        np.mean([row["components"][name] > 0 for row in selected])
                    ),
                }
                for name in components
            },
            "task_pressure_components": {
                name: signed_summary([row["task_components"][name] for row in selected])
                for name in (
                    "pace",
                    "acceleration",
                    "jerk",
                    "gate",
                    "clearance",
                )
            },
            "net_outward_fraction": float(
                np.mean([row["net_outward"] > 0 for row in selected])
            ),
            "task_exceeds_direct_inward_fraction": float(
                np.mean(
                    [
                        row["components"]["task_implicit"]
                        > max(0, -row["direct_outward"])
                        for row in selected
                    ]
                )
            ),
            "room_crossings": len(selected_room),
            "room_target": finite_quantiles(
                [row["room_target"] for row in selected_room]
            ),
            "cspace_half_width": finite_quantiles(
                [row["cspace_half_width"] for row in selected_room]
            ),
            "box_lateral_half_extent": finite_quantiles(
                [row["box_lateral_half_extent"] for row in selected_room]
            ),
            "box_wider_than_cspace_fraction": float(
                np.mean(
                    [
                        row["box_lateral_half_extent"] > row["cspace_half_width"]
                        for row in selected_room
                    ]
                )
            )
            if selected_room
            else 0,
            "node_intervals": node_intervals,
            "input_pressure": input_pressure,
            "saturated_corridor_check": {
                "inputs": len(saturated_corridor),
                "offset_absolute": finite_quantiles(
                    [abs(row["outward"]["total"]) for row in saturated_corridor]
                ),
                "orientation_absolute": finite_quantiles(
                    [abs(row["orientation"]["total"]) for row in saturated_corridor]
                ),
            },
            "policy_constraint_share": finite_quantiles(
                [row["constraint_share"] for row in selected_share]
            ),
            "linearity_max_absolute_error": max(
                row["linearity_max_absolute_error"] for row in selected_fidelity_all
            ),
            "pullback_applied_fraction": len(selected_fidelity)
            / len(selected_fidelity_all),
            "fidelity": {
                name: finite_quantiles([row[name] for row in selected_fidelity])
                for name in (
                    "stationarity",
                    "hessian_damping_shift",
                    "minimum_relative_cholesky_pivot",
                    "adjoint_residual_ratio",
                )
            },
            "lm_shift_positive_fraction": float(
                np.mean([row["hessian_damping_shift"] > 0 for row in selected_fidelity])
            ),
            "no_regularization": {
                "shifted_lanes": len(selected_lm),
                "cholesky_failures": sum(row["cholesky_failed"] for row in selected_lm),
                "comparable_lanes": sum(
                    not row["cholesky_failed"] for row in selected_lm
                ),
                "pressure_cosine": finite_quantiles(
                    [row["pressure_cosine"] for row in selected_lm]
                ),
                "pressure_norm_ratio": finite_quantiles(
                    [row["pressure_norm_ratio"] for row in selected_lm]
                ),
            },
        }
    return profiles


def plot_pressure_summary(profiles: dict, output: Path) -> Path:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile_names = ("hinge", "barrier", "hinge-margin")
    constraints = (
        "corridor",
        "speed_cap",
        "centripetal_cap",
        "forward_acceleration_cap",
        "wheel_speeds",
    )
    short = {
        "corridor": "corridor",
        "speed_cap": "speed",
        "centripetal_cap": "centripetal",
        "forward_acceleration_cap": "forward accel",
        "wheel_speeds": "wheel",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    x = np.arange(len(constraints))
    width = 0.24
    active_colors = ("#94a3b8", "#2563eb", "#dc2626")
    for profile_index, profile in enumerate(profile_names):
        bottom = np.zeros(len(constraints))
        for regime_index, color in enumerate(active_colors):
            values = np.asarray(
                [
                    profiles[profile]["node_intervals"][name]["fractions"][regime_index]
                    for name in constraints
                ]
            )
            axes[0].bar(
                x + (profile_index - 1) * width,
                values,
                width,
                bottom=bottom,
                color=color,
                label=(
                    "inactive",
                    "transition / anticipation",
                    "saturated / quadratic",
                )[regime_index]
                if profile_index == 0
                else None,
            )
            bottom += values
    axes[0].set(
        xticks=x,
        xticklabels=[short[name] for name in constraints],
        ylim=(0, 1),
        ylabel="Quadrature-node fraction",
        title="Constraint profile coverage",
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    for index, profile in enumerate(profile_names):
        axes[0].text(
            0.01,
            0.97 - 0.07 * index,
            ("H", "B", "H+M")[index] + f" = {profile}",
            transform=axes[0].transAxes,
            va="top",
            fontsize=8,
        )

    components = ("acceleration", "jerk", "gate", "clearance")
    colors = {"hinge": "#2563eb", "barrier": "#d97706", "hinge-margin": "#16a34a"}
    for profile_index, profile in enumerate(profile_names):
        active = profiles[profile]["input_pressure"]["corridor"]
        regime = "anticipation_only" if profile == "barrier" else "transition_only"
        values = [active[regime]["components"][name]["median"] for name in components]
        axes[1].bar(
            np.arange(len(components)) + (profile_index - 1) * width,
            values,
            width,
            color=colors[profile],
            label=profile.replace("hinge-margin", "hinge+margin"),
        )
    axes[1].axhline(0, color="#0f172a", linewidth=0.8)
    axes[1].set(
        xticks=np.arange(len(components)),
        xticklabels=components,
        ylabel="Median outward pressure",
        title="Active corridor faces by outer term",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    path = output / "frozen_pressure.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run_pressure(runs_root: Path, output: Path) -> dict:
    import gzip

    output.mkdir(parents=True, exist_ok=True)
    rows, room_rows, input_rows, node_rows = [], [], [], []
    fidelity_rows, lm_rows, share_rows = [], [], []
    for run in frozen_run_directories(runs_root):
        evaluation = load_frozen_evaluation(run)
        scenes = evaluation["scenes"]
        task = evaluation["task"]
        action = evaluation["action"].detach().requires_grad_(True)
        decoded = T.decode_action(
            action,
            scenes,
            capsule_radius=evaluation["config"].get(
                "capsule_radius", task.capsule_radius
            ),
        )
        evaluation = {**evaluation, "action": action, "decoded": decoded}
        validity = T.gate_validity(decoded, evaluation["result"], scenes)
        measured = task_pressures(evaluation)
        if measured["linearity_max_relative_error"] > 2e-12:
            raise ValueError(f"component pullbacks are not linear: {run}")
        applied = measured["applied"]
        profile = run.name.rsplit("_seed", 1)[0]
        seed = evaluation["config"]["seed"]
        mode = (
            "barrier"
            if evaluation["config"]["constraints"] == "relaxed-barrier"
            else "hinge"
        )
        violations = constraint_node_violations(
            evaluation["result"],
            decoded,
            corridor_margin=evaluation["config"]["corridor_margin"],
        )
        calibrations = dict(
            zip(
                T.CONSTRAINT_NAMES,
                (
                    T.Calibration(**value)
                    for value in evaluation["config"]["calibrations"]
                ),
                strict=True,
            )
        )
        node_codes = {
            name: classify_profile_nodes(value, calibrations[name], mode)
            for name, value in violations.items()
        }
        for name, codes in node_codes.items():
            node_rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "constraint": name,
                    "counts": [int((codes == index).sum()) for index in range(3)],
                }
            )
        input_rows.extend(
            pressure_input_rows(evaluation, measured, node_codes, profile, seed)
        )
        diagnostics = measured["diagnostics"]
        for lane in range(scenes.batch_size):
            fidelity_rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "lane": lane,
                    "applied": bool(applied[lane]),
                    "linearity_max_absolute_error": measured[
                        "linearity_max_absolute_error"
                    ],
                    "linearity_max_relative_error": measured[
                        "linearity_max_relative_error"
                    ],
                    **{
                        name: float(value[lane])
                        for name, value in diagnostics.items()
                        if name != "status"
                    },
                }
            )

        paths = policy_paths(
            evaluation["policy"],
            scenes,
            evaluation["setup"],
            weights=evaluation["weights"],
            train_clearance=evaluation["config"]["loss_definition"]["train_clearance"],
            geometry_loss=evaluation["config"].get("geometry_loss", "current"),
            capsule_radius=evaluation["config"].get(
                "capsule_radius", evaluation["task"].capsule_radius
            ),
        )
        constraint_share = paths["vectors"]["constraints"].norm(dim=-1) / paths[
            "vectors"
        ]["total"].norm(dim=-1).clamp_min(1e-300)
        for lane in (
            (paths["eligible"] & torch.isfinite(constraint_share))
            .nonzero()
            .flatten()
            .tolist()
        ):
            share_rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "lane": lane,
                    "constraint_share": float(constraint_share[lane]),
                }
            )

        geometry_kind = evaluation["config"].get("geometry_loss", "current")
        geometry = T.geometry_losses(decoded, scenes, geometry_loss=geometry_kind)
        weights = evaluation["weights"]
        if geometry_kind == "current":
            definition = T.loss_definition(task)
            coefficients = {
                "exclude": weights.geometry,
                "seed": weights.geometry,
                "fill": weights.geometry * definition.fill_weight,
                "size": weights.geometry * definition.size_weight,
                "room": weights.geometry * definition.room_weight,
            }
        else:
            coefficients = {
                "unsafe": weights.geometry,
                "overrun": weights.geometry,
                "underfill": weights.geometry * T.DIFF2_UNDERFILL_WEIGHT,
                "fill": weights.geometry,
                "seed": weights.geometry,
                "size": weights.geometry * T.DIFF2_SIZE_WEIGHT,
            }
        direct = {
            name: component_outward_gradient(action, task, coefficient * geometry[name])
            for name, coefficient in coefficients.items()
        }
        nearest_wall = validity["box_wall_distance"].argmin(-1)
        selected_wall_center = scenes.wall_centers.gather(
            1, nearest_wall[..., None].expand(-1, -1, 2)
        )
        toward_wall = selected_wall_center - decoded.box_centers
        face = torch.einsum(
            "bkfd,bkd->bkf", decoded.corridor[..., :2], toward_wall
        ).argmax(-1)
        eligible_face = validity["unsafe_box"] & applied[:, None]
        for lane, piece in eligible_face.nonzero().tolist():
            face_index = int(face[lane, piece])
            task_components = {
                name: float(
                    measured["pressure"][name]["corridor_offset"][
                        lane, piece, face_index
                    ]
                )
                for name in (
                    "pace",
                    "acceleration",
                    "jerk",
                    "gate",
                    "clearance",
                )
            }
            components = {
                "task_implicit": float(
                    measured["pressure"]["total"]["corridor_offset"][
                        lane, piece, face_index
                    ]
                ),
                **{
                    name: float(value[lane, piece, face_index])
                    for name, value in direct.items()
                },
            }
            direct_outward = sum(
                value for name, value in components.items() if name != "task_implicit"
            )
            rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "lane": lane,
                    "piece": piece,
                    "face": face_index,
                    "wall": wall_name(int(nearest_wall[lane, piece]), task.gates),
                    "intrusion": float(validity["box_intrusion"][lane, piece]),
                    "components": components,
                    "task_components": task_components,
                    "direct_outward": direct_outward,
                    "net_outward": direct_outward + components["task_implicit"],
                }
            )

        room_target = T.lateral_room_target(decoded, scenes)
        crossing_pieces = T.gate_pieces(evaluation["result"], validity["traversal"])
        for lane, pieces in enumerate(crossing_pieces):
            for gate, piece in enumerate(pieces):
                if piece is None:
                    continue
                room_rows.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "lane": lane,
                        "gate": gate,
                        "piece": piece,
                        "room_target": float(room_target[lane, piece]),
                        "cspace_half_width": float(
                            scenes.gate_half_widths[lane, gate] - T.BODY_RADIUS
                        ),
                        "box_lateral_half_extent": float(
                            decoded.half_extents[lane, piece, 1].detach()
                        ),
                    }
                )

        shifted = applied & (diagnostics["hessian_damping_shift"] > 0)
        if bool(shifted.any()):
            no_regularization = pressure_setup(evaluation, p.NoRegularization())
            with torch.no_grad():
                no_regularization_forward = no_regularization.solver.solve(
                    no_regularization.bind(scenes, decoded)
                )
            for name in (*p.ResultSeeds._fields, "status"):
                if not torch.equal(
                    getattr(no_regularization_forward, name),
                    getattr(evaluation["result"], name),
                ):
                    raise ValueError(
                        f"NoRegularization changed the forward readout {name}: {run}"
                    )
            undamped = task_pressures(evaluation, setup=no_regularization, split=False)
            lm_vector = pressure_vector(measured["pressure"]["total"])
            undamped_vector = pressure_vector(undamped["pressure"]["total"])
            lm_norm = lm_vector.norm(dim=-1)
            undamped_norm = undamped_vector.norm(dim=-1)
            common = shifted & undamped["applied"]
            cosine = (lm_vector * undamped_vector).sum(-1) / (
                lm_norm * undamped_norm
            ).clamp_min(1e-300)
            ratio = undamped_norm / lm_norm.clamp_min(1e-300)
            for lane in shifted.nonzero().flatten().tolist():
                usable = bool(common[lane])
                lm_rows.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "lane": lane,
                        "cholesky_failed": not usable,
                        "pressure_cosine": float(cosine[lane]) if usable else None,
                        "pressure_norm_ratio": float(ratio[lane]) if usable else None,
                    }
                )
    raw_path = output / "pressure_rows.json.gz"
    with gzip.open(raw_path, "wt") as stream:
        json.dump(
            {
                "faces": rows,
                "room_crossings": room_rows,
                "inputs": input_rows,
                "node_counts": node_rows,
                "fidelity": fidelity_rows,
                "lm_comparison": lm_rows,
                "policy_constraint_share": share_rows,
            },
            stream,
            separators=(",", ":"),
        )
    profiles = summarize_pressure(
        rows,
        room_rows,
        input_rows,
        node_rows,
        fidelity_rows,
        lm_rows,
        share_rows,
    )
    figure = plot_pressure_summary(profiles, output)
    report = {
        "runs_root": str(runs_root.resolve()),
        "runs": len(frozen_run_directories(runs_root)),
        "replay_verified": True,
        "face_selection": "nearest wall for each V1-unsafe box; face normal most aligned with wall-center direction",
        "pressure_sign": "positive means the weighted term favors increasing b and expanding the face outward",
        "profiles": profiles,
        "node_interval_labels": {
            "hinge": ["inactive", "transition", "saturated"],
            "barrier": ["inactive", "anticipation", "quadratic"],
        },
        "wheel_pressure": "not_applicable: fixed bound",
        "figure": str(figure.resolve()),
        "raw_evidence": str(raw_path.resolve()),
    }
    T.save_json(output / "pressure.json", report)
    return report


def instance_variant_name(geometry_loss: str, capsule_radius: float) -> str:
    if geometry_loss not in T.GEOMETRY_LOSS_KINDS:
        raise ValueError(f"unknown geometry loss: {geometry_loss!r}")
    if capsule_radius not in INSTANCE_CAPSULE_RADII:
        raise ValueError(f"undeclared capsule radius: {capsule_radius}")
    radius = f"{capsule_radius:.2f}".replace(".", "p")
    return f"{geometry_loss}_r{radius}"


def instance_warmup_path(
    output: Path, geometry_loss: str, capsule_radius: float, seed: int
) -> Path:
    return (
        output
        / "warmup"
        / instance_variant_name(geometry_loss, capsule_radius)
        / f"seed{seed}.pt"
    )


@torch.no_grad()
def evaluate_instance_action(
    action: torch.Tensor,
    scenes: T.SceneBatch,
    *,
    geometry_loss: str,
    capsule_radius: float,
) -> dict:
    decoded = T.decode_action(action, scenes, capsule_radius=capsule_radius)
    geometry = T.geometry_losses(decoded, scenes, geometry_loss=geometry_loss)
    report = {
        "lanes": scenes.batch_size,
        "boxes": scenes.batch_size * T.SGATE3.pieces,
        "geometry": {name: float(value.mean()) for name, value in geometry.items()},
        "solvers": {},
    }
    for profile in ("hinge", "relaxed-barrier"):
        setup = T.prepare_problem(
            scenes.batch_size,
            constraints=profile,
            maximum_iterations=T.MAXIMUM_ITERATIONS,
            calibrations=T.SGATE3_CALIBRATIONS,
            task=T.SGATE3,
        )
        result = setup.solver.solve(setup.bind(scenes, decoded))
        losses = T.trajectory_losses(
            result,
            decoded,
            scenes,
            weights=T.SGATE3_LOSS_WEIGHTS,
            train_clearance=False,
            geometry_loss=geometry_loss,
        )
        valid = T.valid_lanes(result, losses)
        validity = T.gate_validity(decoded, result, scenes)
        report["solvers"][profile] = {
            "valid_lanes": int(valid.sum()),
            "failed_lanes": int((~valid).sum()),
            "v1_unsafe_boxes": int(validity["unsafe_box"].sum()),
            "v1_total_boxes": validity["unsafe_box"].numel(),
            "v2_seed_collision_lanes": int(validity["seed_collision"].sum()),
            "v2_total_lanes": scenes.batch_size,
            "v3_inside_collision_lanes": int(validity["inside_collision"].sum()),
            "v3_total_lanes": scenes.batch_size,
            "continuous_collision_lanes": int(validity["continuous_collision"].sum()),
            "evaluation_task_sum": float(losses["task"][valid].sum())
            if bool(valid.any())
            else None,
            "evaluation_task_lanes": int(valid.sum()),
        }
    return report


def run_instance_warmup(
    output: Path,
    *,
    geometry_loss: str,
    capsule_radius: float,
    seed: int,
    steps: int = 300,
) -> dict:
    if seed not in INSTANCE_SEEDS:
        raise ValueError(f"undeclared instance seed: {seed}")
    checkpoint = instance_warmup_path(output, geometry_loss, capsule_radius, seed)
    if checkpoint.exists():
        payload = torch.load(checkpoint, weights_only=True)
        if (
            payload["seed"] != seed
            or payload["warmup_steps"] != steps
            or payload["geometry_loss"] != geometry_loss
            or payload["capsule_radius"] != capsule_radius
        ):
            raise ValueError(
                f"existing instance warm-up has different settings: {checkpoint}"
            )
    else:
        payload = T.warmup_policy(
            seed=seed,
            steps=steps,
            batch_size=48,
            checkpoint=checkpoint,
            weights=T.WARMUP_WEIGHTS,
            task=T.SGATE3,
            geometry_loss=geometry_loss,
            capsule_radius=capsule_radius,
        )
    policy = T.GatePolicy(T.SGATE3).double()
    policy.load_state_dict(payload["policy"])
    scenes = T.sample_scenes(
        128,
        generator=torch.Generator().manual_seed(seed + 1),
        task=T.SGATE3,
        geometry_loss=geometry_loss,
        capsule_radius=capsule_radius,
    )
    with torch.no_grad():
        action = policy(scenes.features())
        evaluation = evaluate_instance_action(
            action,
            scenes,
            geometry_loss=geometry_loss,
            capsule_radius=capsule_radius,
        )
    report = {
        "phase": "warmup",
        "seed": seed,
        "steps": steps,
        "geometry_loss": geometry_loss,
        "capsule_radius": capsule_radius,
        "checkpoint": str(checkpoint.resolve()),
        "evaluation": evaluation,
    }
    T.save_json(checkpoint.with_name(f"seed{seed}_summary.json"), report)
    return report


def run_instance_a(
    output: Path,
    *,
    geometry_loss: str,
    capsule_radius: float,
    seed: int,
    learning_rate: float,
    steps: int = INSTANCE_A_STEPS,
) -> dict:
    if seed not in INSTANCE_SEEDS:
        raise ValueError(f"undeclared instance seed: {seed}")
    if learning_rate not in INSTANCE_LEARNING_RATES:
        raise ValueError(f"undeclared instance learning rate: {learning_rate}")
    warmup = instance_warmup_path(output, geometry_loss, capsule_radius, seed)
    if not warmup.is_file():
        raise ValueError(f"missing instance warm-up: {warmup}")
    payload = torch.load(warmup, weights_only=True)
    if (
        payload["geometry_loss"] != geometry_loss
        or payload["capsule_radius"] != capsule_radius
    ):
        raise ValueError("instance warm-up does not match A settings")
    policy = T.GatePolicy(T.SGATE3).double()
    policy.load_state_dict(payload["policy"])
    scenes = T.sample_scenes(
        128,
        generator=torch.Generator().manual_seed(seed + 1),
        task=T.SGATE3,
        geometry_loss=geometry_loss,
        capsule_radius=capsule_radius,
    )
    with torch.no_grad():
        initial = policy(scenes.features())
    geometry_action = torch.nn.Parameter(initial[:, : T.SGATE3.geometry_dim].clone())
    fixed_action = initial[:, T.SGATE3.geometry_dim :].clone()
    optimizer = torch.optim.Adam((geometry_action,), lr=learning_rate)
    history = []
    for step in range(steps):
        action = torch.cat((geometry_action, fixed_action), -1)
        decoded = T.decode_action(action, scenes, capsule_radius=capsule_radius)
        losses = T.geometry_losses(decoded, scenes, geometry_loss=geometry_loss)
        loss = losses["geometry"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if geometry_action.grad is None or not bool(
            torch.isfinite(geometry_action.grad).all()
        ):
            raise RuntimeError("nonfinite instance-A action gradient")
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            history.append(
                {
                    "step": step + 1,
                    "objective": float(loss.detach()),
                    **{
                        name: float(value.detach().mean())
                        for name, value in losses.items()
                    },
                }
            )
    action = torch.cat((geometry_action.detach(), fixed_action), -1)
    evaluation = evaluate_instance_action(
        action,
        scenes,
        geometry_loss=geometry_loss,
        capsule_radius=capsule_radius,
    )
    variant = instance_variant_name(geometry_loss, capsule_radius)
    rate = f"{learning_rate:.0e}".replace("-", "m")
    run = output / "A" / variant / f"lr{rate}" / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase": "A",
            "seed": seed,
            "geometry_loss": geometry_loss,
            "capsule_radius": capsule_radius,
            "learning_rate": learning_rate,
            "steps": steps,
            "action": action,
            "warm_action": initial,
        },
        run / "actions.pt",
    )
    report = {
        "phase": "A",
        "seed": seed,
        "geometry_loss": geometry_loss,
        "capsule_radius": capsule_radius,
        "learning_rate": learning_rate,
        "steps": steps,
        "final_objective": evaluation["geometry"]["geometry"],
        "history": history,
        "evaluation": evaluation,
        "actions": str((run / "actions.pt").resolve()),
    }
    T.save_json(run / "summary.json", report)
    return report


def summarize_instance_a(output: Path) -> dict:
    rows = [
        json.loads(path.read_text())
        for path in sorted((output / "A").glob("*/lr*/seed*/summary.json"))
    ]
    expected = (
        len(T.GEOMETRY_LOSS_KINDS)
        * len(INSTANCE_CAPSULE_RADII)
        * len(INSTANCE_LEARNING_RATES)
        * len(INSTANCE_SEEDS)
    )
    if len(rows) != expected:
        raise ValueError(f"instance A requires {expected} runs, found {len(rows)}")
    variants = []
    for geometry_loss in T.GEOMETRY_LOSS_KINDS:
        for capsule_radius in INSTANCE_CAPSULE_RADII:
            candidates = []
            for learning_rate in INSTANCE_LEARNING_RATES:
                selected = [
                    row
                    for row in rows
                    if row["geometry_loss"] == geometry_loss
                    and row["capsule_radius"] == capsule_radius
                    and row["learning_rate"] == learning_rate
                ]
                if {row["seed"] for row in selected} != set(INSTANCE_SEEDS):
                    raise ValueError("instance A seed set is incomplete")
                candidates.append(
                    {
                        "learning_rate": learning_rate,
                        "mean_final_objective": float(
                            np.mean([row["final_objective"] for row in selected])
                        ),
                        "runs": selected,
                    }
                )
            chosen = min(candidates, key=lambda value: value["mean_final_objective"])
            aggregate = {}
            for profile in ("hinge", "relaxed-barrier"):
                solver = [
                    row["evaluation"]["solvers"][profile] for row in chosen["runs"]
                ]
                total_boxes = sum(row["v1_total_boxes"] for row in solver)
                total_lanes = sum(row["v2_total_lanes"] for row in solver)
                task_lanes = sum(row["evaluation_task_lanes"] for row in solver)
                aggregate[profile] = {
                    "v1": sum(row["v1_unsafe_boxes"] for row in solver) / total_boxes,
                    "v2": sum(row["v2_seed_collision_lanes"] for row in solver)
                    / total_lanes,
                    "v3": sum(row["v3_inside_collision_lanes"] for row in solver)
                    / total_lanes,
                    "continuous_collision": sum(
                        row["continuous_collision_lanes"] for row in solver
                    )
                    / total_lanes,
                    "failed_lanes": sum(row["failed_lanes"] for row in solver),
                    "evaluation_task": sum(
                        row["evaluation_task_sum"] or 0 for row in solver
                    )
                    / task_lanes
                    if task_lanes
                    else None,
                }
            passed = all(
                aggregate[profile][name] <= 0.01
                for profile in aggregate
                for name in ("v1", "v2", "v3")
            ) and all(aggregate[profile]["failed_lanes"] == 0 for profile in aggregate)
            variants.append(
                {
                    "geometry_loss": geometry_loss,
                    "capsule_radius": capsule_radius,
                    "learning_rate_candidates": [
                        {
                            "learning_rate": value["learning_rate"],
                            "mean_final_objective": value["mean_final_objective"],
                        }
                        for value in candidates
                    ],
                    "selected_learning_rate": chosen["learning_rate"],
                    "mean_final_objective": chosen["mean_final_objective"],
                    "fill_coverage": float(
                        np.mean(
                            [
                                row["evaluation"]["geometry"]["fill_coverage"]
                                for row in chosen["runs"]
                            ]
                        )
                    ),
                    "solvers": aggregate,
                    "passed": passed,
                }
            )
    outcome = "continue_to_b" if any(row["passed"] for row in variants) else "cp1_stop"
    report = {"phase": "A", "outcome": outcome, "variants": variants}
    T.save_json(output / "A_summary.json", report)
    return report


def plot_results(
    runs_root: Path, output: Path, *, seed=42, trajectories_only=False
) -> dict:
    """Render reproducible validation curves and actual body/gate crossings."""
    import os

    configs = {
        mode: json.loads((runs_root / f"{mode}_seed{seed}" / "config.json").read_text())
        for mode in PROFILES
    }
    first = configs[PROFILES[0]]
    if any(
        value["validation_seed"] != first["validation_seed"]
        or value["validation_size"] != first["validation_size"]
        for value in configs.values()
    ):
        raise ValueError("paired plots require the same validation scenes")

    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    colors = {"hinge": "#2563eb", "relaxed-barrier": "#d97706"}
    comparison_path = None
    if not trajectories_only:
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
        for mode, color in colors.items():
            histories = [
                json.loads(
                    (
                        runs_root / f"{mode}_seed{value}" / "training_history.json"
                    ).read_text()
                )["evaluations"]
                for value in (42, 43, 44)
            ]
            steps = [row["step"] for row in histories[0]]
            if not all(
                [row["step"] for row in history] == steps for history in histories
            ):
                raise ValueError("plot requires matching validation checkpoint steps")
            for axis, name, label in zip(
                axes,
                ("task", "gate_success_fraction", "stationarity_mean"),
                ("Task loss", "True gate success fraction", "Stationarity residual"),
                strict=True,
            ):
                values = np.asarray(
                    [[row[name] for row in history] for history in histories]
                )
                axis.plot(steps, values.mean(0), color=color, label=mode)
                axis.fill_between(
                    steps, values.min(0), values.max(0), color=color, alpha=0.16
                )
                axis.set(xlabel="Training step", ylabel=label)
                axis.grid(alpha=0.2)
        axes[0].legend()
        axes[1].set_ylim(0, 1.02)
        axes[2].set_yscale("log")
        axes[2].axhline(
            T.GRADIENT_TOLERANCE,
            color="#64748b",
            linestyle=":",
            label="Reference tolerance",
        )
        axes[2].legend(fontsize=8)
        figure.suptitle("Held-out validation: three-seed mean and min/max range")
        comparison_path = output / "training_comparison.png"
        figure.savefig(comparison_path, dpi=170)
        plt.close(figure)

    scenes = T.sample_scenes(
        first["validation_size"],
        generator=torch.Generator().manual_seed(first["validation_seed"]),
    )
    lane_count = min(4, scenes.batch_size)
    figure, grid = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    corners = torch.tensor(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)), dtype=torch.float64
    )
    for lane, axis in enumerate(grid.flat):
        if lane >= lane_count:
            axis.set_visible(False)
            continue
        for center, frame, half in zip(
            scenes.wall_centers[lane],
            scenes.wall_rotations[lane],
            scenes.wall_half_extents[lane],
            strict=True,
        ):
            axis.add_patch(
                Polygon(
                    (center + (corners * half) @ frame.T).numpy(),
                    color="#64748b",
                    alpha=0.5,
                )
            )
        for center, angle, half in zip(
            scenes.gate_centers[lane],
            scenes.gate_angles[lane],
            scenes.gate_half_widths[lane],
            strict=True,
        ):
            side = half * torch.stack((-angle.sin(), angle.cos()))
            opening = torch.stack((center - side, center + side)).numpy()
            axis.plot(opening[:, 0], opening[:, 1], color="#16a34a", linewidth=3)
        axis.scatter(
            [float(scenes.start[lane, 0]), float(scenes.goal[lane, 0])],
            [float(scenes.start[lane, 1]), float(scenes.goal[lane, 1])],
            color="#0f172a",
            zorder=5,
        )
        axis.set(
            xlim=(-0.4, 6.4),
            ylim=(-2.25, 2.25),
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"Validation scene {lane}",
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    provenance = []
    for mode, color in colors.items():
        run = runs_root / f"{mode}_seed{seed}"
        summary = json.loads((run / "summary.json").read_text())
        checkpoint = run / f"checkpoint_{summary['completed_steps']:06d}.pt"
        policy, _ = load_policy(checkpoint)
        config = configs[mode]
        setup = T.prepare_problem(
            scenes.batch_size,
            constraints=mode,
            maximum_iterations=config["maximum_iterations"],
            calibrations=tuple(
                T.Calibration(**value) for value in config["calibrations"]
            ),
        )
        with torch.no_grad():
            decoded = T.decode_action(policy(scenes.features()), scenes)
            result = setup.solver.solve(setup.bind(scenes, decoded))
            samples = T.sample_derivative(result.coefficients, result.durations, 0)
            crossings = T.gate_traversal(result, scenes)
        for lane, axis in enumerate(grid.flat):
            if lane >= lane_count:
                continue
            if int(result.status[lane]) != int(ForwardStatus.RESULT_AVAILABLE):
                axis.text(
                    0.02,
                    0.95,
                    f"{mode}: native failure",
                    transform=axis.transAxes,
                    color=color,
                )
                continue
            path = samples[lane, :, :, :2].reshape(-1, 2).numpy()
            axis.plot(path[:, 0], path[:, 1], color=color, label=mode, linewidth=1.8)
            for center, frame, half in zip(
                decoded.box_centers[lane],
                decoded.rotations[lane],
                decoded.half_extents[lane],
                strict=True,
            ):
                axis.add_patch(
                    Polygon(
                        (center + (corners * half) @ frame.T).numpy(),
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.07,
                    )
                )
            cumulative = np.cumsum(result.durations[lane].numpy())
            for when in crossings["crossing_times"][lane]:
                if when is None:
                    continue
                piece = min(int(np.searchsorted(cumulative, when)), T.PIECES - 1)
                local = when - (0 if piece == 0 else cumulative[piece - 1])
                center = np.polynomial.polynomial.polyval(
                    local, result.coefficients[lane, piece, :, :2].numpy()
                )
                axis.add_patch(
                    Circle(
                        center,
                        T.BODY_RADIUS,
                        edgecolor=color,
                        facecolor="none",
                        linestyle="--",
                        alpha=0.7,
                    )
                )
            extent = max(2.25, float(np.abs(path[:, 1]).max()) + 0.4)
            bottom, top = axis.get_ylim()
            axis.set_ylim(min(bottom, -extent), max(top, extent))
        provenance.append(
            {
                "profile": mode,
                "checkpoint": str(checkpoint),
                "validation_seed": config["validation_seed"],
                "validation_size": config["validation_size"],
                "displayed_scene_indices": list(range(lane_count)),
                "body_radius": T.BODY_RADIUS,
            }
        )
    grid.flat[0].legend()
    figure.suptitle(
        f"Seed {seed}, first held-out scenes; dashed circles show body at true plane crossings"
    )
    trajectories_path = output / "validation_trajectories.png"
    figure.savefig(trajectories_path, dpi=170)
    plt.close(figure)
    T.save_json(output / "plot_sources.json", provenance)
    return {
        "comparison": str(comparison_path.resolve())
        if comparison_path is not None
        else None,
        "trajectories": str(trajectories_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser(
        "calibrate", help="freeze development rho, outer weights and learning rate"
    )
    calibrate.add_argument("--warmup-checkpoint", type=Path, required=True)
    calibrate.add_argument("--task", choices=T.TASKS, default="gate2")
    calibrate.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/gate2/calibration")
    )
    calibrate.add_argument("--scenes", type=int, default=32)
    calibrate.add_argument("--reference-iterations", type=int, default=2048)
    calibrate.add_argument("--learning-rate-steps", type=int, default=64)
    gate = commands.add_parser(
        "gate", help="run paired stage-0 references, VJPs, finite steps and rho cliffs"
    )
    gate.add_argument("--warmup-checkpoint", type=Path, required=True)
    gate.add_argument("--trained-checkpoint", type=Path, required=True)
    gate.add_argument("--task", choices=T.TASKS, default="gate2")
    gate.add_argument("--calibration", type=Path)
    gate.add_argument("--output-dir", type=Path, default=Path("artifacts/gate2/probes"))
    gate.add_argument("--scenes", type=int, default=32)
    gate.add_argument("--reference-iterations", type=int, default=2048)
    gate.add_argument("--runs-root", type=Path)
    margin = commands.add_parser(
        "margin", help="derive the frozen hinge margin from ten barrier runs"
    )
    margin.add_argument("--runs-root", type=Path, required=True)
    margin.add_argument("--output", type=Path, required=True)
    attribution = commands.add_parser(
        "attribution", help="attribute frozen S-gate collisions to corridor validity"
    )
    attribution.add_argument("--runs-root", type=Path, required=True)
    attribution.add_argument("--output-dir", type=Path, required=True)
    pressure = commands.add_parser(
        "pressure", help="measure weighted face pressure on frozen unsafe corridors"
    )
    pressure.add_argument("--runs-root", type=Path, required=True)
    pressure.add_argument("--output-dir", type=Path, required=True)
    instance = commands.add_parser(
        "instance", help="run or summarize the predeclared per-scene validity study"
    )
    instance.add_argument(
        "--phase", choices=("warmup", "a", "summarize-a"), required=True
    )
    instance.add_argument("--geometry-loss", choices=T.GEOMETRY_LOSS_KINDS)
    instance.add_argument("--capsule-radius", type=float)
    instance.add_argument("--seed", type=int)
    instance.add_argument("--learning-rate", type=float)
    instance.add_argument("--steps", type=int)
    instance.add_argument("--output-dir", type=Path, required=True)
    plot = commands.add_parser(
        "plot", help="render paired validation curves and body footprints"
    )
    plot.add_argument("--runs-root", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.add_argument("--seed", type=int, default=42)
    plot.add_argument("--trajectories-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "calibrate":
        report = run_calibrate(
            warmup_checkpoint=args.warmup_checkpoint,
            output=args.output_dir,
            scene_count=args.scenes,
            iterations=args.reference_iterations,
            learning_rate_steps=args.learning_rate_steps,
            task=T.TASKS[args.task],
        )
        print(
            json.dumps(
                {
                    name: value
                    for name, value in report.items()
                    if name not in ("rho", "trace")
                },
                indent=2,
            )
        )
        if report["outcome"] != "calibrated":
            raise SystemExit(2)
    elif args.command == "gate":
        calibration = (
            None
            if args.calibration is None
            else json.loads(args.calibration.read_text())
        )
        calibrations = (
            None
            if calibration is None
            else tuple(T.Calibration(**value) for value in calibration["calibrations"])
        )
        weights = (
            T.task_weights(T.TASKS[args.task])
            if calibration is None
            else T.LossWeights(**calibration["weights"])
        )
        run_dirs = (
            []
            if args.runs_root is None
            else [
                args.runs_root / f"{mode}_seed{seed}"
                for seed in (42, 43, 44)
                for mode in PROFILES
            ]
        )
        report = run_gate(
            warmup_checkpoint=args.warmup_checkpoint,
            trained_checkpoint=args.trained_checkpoint,
            output=args.output_dir,
            scene_count=args.scenes,
            iterations=args.reference_iterations,
            run_dirs=run_dirs,
            task=T.TASKS[args.task],
            calibrations=calibrations,
            weights=weights,
        )
        print(
            json.dumps(
                {
                    name: report[name]
                    for name in (
                        "outcome",
                        "states",
                        "eligible_states",
                        "criterion_1",
                        "criterion_1_converged_subset",
                        "directional_summary",
                        "descent_summary",
                    )
                },
                indent=2,
            )
        )
    elif args.command == "margin":
        report = measure_barrier_margin(
            [args.runs_root / f"barrier_seed{seed}" for seed in range(42, 52)],
            output=args.output,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "attribution":
        report = run_attribution(args.runs_root, args.output_dir)
        print(
            json.dumps(
                {"profiles": report["profiles"], "figure": report["figure"]}, indent=2
            )
        )
    elif args.command == "pressure":
        report = run_pressure(args.runs_root, args.output_dir)
        print(json.dumps({"profiles": report["profiles"]}, indent=2))
    elif args.command == "instance":
        if args.phase == "summarize-a":
            report = summarize_instance_a(args.output_dir)
        else:
            if (
                args.geometry_loss is None
                or args.capsule_radius is None
                or args.seed is None
            ):
                raise ValueError("instance warmup/A requires geometry, radius and seed")
            if args.phase == "warmup":
                report = run_instance_warmup(
                    args.output_dir,
                    geometry_loss=args.geometry_loss,
                    capsule_radius=args.capsule_radius,
                    seed=args.seed,
                    steps=300 if args.steps is None else args.steps,
                )
            else:
                if args.learning_rate is None:
                    raise ValueError("instance A requires --learning-rate")
                report = run_instance_a(
                    args.output_dir,
                    geometry_loss=args.geometry_loss,
                    capsule_radius=args.capsule_radius,
                    seed=args.seed,
                    learning_rate=args.learning_rate,
                    steps=INSTANCE_A_STEPS if args.steps is None else args.steps,
                )
        print(json.dumps(report, indent=2))
    else:
        print(
            json.dumps(
                plot_results(
                    args.runs_root,
                    args.output_dir,
                    seed=args.seed,
                    trajectories_only=args.trajectories_only,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
