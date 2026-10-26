"""Aggregate the frozen three-group robust S-gate experiment.

The statistical unit is the paired training seed. This tool implements the
predeclared 10,000-draw bootstrap decision, then renders the trajectory,
clearance-CDF and learning-curve figures from the completed run artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import cpu_gate_probe as P
import cpu_gate_training as T
import numpy as np
import torch

GROUPS = ("hinge", "barrier", "hinge-margin")
SEEDS = tuple(range(42, 52))
COLORS = {
    "hinge": "#2563eb",
    "barrier": "#d97706",
    "hinge-margin": "#16a34a",
}
LABELS = {
    "hinge": "hinge",
    "barrier": "relaxed barrier",
    "hinge-margin": "hinge + 0.06 m margin",
}
EXPECTED_STEPS = 2000
EXPECTED_WARMUP_STEPS = 300
EXPECTED_BATCH_SIZE = 48
EXPECTED_VALIDATION_SIZE = 128
EXPECTED_CHECKPOINT_EVERY = 200
EXPECTED_CORRIDOR_MARGIN = 0.06


def run_path(root: Path, group: str, seed: int) -> Path:
    return root / f"{group}_seed{seed}"


def paired_bootstrap(differences, *, draws=10000, seed=20261020) -> dict:
    """Percentile CI for the mean of already-paired seed differences."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError("paired bootstrap requires at least two differences")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(draws, len(differences)))
    samples = differences[indices].mean(-1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "paired_differences": differences.tolist(),
        "mean": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "draws": draws,
        "seed": seed,
    }


def robustness_decision(collision, relative_task, *, task_tolerance=0.10) -> dict:
    """The frozen section-7 rule; negative collision differences favor barrier."""
    collision_passed = collision["ci95"][1] < 0
    task_passed = relative_task["mean"] <= task_tolerance
    return {
        "passed": collision_passed and task_passed,
        "collision_ci_upper_below_zero": collision_passed,
        "nominal_task_relative_point_within_tolerance": task_passed,
        "nominal_task_relative_tolerance": task_tolerance,
    }


def read_runs(root: Path) -> dict[str, dict[int, dict]]:
    runs = {group: {} for group in GROUPS}
    for group in GROUPS:
        for seed in SEEDS:
            path = run_path(root, group, seed)
            config = json.loads((path / "config.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            history = json.loads((path / "training_history.json").read_text())
            lanes = json.loads((path / "validation_lanes.json").read_text())
            expected_margin = (
                EXPECTED_CORRIDOR_MARGIN if group == "hinge-margin" else 0.0
            )
            expected_constraints = "relaxed-barrier" if group == "barrier" else "hinge"
            expected_config_group = "relaxed-barrier" if group == "barrier" else group
            if (
                config["task"] != "sgate3"
                or config["group"] != expected_config_group
                or config["constraints"] != expected_constraints
                or config["corridor_margin"] != expected_margin
                or config["seed"] != seed
                or config["validation_seed"] != seed + 1
                or config["steps"] != EXPECTED_STEPS
                or config["warmup_steps"] != EXPECTED_WARMUP_STEPS
                or config["batch_size"] != EXPECTED_BATCH_SIZE
                or config["validation_size"] != EXPECTED_VALIDATION_SIZE
                or config["checkpoint_every"] != EXPECTED_CHECKPOINT_EVERY
                or config["maximum_iterations"] != T.MAXIMUM_ITERATIONS
                or config["learning_rate"] != T.LEARNING_RATE
                or config["weights"] != asdict(T.SGATE3_LOSS_WEIGHTS)
                or config["loss_definition"] != asdict(T.loss_definition(T.SGATE3))
                or config["calibrations"]
                != [asdict(value) for value in T.SGATE3_CALIBRATIONS]
                or summary["outcome"] != "completed"
                or summary["completed_steps"] != EXPECTED_STEPS
                or summary["lanes"] != EXPECTED_VALIDATION_SIZE
                or summary["valid_lanes"] != EXPECTED_VALIDATION_SIZE
                or summary["perturbed"]["environments"]
                != EXPECTED_VALIDATION_SIZE * T.SGATE3.perturbations
                or summary["perturbed"]["repeats"] != T.SGATE3.perturbations
                or len(history["history"]) != EXPECTED_STEPS
                or [row["step"] for row in history["history"]]
                != list(range(1, EXPECTED_STEPS + 1))
                or any(
                    row["failure_fraction"] != 0
                    or not math.isfinite(row["gradient_norm"])
                    for row in history["history"]
                )
                or [row["step"] for row in history["evaluations"]]
                != list(
                    range(
                        0,
                        EXPECTED_STEPS + 1,
                        EXPECTED_CHECKPOINT_EVERY,
                    )
                )
            ):
                raise ValueError(f"run does not satisfy the frozen protocol: {path}")
            nominal_collision = float(np.mean(lanes["continuous_collision"]))
            perturbed_collision = float(
                np.mean(lanes["perturbed"]["continuous_collision"])
            )
            robust_gate_success = float(
                np.asarray(
                    lanes["perturbed"]["continuous_gate_success"],
                    dtype=np.bool_,
                )
                .reshape(EXPECTED_VALIDATION_SIZE, T.SGATE3.perturbations)
                .all(-1)
                .mean()
            )
            if (
                nominal_collision != summary["continuous_collision_fraction"]
                or perturbed_collision
                != summary["perturbed"]["continuous_collision_fraction"]
                or robust_gate_success
                != summary["perturbed"]["continuous_robust_gate_success_fraction"]
            ):
                raise ValueError(f"lane data does not reproduce summary: {path}")
            runs[group][seed] = {
                "path": path,
                "config": config,
                "summary": summary,
                "history": history,
                "lanes": lanes,
            }
    return runs


def verify_pairing(runs) -> None:
    for seed in SEEDS:
        states = []
        initial_generators = []
        final_generators = []
        for group in GROUPS:
            initial = torch.load(
                runs[group][seed]["path"] / "checkpoint_000000.pt",
                weights_only=True,
            )
            final = torch.load(
                runs[group][seed]["path"] / "checkpoint_002000.pt",
                weights_only=True,
            )
            states.append(initial["policy"])
            initial_generators.append(initial["scene_generator_state"])
            final_generators.append(final["scene_generator_state"])
        for state in states[1:]:
            if not all(torch.equal(states[0][name], state[name]) for name in states[0]):
                raise ValueError(f"seed {seed} does not share its initial policy")
        if not all(
            torch.equal(initial_generators[0], value)
            for value in initial_generators[1:]
        ):
            raise ValueError(f"seed {seed} does not share its initial scene RNG state")
        if not all(
            torch.equal(final_generators[0], value) for value in final_generators[1:]
        ):
            raise ValueError(f"seed {seed} does not share its final scene RNG state")


def group_statistics(runs) -> dict:
    accessors = {
        "task": lambda row: row["task"],
        "pace_error": lambda row: row["pace_error"],
        "acceleration": lambda row: row["acceleration"],
        "jerk": lambda row: row["jerk"],
        "nominal_gate_success": lambda row: row["gate_success_fraction"],
        "nominal_collision": lambda row: row["continuous_collision_fraction"],
        "nominal_clearance_p05": lambda row: row["clearance_p05"],
        "nominal_clearance_min": lambda row: row["clearance_min"],
        "perturbed_collision": lambda row: row["perturbed"][
            "continuous_collision_fraction"
        ],
        "perturbed_clearance_p05": lambda row: row["perturbed"][
            "minimum_clearance_p05"
        ],
        "perturbed_clearance_min": lambda row: row["perturbed"]["clearance_min"],
        "robust_gate_success": lambda row: row["perturbed"][
            "continuous_robust_gate_success_fraction"
        ],
        "nominal_physical_speed_violation": lambda row: row[
            "physical_speed_cap_violation_fraction"
        ],
        "nominal_physical_speed_max_violation": lambda row: row[
            "physical_speed_cap_max_violation"
        ],
        "nominal_physical_centripetal_violation": lambda row: row[
            "physical_centripetal_cap_violation_fraction"
        ],
        "nominal_physical_centripetal_max_violation": lambda row: row[
            "physical_centripetal_cap_max_violation"
        ],
        "nominal_physical_forward_acceleration_violation": lambda row: row[
            "physical_forward_acceleration_cap_violation_fraction"
        ],
        "nominal_physical_forward_acceleration_max_violation": lambda row: row[
            "physical_forward_acceleration_cap_max_violation"
        ],
        "nominal_physical_wheel_violation": lambda row: row[
            "physical_wheel_speeds_violation_fraction"
        ],
        "nominal_physical_wheel_max_violation": lambda row: row[
            "physical_wheel_speeds_max_violation"
        ],
        "perturbed_physical_speed_violation": lambda row: row["perturbed"][
            "physical_speed_violation_fraction"
        ],
        "perturbed_physical_speed_max_violation": lambda row: row["perturbed"][
            "physical_speed_max_violation"
        ],
        "perturbed_physical_wheel_violation": lambda row: row["perturbed"][
            "physical_wheel_violation_fraction"
        ],
        "perturbed_physical_wheel_max_violation": lambda row: row["perturbed"][
            "physical_wheel_max_violation"
        ],
        "corridor_active_sample_fraction": lambda row: row[
            "corridor_active_sample_fraction"
        ],
        "stationarity_mean": lambda row: row["stationarity_mean"],
        "maximum_iterations_fraction": lambda row: row["maximum_iterations_fraction"],
        "objective_evaluations_mean": lambda row: row["objective_evaluations_mean"],
    }
    report = {}
    for group in GROUPS:
        report[group] = {}
        for name, accessor in accessors.items():
            values = np.asarray(
                [accessor(runs[group][seed]["summary"]) for seed in SEEDS]
            )
            report[group][name] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return report


def comparison(runs, comparator: str) -> dict:
    barrier_collision = np.asarray(
        [
            runs["barrier"][seed]["summary"]["perturbed"][
                "continuous_collision_fraction"
            ]
            for seed in SEEDS
        ]
    )
    other_collision = np.asarray(
        [
            runs[comparator][seed]["summary"]["perturbed"][
                "continuous_collision_fraction"
            ]
            for seed in SEEDS
        ]
    )
    barrier_task = np.asarray(
        [runs["barrier"][seed]["summary"]["task"] for seed in SEEDS]
    )
    other_task = np.asarray(
        [runs[comparator][seed]["summary"]["task"] for seed in SEEDS]
    )
    collision = paired_bootstrap(barrier_collision - other_collision)
    relative_task = paired_bootstrap(barrier_task / other_task - 1)
    return {
        "definition": f"barrier minus {comparator}",
        "perturbed_continuous_collision": collision,
        "nominal_task_relative": relative_task,
        "decision": robustness_decision(collision, relative_task),
    }


def threshold_steps(runs) -> dict:
    report = {}
    for group in GROUPS:
        values = []
        for seed in SEEDS:
            reached = next(
                (
                    row["step"]
                    for row in runs[group][seed]["history"]["evaluations"]
                    if row["perturbed"]["dense_robust_gate_success_fraction"] >= 0.95
                ),
                None,
            )
            values.append(reached)
        finite = [value for value in values if value is not None]
        report[group] = {
            "per_seed": values,
            "reached_seeds": len(finite),
            "median_step_when_reached": None
            if not finite
            else float(np.median(finite)),
        }
    return report


def bootstrap_curve(values, *, draws=10000, seed=20261020):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.shape[0], size=(draws, values.shape[0]))
    samples = values[indices].mean(1)
    return values.mean(0), *np.quantile(samples, (0.025, 0.975), axis=0)


def configure_matplotlib(output: Path):
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_clearance_cdf(runs, output: Path) -> Path:
    plt = configure_matplotlib(output)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, key, title in (
        (axes[0], "dense_minimum_clearance", "Nominal environment"),
        (axes[1], "minimum_clearance", "Perturbed environments"),
    ):
        for group in GROUPS:
            values = []
            for seed in SEEDS:
                lanes = runs[group][seed]["lanes"]
                source = (
                    lanes if key == "dense_minimum_clearance" else lanes["perturbed"]
                )
                values.extend(source[key])
            values = np.sort(np.asarray(values, dtype=np.float64))
            cdf = np.arange(1, len(values) + 1) / len(values)
            axis.plot(values, cdf, color=COLORS[group], label=LABELS[group])
        axis.axvline(0, color="#64748b", linestyle=":")
        axis.set(
            xlabel="Minimum clearance (m)",
            ylabel="Empirical CDF",
            title=title,
            ylim=(0, 1.01),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    path = output / "clearance_cdf.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_collision_learning(runs, output: Path) -> Path:
    plt = configure_matplotlib(output)
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for index, group in enumerate(GROUPS):
        histories = [runs[group][seed]["history"]["evaluations"] for seed in SEEDS]
        steps = np.asarray([row["step"] for row in histories[0]])
        values = np.asarray(
            [
                [row["perturbed"]["dense_collision_fraction"] for row in history]
                for history in histories
            ]
        )
        mean, low, high = bootstrap_curve(values, seed=20261020 + index)
        axis.plot(steps, mean, color=COLORS[group], label=LABELS[group])
        axis.fill_between(steps, low, high, color=COLORS[group], alpha=0.16)
    axis.set(
        xlabel="Training step",
        ylabel="Perturbed dense collision fraction",
        title="Robust collision during training (seed mean and 95% bootstrap CI)",
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    path = output / "robust_collision_learning.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def add_walls(axis, scenes, lane, *, perturbed=False):
    from matplotlib.patches import Polygon

    corners = torch.tensor(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
        dtype=torch.float64,
    )
    for center, frame, half in zip(
        scenes.wall_centers[lane],
        scenes.wall_rotations[lane],
        scenes.wall_half_extents[lane],
        strict=True,
    ):
        axis.add_patch(
            Polygon(
                (center + (corners * half) @ frame.T).numpy(),
                facecolor="none" if perturbed else "#64748b",
                edgecolor="#dc2626" if perturbed else "#64748b",
                linestyle="--" if perturbed else "-",
                alpha=0.65 if perturbed else 0.30,
                linewidth=1.0,
            )
        )


def add_openings(axis, scenes, lane, *, perturbed=False):
    for center, angle, half_width in zip(
        scenes.gate_centers[lane],
        scenes.gate_angles[lane],
        scenes.gate_half_widths[lane],
        strict=True,
    ):
        side = half_width * torch.stack((-angle.sin(), angle.cos()))
        opening = torch.stack((center - side, center + side)).numpy()
        axis.plot(
            opening[:, 0],
            opening[:, 1],
            color="#dc2626" if perturbed else "#16a34a",
            linestyle="--" if perturbed else "-",
            linewidth=2 if perturbed else 3,
            alpha=0.8,
        )


def plot_trajectories(runs, output: Path) -> tuple[Path, list]:
    plt = configure_matplotlib(output)
    from matplotlib.patches import Circle

    seed = 42
    config = runs["barrier"][seed]["config"]
    scenes = T.sample_scenes(
        config["validation_size"],
        generator=torch.Generator().manual_seed(config["validation_seed"]),
        task=T.SGATE3,
    )
    perturbations = T.sample_perturbations(
        scenes,
        generator=torch.Generator().manual_seed(config["validation_seed"] + 10000),
        repeats=T.SGATE3.perturbations,
    )
    lane = 0
    perturbed_lane = 0
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    provenance = []
    for axis, group in zip(axes, GROUPS, strict=True):
        entry = runs[group][seed]
        config = entry["config"]
        checkpoint = entry["path"] / "checkpoint_002000.pt"
        policy, _ = P.load_policy(checkpoint, T.SGATE3)
        setup = T.prepare_problem(
            scenes.batch_size,
            constraints=config["constraints"],
            maximum_iterations=config["maximum_iterations"],
            calibrations=tuple(
                T.Calibration(**value) for value in config["calibrations"]
            ),
            task=T.SGATE3,
            corridor_margin=config["corridor_margin"],
        )
        with torch.no_grad():
            decoded = T.decode_action(policy(scenes.features()), scenes)
            result = setup.solver.solve(setup.bind(scenes, decoded))
            samples = T.sample_derivative(result.coefficients, result.durations, 0)[
                lane, ..., :2
            ].reshape(-1, 2)
        add_walls(axis, scenes, lane)
        add_walls(axis, perturbations.scenes, perturbed_lane, perturbed=True)
        add_openings(axis, scenes, lane)
        add_openings(axis, perturbations.scenes, perturbed_lane, perturbed=True)
        axis.plot(
            samples[:, 0],
            samples[:, 1],
            color=COLORS[group],
            linewidth=2,
            label=LABELS[group],
        )
        for point in samples[:: max(1, len(samples) // 14)]:
            axis.add_patch(
                Circle(
                    point.numpy(),
                    T.BODY_RADIUS,
                    facecolor=COLORS[group],
                    edgecolor="none",
                    alpha=0.055,
                )
            )
        axis.scatter(
            [float(scenes.start[lane, 0]), float(scenes.goal[lane, 0])],
            [float(scenes.start[lane, 1]), float(scenes.goal[lane, 1])],
            color="#0f172a",
            zorder=5,
        )
        axis.set(
            xlim=(-0.5, 9.5),
            ylim=(-2.25, 2.25),
            xlabel="x (m)",
            ylabel="y (m)",
            title=LABELS[group],
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
        summary = entry["summary"]
        axis.text(
            0.02,
            0.98,
            f"seed-42 nominal collision fraction={summary['continuous_collision_fraction']:.3f}\n"
            f"seed-42 perturbed collision fraction={summary['perturbed']['continuous_collision_fraction']:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        provenance.append(
            {
                "group": group,
                "checkpoint": str(checkpoint),
                "validation_seed": config["validation_seed"],
                "scene": lane,
                "perturbation": 0,
                "forward_status": int(result.status[lane]),
            }
        )
    figure.suptitle(
        "Seed 42, held-out scene 0; gray walls are nominal, red dashed walls are perturbation 0"
    )
    path = output / "validation_trajectories.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path, provenance


def build_report(runs_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runs = read_runs(runs_root)
    verify_pairing(runs)
    barrier_hinge = comparison(runs, "hinge")
    barrier_margin = comparison(runs, "hinge-margin")
    trajectories, provenance = plot_trajectories(runs, output)
    cdf = plot_clearance_cdf(runs, output)
    learning = plot_collision_learning(runs, output)
    report = {
        "protocol_complete": True,
        "seeds": list(SEEDS),
        "bootstrap_draws": 10000,
        "paired_initial_policy_and_scene_rng_verified": True,
        "paired_final_scene_rng_verified": True,
        "groups": group_statistics(runs),
        "barrier_vs_hinge": barrier_hinge,
        "barrier_vs_hinge_margin": barrier_margin,
        "primary_barrier_more_robust": barrier_hinge["decision"]["passed"],
        "learning_signal_beyond_forward_margin": barrier_margin["decision"]["passed"],
        "robust_gate_95_percent_steps": threshold_steps(runs),
        "figures": {
            "trajectories": str(trajectories.resolve()),
            "clearance_cdf": str(cdf.resolve()),
            "robust_collision_learning": str(learning.resolve()),
        },
        "trajectory_sources": provenance,
    }
    T.save_json(output / "robust_comparison.json", report)
    T.save_json(output / "plot_sources.json", provenance)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = build_report(args.runs_root, args.output_dir)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol_complete",
                    "barrier_vs_hinge",
                    "barrier_vs_hinge_margin",
                    "primary_barrier_more_robust",
                    "learning_signal_beyond_forward_margin",
                    "robust_gate_95_percent_steps",
                    "figures",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
