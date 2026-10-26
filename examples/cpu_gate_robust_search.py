"""Analyze longer and counterfactual robust S-gate exploratory runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cpu_gate_probe as P
import cpu_gate_training as T
import numpy as np
import torch

GROUPS = ("hinge", "barrier", "hinge-margin")
PILOT_SEEDS = (42, 46, 50)
SEVERITY_FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0)
COLORS = {
    "hinge": "#2563eb",
    "barrier": "#d97706",
    "hinge-margin": "#16a34a",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def group_solver(group: str) -> tuple[str, float]:
    if group == "barrier":
        return "relaxed-barrier", 0.0
    if group == "hinge-margin":
        return "hinge", 0.06
    if group == "hinge":
        return "hinge", 0.0
    raise ValueError(f"unknown group: {group}")


def run_path(root: Path, group: str, seed: int) -> Path:
    return root / f"{group}_seed{seed}"


def checkpoint_path(root: Path, group: str, seed: int, step: int) -> Path:
    return run_path(root, group, seed) / f"checkpoint_{step:06d}.pt"


def load_solution(run: Path, checkpoint: Path, evaluation_group: str):
    config = load_json(run / "config.json")
    policy, payload = P.load_policy(checkpoint, T.SGATE3)
    scenes = T.sample_scenes(
        config["validation_size"],
        generator=torch.Generator().manual_seed(config["validation_seed"]),
        task=T.SGATE3,
    )
    constraints, margin = group_solver(evaluation_group)
    setup = T.prepare_problem(
        scenes.batch_size,
        constraints=constraints,
        maximum_iterations=config["maximum_iterations"],
        calibrations=tuple(T.Calibration(**value) for value in config["calibrations"]),
        task=T.SGATE3,
        corridor_margin=margin,
    )
    with torch.no_grad():
        decoded = T.decode_action(policy(scenes.features()), scenes)
        result = setup.solver.solve(setup.bind(scenes, decoded))
        losses = T.trajectory_losses(
            result,
            decoded,
            scenes,
            weights=T.LossWeights(**config["weights"]),
        )
    return config, payload, scenes, result, losses


def perturbation_metrics(result, scenes, *, seed: int, scale: float) -> dict:
    perturbations = T.sample_perturbations(
        scenes,
        generator=torch.Generator().manual_seed(seed + 10001),
        repeats=T.SGATE3.perturbations,
        scale=scale,
    )
    summary, _ = T.evaluate_perturbations(result, perturbations, final=True)
    return summary


def configure_matplotlib(output: Path):
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def severity_sweep(runs_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in GROUPS:
        for seed in range(42, 52):
            run = run_path(runs_root, group, seed)
            frozen = load_json(run / "summary.json")
            _, _, scenes, result, _ = load_solution(
                run,
                checkpoint_path(runs_root, group, seed, 2000),
                group,
            )
            for scale in SEVERITY_FACTORS:
                metrics = perturbation_metrics(result, scenes, seed=seed, scale=scale)
                row = {
                    "group": group,
                    "seed": seed,
                    "scale": scale,
                    "continuous_collision_fraction": metrics[
                        "continuous_collision_fraction"
                    ],
                    "continuous_gate_success_fraction": metrics[
                        "continuous_gate_success_fraction"
                    ],
                    "continuous_robust_gate_success_fraction": metrics[
                        "continuous_robust_gate_success_fraction"
                    ],
                    "minimum_clearance_p05": metrics["minimum_clearance_p05"],
                    "clearance_min": metrics["clearance_min"],
                }
                if (
                    scale == 0
                    and row["continuous_collision_fraction"]
                    != frozen["continuous_collision_fraction"]
                ):
                    raise RuntimeError(f"scale 0 does not reproduce nominal: {run}")
                if (
                    scale == 1
                    and row["continuous_collision_fraction"]
                    != frozen["perturbed"]["continuous_collision_fraction"]
                ):
                    raise RuntimeError(f"scale 1 does not reproduce frozen: {run}")
                rows.append(row)

    groups = {}
    for group in GROUPS:
        groups[group] = {}
        for scale in SEVERITY_FACTORS:
            selected = [
                row for row in rows if row["group"] == group and row["scale"] == scale
            ]
            groups[group][str(scale)] = {
                name: {
                    "mean": float(np.mean([row[name] for row in selected])),
                    "sample_sd": float(np.std([row[name] for row in selected], ddof=1)),
                }
                for name in (
                    "continuous_collision_fraction",
                    "continuous_gate_success_fraction",
                    "continuous_robust_gate_success_fraction",
                    "minimum_clearance_p05",
                )
            }

    plt = configure_matplotlib(output)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for group in GROUPS:
        collision = [
            groups[group][str(scale)]["continuous_collision_fraction"]["mean"]
            for scale in SEVERITY_FACTORS
        ]
        clearance = [
            groups[group][str(scale)]["minimum_clearance_p05"]["mean"]
            for scale in SEVERITY_FACTORS
        ]
        axes[0].plot(
            SEVERITY_FACTORS,
            collision,
            marker="o",
            color=COLORS[group],
            label=group,
        )
        axes[1].plot(
            SEVERITY_FACTORS,
            clearance,
            marker="o",
            color=COLORS[group],
            label=group,
        )
    axes[0].set(
        xlabel="Perturbation severity factor",
        ylabel="Continuous collision fraction",
        ylim=(0, 1),
    )
    axes[1].set(
        xlabel="Perturbation severity factor",
        ylabel="Minimum clearance p05 (m)",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Frozen step-2000 policies across perturbation severity")
    figure_path = output / "severity_sweep.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report = {
        "factors": list(SEVERITY_FACTORS),
        "endpoint_replay": True,
        "groups": groups,
        "rows": rows,
        "figure": str(figure_path.resolve()),
    }
    T.save_json(output / "severity_sweep.json", report)
    return report


def long_summary(formal_root: Path, long_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    metrics = (
        "task",
        "continuous_collision_fraction",
        "clearance_p05",
    )
    rows = []
    for group in GROUPS:
        for seed in PILOT_SEEDS:
            frozen = load_json(run_path(formal_root, group, seed) / "summary.json")
            extended = load_json(run_path(long_root, group, seed) / "summary.json")
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "step_2000": {
                        **{name: frozen[name] for name in metrics},
                        "perturbed_continuous_collision_fraction": frozen["perturbed"][
                            "continuous_collision_fraction"
                        ],
                    },
                    "step_5000": {
                        **{name: extended[name] for name in metrics},
                        "perturbed_continuous_collision_fraction": extended[
                            "perturbed"
                        ]["continuous_collision_fraction"],
                    },
                }
            )

    comparisons = {}
    for comparator in ("hinge", "hinge-margin"):
        by_budget = {}
        for budget in (2000, 5000):
            key = f"step_{budget}"
            differences = []
            for seed in PILOT_SEEDS:
                barrier = next(
                    row
                    for row in rows
                    if row["group"] == "barrier" and row["seed"] == seed
                )[key]["perturbed_continuous_collision_fraction"]
                other = next(
                    row
                    for row in rows
                    if row["group"] == comparator and row["seed"] == seed
                )[key]["perturbed_continuous_collision_fraction"]
                differences.append(barrier - other)
            by_budget[str(budget)] = {
                "paired_differences": differences,
                "mean": float(np.mean(differences)),
            }
        improvements = [
            end - start
            for start, end in zip(
                by_budget["2000"]["paired_differences"],
                by_budget["5000"]["paired_differences"],
                strict=True,
            )
        ]
        by_budget["differential_improvement"] = {
            "paired_values": improvements,
            "mean": float(np.mean(improvements)),
            "favoring_barrier": sum(value < 0 for value in improvements),
            "expansion_rule_passed": float(np.mean(improvements)) <= -0.01
            and sum(value < 0 for value in improvements) >= 2,
        }
        comparisons[comparator] = by_budget
    report = {
        "seeds": list(PILOT_SEEDS),
        "comparisons": comparisons,
        "expand_remaining_seeds": any(
            value["differential_improvement"]["expansion_rule_passed"]
            for value in comparisons.values()
        ),
        "rows": rows,
    }
    T.save_json(output / "long_summary.json", report)
    return report


def cross_evaluation(formal_root: Path, long_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget, root in ((2000, formal_root), (5000, long_root)):
        for seed in PILOT_SEEDS:
            for training_group in GROUPS:
                run = run_path(root, training_group, seed)
                checkpoint = checkpoint_path(root, training_group, seed, budget)
                for evaluation_group in GROUPS:
                    _, _, scenes, result, losses = load_solution(
                        run, checkpoint, evaluation_group
                    )
                    valid = T.valid_lanes(result, losses)
                    perturbed = perturbation_metrics(
                        result, scenes, seed=seed, scale=1.0
                    )
                    rows.append(
                        {
                            "budget": budget,
                            "seed": seed,
                            "training_group": training_group,
                            "evaluation_group": evaluation_group,
                            "valid_fraction": float(valid.double().mean()),
                            "task": float(losses["task"][valid].mean()),
                            "perturbed_continuous_collision_fraction": perturbed[
                                "continuous_collision_fraction"
                            ],
                            "perturbed_clearance_p05": perturbed[
                                "minimum_clearance_p05"
                            ],
                        }
                    )

    matrices = {}
    for budget in (2000, 5000):
        matrix = []
        for training_group in GROUPS:
            line = []
            for evaluation_group in GROUPS:
                selected = [
                    row
                    for row in rows
                    if row["budget"] == budget
                    and row["training_group"] == training_group
                    and row["evaluation_group"] == evaluation_group
                ]
                line.append(
                    float(
                        np.mean(
                            [
                                row["perturbed_continuous_collision_fraction"]
                                for row in selected
                            ]
                        )
                    )
                )
            matrix.append(line)
        matrices[str(budget)] = matrix

    plt = configure_matplotlib(output)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, budget in zip(axes, (2000, 5000), strict=True):
        image = axis.imshow(matrices[str(budget)], vmin=0.6, vmax=0.9, cmap="magma")
        axis.set_xticks(range(3), GROUPS, rotation=25, ha="right")
        axis.set_yticks(range(3), GROUPS)
        axis.set(
            xlabel="Evaluation solver",
            ylabel="Training profile",
            title=f"step {budget}",
        )
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{matrices[str(budget)][row][column]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axes, label="Perturbed continuous collision fraction")
    figure_path = output / "cross_evaluation.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report = {
        "seeds": list(PILOT_SEEDS),
        "matrices": matrices,
        "rows": rows,
        "figure": str(figure_path.resolve()),
    }
    T.save_json(output / "cross_evaluation.json", report)
    return report


def run_clearance_training(
    *,
    group: str,
    seed: int,
    warmup_checkpoint: Path,
    output: Path,
    steps: int = 3000,
) -> dict:
    constraints, margin = group_solver(group)
    return T.run_training(
        constraints=constraints,
        steps=steps,
        warmup_steps=300,
        batch_size=48,
        seed=seed,
        validation_size=128,
        checkpoint_every=200,
        maximum_iterations=T.MAXIMUM_ITERATIONS,
        learning_rate=T.LEARNING_RATE,
        weights=T.SGATE3_LOSS_WEIGHTS,
        train_clearance=True,
        calibrations=T.SGATE3_CALIBRATIONS,
        task=T.SGATE3,
        corridor_margin=margin,
        output_dir=output,
        warmup_checkpoint=warmup_checkpoint,
    )


def clearance_summary(formal_root: Path, clearance_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    initial_states: dict[int, list[dict]] = {seed: [] for seed in PILOT_SEEDS}
    final_generators: dict[int, list[torch.Tensor]] = {seed: [] for seed in PILOT_SEEDS}
    for group in GROUPS:
        for seed in PILOT_SEEDS:
            frozen_run = run_path(formal_root, group, seed)
            aligned_run = run_path(clearance_root, group, seed)
            frozen = load_json(frozen_run / "summary.json")
            frozen_config = load_json(frozen_run / "config.json")
            aligned = load_json(aligned_run / "summary.json")
            config = load_json(aligned_run / "config.json")
            training = load_json(aligned_run / "training_history.json")
            history = training["history"]
            expected_config = {
                **frozen_config,
                "steps": 3000,
                "loss_definition": {
                    **frozen_config["loss_definition"],
                    "train_clearance": True,
                },
            }
            if (
                config != expected_config
                or aligned["outcome"] != "completed"
                or aligned["completed_steps"] != 3000
                or aligned["failure_fraction"] != 0
                or aligned["valid_lanes"] != 128
                or [row["step"] for row in history] != list(range(1, 3001))
                or [row["step"] for row in training["evaluations"]]
                != list(range(0, 3001, 200))
                or any(
                    row["failure_fraction"] != 0
                    or not np.isfinite(row["gradient_norm"])
                    for row in history
                )
            ):
                raise ValueError(f"invalid clearance-aligned run: {aligned_run}")
            initial = torch.load(
                aligned_run / "checkpoint_000000.pt", weights_only=True
            )
            frozen_initial = torch.load(
                frozen_run / "checkpoint_000000.pt", weights_only=True
            )
            final = torch.load(aligned_run / "checkpoint_003000.pt", weights_only=True)
            if not all(
                torch.equal(initial["policy"][name], frozen_initial["policy"][name])
                for name in initial["policy"]
            ):
                raise ValueError(
                    f"initial policy does not match frozen run: {aligned_run}"
                )
            initial_states[seed].append(initial["policy"])
            final_generators[seed].append(final["scene_generator_state"])
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "frozen": {
                        "nominal_collision": frozen["continuous_collision_fraction"],
                        "perturbed_collision": frozen["perturbed"][
                            "continuous_collision_fraction"
                        ],
                        "perturbed_clearance_p05": frozen["perturbed"][
                            "minimum_clearance_p05"
                        ],
                        "robust_gate_success": frozen["perturbed"][
                            "continuous_robust_gate_success_fraction"
                        ],
                    },
                    "clearance_aligned": {
                        "nominal_collision": aligned["continuous_collision_fraction"],
                        "perturbed_collision": aligned["perturbed"][
                            "continuous_collision_fraction"
                        ],
                        "perturbed_clearance_p05": aligned["perturbed"][
                            "minimum_clearance_p05"
                        ],
                        "robust_gate_success": aligned["perturbed"][
                            "continuous_robust_gate_success_fraction"
                        ],
                    },
                }
            )
    for seed in PILOT_SEEDS:
        reference = initial_states[seed][0]
        for state in initial_states[seed][1:]:
            if not all(torch.equal(reference[name], state[name]) for name in reference):
                raise ValueError(f"seed {seed} initial policies differ")
        if not all(
            torch.equal(final_generators[seed][0], state)
            for state in final_generators[seed][1:]
        ):
            raise ValueError(f"seed {seed} final scene RNG states differ")

    metric_names = (
        "nominal_collision",
        "perturbed_collision",
        "perturbed_clearance_p05",
        "robust_gate_success",
    )
    groups = {}
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {}
        for metric in metric_names:
            frozen_values = np.asarray([row["frozen"][metric] for row in selected])
            aligned_values = np.asarray(
                [row["clearance_aligned"][metric] for row in selected]
            )
            groups[group][metric] = {
                "frozen_mean": float(frozen_values.mean()),
                "clearance_aligned_mean": float(aligned_values.mean()),
                "paired_changes": (aligned_values - frozen_values).tolist(),
                "paired_change_mean": float((aligned_values - frozen_values).mean()),
            }
    comparisons = {}
    for comparator in ("hinge", "hinge-margin"):
        differences = []
        for seed in PILOT_SEEDS:
            barrier = next(
                row for row in rows if row["group"] == "barrier" and row["seed"] == seed
            )["clearance_aligned"]["perturbed_collision"]
            other = next(
                row
                for row in rows
                if row["group"] == comparator and row["seed"] == seed
            )["clearance_aligned"]["perturbed_collision"]
            differences.append(barrier - other)
        comparisons[comparator] = {
            "paired_differences": differences,
            "mean": float(np.mean(differences)),
        }

    plt = configure_matplotlib(output)
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    x = np.arange(len(GROUPS))
    width = 0.36
    for axis, metric, title in (
        (axes[0], "nominal_collision", "Nominal collision"),
        (axes[1], "perturbed_collision", "Perturbed collision"),
        (axes[2], "robust_gate_success", "Robust gate success"),
    ):
        frozen = [groups[group][metric]["frozen_mean"] for group in GROUPS]
        aligned = [groups[group][metric]["clearance_aligned_mean"] for group in GROUPS]
        axis.bar(x - width / 2, frozen, width, label="frozen objective")
        axis.bar(x + width / 2, aligned, width, label="+ clearance loss")
        axis.set_xticks(x, GROUPS, rotation=25, ha="right")
        axis.set(title=title, ylim=(0, 1))
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Three-seed loss-alignment pilot")
    figure_path = output / "clearance_alignment.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report = {
        "seeds": list(PILOT_SEEDS),
        "pairing_verified": True,
        "groups": groups,
        "comparisons": comparisons,
        "rows": rows,
        "figure": str(figure_path.resolve()),
    }
    T.save_json(output / "clearance_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    severity = commands.add_parser("severity")
    severity.add_argument("--runs-root", type=Path, required=True)
    severity.add_argument("--output-dir", type=Path, required=True)
    long = commands.add_parser("long")
    long.add_argument("--formal-root", type=Path, required=True)
    long.add_argument("--long-root", type=Path, required=True)
    long.add_argument("--output-dir", type=Path, required=True)
    cross = commands.add_parser("cross")
    cross.add_argument("--formal-root", type=Path, required=True)
    cross.add_argument("--long-root", type=Path, required=True)
    cross.add_argument("--output-dir", type=Path, required=True)
    train_clearance = commands.add_parser("train-clearance")
    train_clearance.add_argument("--group", choices=GROUPS, required=True)
    train_clearance.add_argument("--seed", type=int, required=True)
    train_clearance.add_argument("--warmup-checkpoint", type=Path, required=True)
    train_clearance.add_argument("--output-dir", type=Path, required=True)
    train_clearance.add_argument("--steps", type=int, default=3000)
    clearance = commands.add_parser("clearance")
    clearance.add_argument("--formal-root", type=Path, required=True)
    clearance.add_argument("--clearance-root", type=Path, required=True)
    clearance.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "severity":
        report = severity_sweep(args.runs_root, args.output_dir)
        printable = {
            "endpoint_replay": report["endpoint_replay"],
            "groups": report["groups"],
        }
    elif args.command == "long":
        report = long_summary(args.formal_root, args.long_root, args.output_dir)
        printable = {
            "comparisons": report["comparisons"],
            "expand_remaining_seeds": report["expand_remaining_seeds"],
        }
    elif args.command == "cross":
        report = cross_evaluation(args.formal_root, args.long_root, args.output_dir)
        printable = {"matrices": report["matrices"], "figure": report["figure"]}
    elif args.command == "train-clearance":
        report = run_clearance_training(
            group=args.group,
            seed=args.seed,
            warmup_checkpoint=args.warmup_checkpoint,
            output=args.output_dir,
            steps=args.steps,
        )
        printable = report
    else:
        report = clearance_summary(
            args.formal_root,
            args.clearance_root,
            args.output_dir,
        )
        printable = {
            "groups": report["groups"],
            "comparisons": report["comparisons"],
            "figure": report["figure"],
        }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
