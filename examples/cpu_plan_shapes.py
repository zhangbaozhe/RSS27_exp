"""Resolve static or dynamic trajectory declarations into reusable CPU Plans.

Run: python examples/cpu_plan_shapes.py
Works with full and NumPy light presets. Every Plan has concrete dimensions and
fixed slots before preparation; changed numerical values can reuse that layout.
"""

import argparse

import numpy as np
from constraints import Calibration, add_constraint_argument, term

from pivopt import ForwardStatus, OptimizerStatus
from pivopt import numpy as cpu
from pivopt import problem as p

DECLARATIONS = (
    ("fixed K and Q", 4, 16),
    ("K supplied by PlanShape", p.DYNAMIC, 16),
    ("Q supplied by PlanShape", 4, p.DYNAMIC),
    ("K and Q supplied by PlanShape", p.DYNAMIC, p.DYNAMIC),
)


def make_spec(pieces=p.DYNAMIC, intervals=p.DYNAMIC, *, constraints="hinge"):
    trajectory = p.MincoSpec(
        p.Space(position=3, yaw=True),
        derivative_order=3,
        pieces=pieces,
        head_state=p.Input("head"),
        tail_state=p.Input("tail"),
        time=p.FixedTotalTime(total_duration=4, minimum_piece_duration=0.1),
        waypoints=p.DirectWaypoints(),
    )
    quadrotor = p.Quadrotor()
    return p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=intervals),
        objective=(
            p.Energy("energy", weights={"position": 1.0, "yaw": 0.5}),
            term(
                p.Integral,
                "corridor",
                lambda profile: p.Inside(
                    trajectory.position, p.Input("corridor"), profile=profile
                ),
                Calibration(
                    hinge_weight=10,
                    width=0.1,
                    barrier_weight=0.22,
                    delta=0.022,
                    activation=1.0,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "speed",
                lambda profile: p.NormBelow(trajectory.velocity, 1.5, profile=profile),
                Calibration(
                    hinge_weight=1,
                    width=0.1,
                    barrier_weight=0.022,
                    delta=0.022,
                    activation=0.15,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "acceleration",
                lambda profile: p.NormBelow(
                    trajectory.acceleration, 0, profile=profile
                ),
                Calibration(
                    hinge_weight=0,
                    width=0.1,
                    barrier_weight=0.0,
                    delta=0.022,
                    activation=0.1,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "jerk",
                lambda profile: p.NormBelow(trajectory.jerk, 0, profile=profile),
                Calibration(
                    hinge_weight=0,
                    width=0.1,
                    barrier_weight=0.0,
                    delta=0.022,
                    activation=0.1,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "body_rate",
                lambda profile: p.NormBelow(quadrotor.body_rate, 0, profile=profile),
                Calibration(
                    hinge_weight=0,
                    width=0.1,
                    barrier_weight=0.0,
                    delta=0.022,
                    activation=0.1,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "tilt",
                lambda profile: p.TiltBelow(quadrotor.attitude, 0, profile=profile),
                Calibration(
                    hinge_weight=0,
                    width=0.1,
                    barrier_weight=0.0,
                    delta=0.022,
                    activation=0.1,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "thrust",
                lambda profile: p.Within(quadrotor.thrust, 0, 0, profile=profile),
                Calibration(
                    hinge_weight=0,
                    width=0.1,
                    barrier_weight=0.0,
                    delta=0.022,
                    activation=0.1,
                ),
                constraints,
            ),
        ),
        forward=p.BFGS(initial_guess=p.Input("initial")),
        backward=p.ForwardOnly(),
    )


def make_data(pieces, faces):
    """Fixed endpoints/time, varying discretization; every array retains B=1."""
    head = np.zeros((1, 4, 3), dtype=np.float64)
    tail = np.zeros_like(head)
    tail[0, 0, 0], tail[0, 3, 0] = 4, 1
    fraction = np.arange(1, pieces, dtype=np.float64) / pieces
    waypoints = np.zeros((pieces - 1, 4), dtype=np.float64)
    waypoints[:, 0] = 4 * fraction
    waypoints[:, 1] = 0.2 * np.sin(np.pi * fraction)
    waypoints[:, 3] = fraction
    halfspaces = np.zeros((1, pieces, faces, 4), dtype=np.float64)
    for face in range(faces):
        if face < 6:
            halfspaces[..., face, face // 2] = -1 if face % 2 else 1
            halfspaces[..., face, 3] = -10
        else:
            halfspaces[..., face, 0] = 1
            halfspaces[..., face, 3] = -10 - face
    return dict(
        head=head,
        tail=tail,
        corridor=halfspaces,
        initial=np.concatenate([np.zeros(pieces - 1), waypoints.ravel()])[None, :],
    )


def build_case(spec, data, intervals):
    """Resolve names/shapes once; no numerical arrays are retained by the Plan."""
    plan = p.build_plan(
        spec,
        p.PlanShape(
            batch_size=1,
            piece_count=data["corridor"].shape[1]
            if spec.trajectory.pieces == p.DYNAMIC
            else None,
            quadrature_intervals=intervals
            if spec.sampling.intervals_per_piece == p.DYNAMIC
            else None,
            input_shapes=(
                p.InputShape(p.InputRef("corridor"), data["corridor"].shape),
            ),
        ),
    )
    inputs = plan.inputs(**data)
    return cpu.build_solver(plan, dtype=np.float64), inputs


def solve_and_print(label, solver, inputs):
    result = solver.solve(inputs)
    status = ForwardStatus(int(result.status[0]))
    diagnostics = result.diagnostics
    print(f"\n{label}")
    print(
        f"  decision shape: {result.x.shape}; coefficients: {result.coefficients.shape}"
    )
    print(
        f"  forward={status.name}, optimizer={OptimizerStatus(int(diagnostics.optimizer_status[0])).name}, iterations={int(diagnostics.iterations[0])}"
    )
    if status != ForwardStatus.RESULT_AVAILABLE:
        raise RuntimeError(f"solve failed for {label}: {status.name}")
    print(
        f"  cost={float(result.cost[0]):.9g}, stationarity={float(diagnostics.stationarity[0]):.3e}"
    )
    print("  durations:", result.durations)
    print("  waypoints [x, y, z, yaw]:\n", result.waypoints)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_constraint_argument(parser)
    args = parser.parse_args()
    np.set_printoptions(precision=5, suppress=True)
    print("Same problem: K=4, Q=16, Fc=6; compare declaration shapes.")
    print("Corridor face counts are fixed by each Plan's InputShape.")
    data = make_data(pieces=4, faces=6)
    reference = None
    for label, pieces, quadrature in DECLARATIONS:
        solver, inputs = build_case(
            make_spec(pieces, quadrature, constraints=args.constraints),
            data,
            intervals=16,
        )
        result = solve_and_print(label, solver, inputs)
        if reference is None:
            reference = result
        else:
            for name in p.ResultSeeds._fields:
                delta = np.max(np.abs(getattr(result, name) - getattr(reference, name)))
                print(f"  {name}: max absolute difference {float(delta):.3e}")
    print("\nReuse one declaration with new K/Q/Fc; each shape gets a new Plan.")
    spec = make_spec(constraints=args.constraints)
    for pieces, intervals, faces in ((8, 32, 16), (12, 24, 10)):
        solver, inputs = build_case(spec, make_data(pieces, faces), intervals)
        solve_and_print(f"K={pieces}, Q={intervals}, Fc={faces}", solver, inputs)


if __name__ == "__main__":
    main()
