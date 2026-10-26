"""Plan a smooth xy+yaw trajectory for an omnidirectional planar base."""

import argparse

import numpy as np
from constraints import Calibration, add_constraint_argument, term

from pivopt import ForwardStatus
from pivopt import numpy as cpu
from pivopt import problem as p


def make_problem(*, constraints="hinge"):
    pieces = 6
    trajectory = p.MincoSpec(
        p.Space(position=2, yaw=True),
        derivative_order=3,
        pieces=pieces,
        head_state=p.Input("head"),
        tail_state=p.Input("tail"),
        time=p.FixedTotalTime(total_duration=6.0, minimum_piece_duration=0.1),
        waypoints=p.DirectWaypoints(),
    )
    corridor = p.Input("corridor")
    spec = p.ProblemSpec(
        trajectory=trajectory,
        objective=(
            p.Energy("smooth", weights={"position": 1.0, "yaw": 0.3}),
            term(
                p.Integral,
                "speed",
                lambda profile: p.NormBelow(trajectory.velocity, 1.5, profile=profile),
                Calibration(
                    hinge_weight=1000.0,
                    width=0.05,
                    barrier_weight=11.0,
                    delta=0.011,
                    activation=0.15,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "turn_rate",
                lambda profile: p.NormBelow(trajectory.yaw.d(1), 2.0, profile=profile),
                Calibration(
                    hinge_weight=1000.0,
                    width=0.05,
                    barrier_weight=11.0,
                    delta=0.011,
                    activation=0.2,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "corridor",
                lambda profile: p.Inside(
                    trajectory.position, corridor, profile=profile
                ),
                Calibration(
                    hinge_weight=10000.0,
                    width=0.1,
                    barrier_weight=220.0,
                    delta=0.022,
                    activation=0.2,
                ),
                constraints,
            ),
        ),
        sampling=p.Trapezoidal(intervals_per_piece=16),
        forward=p.LBFGS(initial_guess=p.Input("initial")),
        backward=p.ForwardOnly(),
    )
    plan = p.build_plan(
        spec,
        p.PlanShape(
            batch_size=1,
            input_shapes=(p.InputShape(corridor, (1, pieces, 4, 3)),),
        ),
    )

    head = np.zeros((1, 3, 3), dtype=np.float64)
    tail = np.zeros_like(head)
    tail[0, :, 0] = (4.0, 1.0, 0.6)
    waypoints = np.stack(
        [tail[0, :, 0] * (index / pieces) for index in range(1, pieces)],
        axis=0,
    )[None]
    durations = np.ones((1, pieces), dtype=np.float64)
    rows = np.array(
        [[1.0, 0.0, -5.0], [-1.0, 0.0, -1.0], [0.0, 1.0, -2.0], [0.0, -1.0, -2.0]],
        dtype=np.float64,
    )
    rows = np.broadcast_to(rows, (1, pieces, 4, 3)).copy()
    inputs = plan.inputs(
        head=head,
        tail=tail,
        corridor=rows,
        initial=plan.initial_guess(waypoints=waypoints, durations=durations),
    )
    return plan, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_constraint_argument(parser)
    args = parser.parse_args()
    plan, inputs = make_problem(constraints=args.constraints)
    result = cpu.build_solver(plan, dtype=np.float64).solve(inputs)
    if np.any(result.status != ForwardStatus.RESULT_AVAILABLE):
        raise RuntimeError(f"solve failed: {result.diagnostics}")
    print("forward status:", ForwardStatus(int(result.status[0])).name)
    print(result.waypoints)


if __name__ == "__main__":
    main()
