"""One explicit trajectory model shared by the three minimal CPU examples.

The Plan fixes structure and input slots. Each frontend binds its own arrays;
reuse the Solver when only the numerical data changes.
"""

import numpy as np
from constraints import Calibration, term

from pivopt import problem as p


def make_problem(dtype=np.float64, *, gradient_tolerance=1e-6, constraints="hinge"):
    trajectory = p.MincoSpec(
        p.Space(position=3),
        derivative_order=2,
        pieces=2,
        head_state=p.Input("head", differentiate=True),
        tail_state=p.Input("tail", differentiate=True),
        time=p.FixedTotalTime(total_duration=2.0, minimum_piece_duration=0.1),
        waypoints=p.DirectWaypoints(),
    )
    quad = p.Quadrotor()
    spec = p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=4),
        objective=(
            p.Energy("energy"),
            term(
                p.Integral,
                "corridor",
                lambda profile: p.Inside(
                    trajectory.position,
                    p.Input("corridor", differentiate=True),
                    profile=profile,
                ),
                Calibration(
                    hinge_weight=10,
                    width=0.1,
                    barrier_weight=0.22,
                    delta=0.022,
                    activation=0.5,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "speed",
                lambda profile: p.NormBelow(trajectory.velocity, 3, profile=profile),
                Calibration(
                    hinge_weight=1,
                    width=0.1,
                    barrier_weight=0.022,
                    delta=0.022,
                    activation=0.3,
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
                lambda profile: p.NormBelow(quad.body_rate, 0, profile=profile),
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
                lambda profile: p.TiltBelow(quad.attitude, 0, profile=profile),
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
                lambda profile: p.Within(quad.thrust, 0, 0, profile=profile),
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
        forward=p.LBFGS(
            initial_guess=p.Input("initial"),
            settings=p.OptimizerSettings(
                past=0,
                gradient_tolerance=gradient_tolerance,
                maximum_iterations=50,
                maximum_line_searches=64,
            ),
        ),
        backward=p.ForwardOnly(),
    )
    plan = p.build_plan(
        spec,
        p.PlanShape(
            batch_size=1,
            input_shapes=(p.InputShape(p.InputRef("corridor"), (1, 2, 6, 4)),),
        ),
    )
    halfspaces = np.zeros((1, 2, 6, 4), dtype=dtype)
    for face in range(6):
        halfspaces[..., face, face // 2] = 1 if face % 2 == 0 else -1
        halfspaces[..., face, 3] = -5
    values = (
        (plan.input("head"), np.zeros((1, 3, 2), dtype=dtype)),
        (plan.input("tail"), np.asarray([[[2, 0], [0, 0], [0, 0]]], dtype=dtype)),
        (plan.input("corridor"), halfspaces),
        (plan.input("initial"), np.asarray([[0, 0.7, 0.1, -0.1]], dtype=dtype)),
    )
    return plan, p.Inputs(plan=plan, values=values)
