"""Plan a UR5e joint trajectory to a task-space terminal target."""

import argparse
from pathlib import Path

import numpy as np
from constraints import Calibration, add_constraint_argument, term

from pivopt import ForwardStatus
from pivopt import numpy as cpu
from pivopt import problem as p


def make_problem(*, constraints="hinge"):
    urdf = Path(__file__).parent / "assets" / "ur5e.urdf"
    spheres = {
        "shoulder_link": (((0.0, 0.0, 0.0), 0.10),),
        "upper_arm_link": (((-0.18, 0.0, 0.0), 0.09), ((-0.36, 0.0, 0.0), 0.09)),
        "forearm_link": (((-0.16, 0.0, 0.05), 0.08), ((-0.34, 0.0, 0.11), 0.08)),
        "wrist_1_link": (((0.0, -0.05, 0.0), 0.07),),
        "wrist_2_link": (((0.0, 0.05, 0.0), 0.07),),
        "tool0": (((0.0, 0.0, 0.0), 0.06),),
    }
    chain, limits = p.SerialChain.from_urdf(urdf, tip="tool0", spheres=spheres)
    pieces = 4
    trajectory = p.MincoSpec(
        p.Space(joints=chain.dof),
        derivative_order=3,
        pieces=pieces,
        head_state=p.Input("head"),
        tail_state=p.FreeTail(derivatives=p.Input("tail_derivatives")),
        time=p.FixedTotalTime(total_duration=4.0, minimum_piece_duration=0.1),
        waypoints=p.DirectWaypoints(),
    )
    # Continuous URDF joints are unbounded. This example chooses a planning
    # branch of [-pi, pi]; callers may choose another branch or omit this term.
    lower = np.where(np.isfinite(limits.lower), limits.lower, -np.pi)
    upper = np.where(np.isfinite(limits.upper), limits.upper, np.pi)
    spec = p.ProblemSpec(
        trajectory=trajectory,
        objective=(
            p.Energy("smooth"),
            term(
                p.Integral,
                "joint_limits",
                lambda profile: p.Within(
                    trajectory.joints, lower, upper, profile=profile
                ),
                Calibration(
                    hinge_weight=1000.0,
                    width=0.03,
                    barrier_weight=6.7,
                    delta=0.0067,
                    activation=0.314159,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "speed_limits",
                lambda profile: p.Within(
                    trajectory.joints.d(1),
                    -limits.velocity,
                    limits.velocity,
                    profile=profile,
                ),
                Calibration(
                    hinge_weight=1000.0,
                    width=0.05,
                    barrier_weight=11.0,
                    delta=0.011,
                    activation=0.314,
                ),
                constraints,
            ),
            p.AtEnd(
                "tip_target",
                p.Quadratic(chain.tip_position, target=(0.64, 0.13, 0.38)),
                weight=200.0,
            ),
            term(
                p.Integral,
                "workspace",
                lambda profile: p.Inside(
                    chain.spheres, p.Input("workspace"), profile=profile
                ),
                Calibration(
                    hinge_weight=10000.0,
                    width=0.05,
                    barrier_weight=110.0,
                    delta=0.011,
                    activation=0.15,
                ),
                constraints,
            ),
        ),
        sampling=p.Trapezoidal(intervals_per_piece=12),
        forward=p.LBFGS(
            initial_guess=p.Input("initial"),
            memory_size=12,
            settings=p.OptimizerSettings(maximum_iterations=300, past=0),
        ),
        backward=p.ForwardOnly(),
    )
    plan = p.build_plan(
        spec,
        p.PlanShape(
            batch_size=1,
            input_shapes=(p.InputShape(p.InputRef("workspace"), (1, pieces, 6, 4)),),
        ),
    )
    head = np.zeros((1, chain.dof, 3))
    head[0, :, 0] = (0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0)
    tail_derivatives = np.zeros((1, chain.dof, 2))
    terminal = np.asarray(((0.0, -1.2, 1.4, -1.8, -np.pi / 2, 0.0),))
    waypoints = np.stack(
        [
            head[0, :, 0] + (terminal[0] - head[0, :, 0]) * index / pieces
            for index in range(1, pieces)
        ]
    )[None]
    rows = np.asarray(
        (
            (1, 0, 0, -3),
            (-1, 0, 0, -1),
            (0, 1, 0, -2),
            (0, -1, 0, -2),
            (0, 0, 1, -2),
            (0, 0, -1, -1),
        ),
        dtype=np.float64,
    )
    workspace = np.broadcast_to(rows, (1, pieces, 6, 4)).copy()
    inputs = plan.inputs(
        head=head,
        tail_derivatives=tail_derivatives,
        workspace=workspace,
        initial=plan.initial_guess(
            waypoints=waypoints,
            terminal=terminal,
            durations=np.ones((1, pieces)),
        ),
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
    powers = result.durations[:, -1, None] ** np.arange(6)
    terminal_joints = np.einsum("bpd,bp->bd", result.coefficients[:, -1], powers)
    print("forward status:", ForwardStatus(int(result.status[0])).name)
    print("interior waypoints:", result.waypoints)
    print("terminal joints:", terminal_joints)


if __name__ == "__main__":
    main()
