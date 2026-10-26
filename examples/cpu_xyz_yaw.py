"""Solve an xyz+yaw trajectory with either CPU installation preset.

Run: python examples/cpu_xyz_yaw.py
Yaw uses unwrapped radians; geometry remains three-dimensional.
"""

import numpy as np

from pivopt import ForwardStatus
from pivopt import numpy as cpu
from pivopt import problem as p


def main():
    # Names and shapes are resolved once. Only Inputs contains per-call arrays.
    # This example explicitly minimizes xyz/yaw jerk energy; it adds no limits
    # or geometric penalties. Add named sampled terms when those are needed.
    trajectory = p.MincoSpec(
        p.Space(position=3, yaw=True),
        derivative_order=3,
        pieces=4,
        head_state=p.Input("head"),
        tail_state=p.Input("tail"),
        time=p.FixedTotalTime(
            total_duration=p.Input("total"), minimum_piece_duration=0.1
        ),
        waypoints=p.DirectWaypoints(),
    )
    spec = p.ProblemSpec(
        trajectory=trajectory,
        objective=(p.Energy("jerk_energy", weights={"position": 1.0, "yaw": 0.5}),),
        forward=p.BFGS(initial_guess=p.Input("initial")),
        backward=p.ForwardOnly(),
    )
    plan = p.build_plan(spec, p.PlanShape(batch_size=1))
    solver = cpu.build_solver(plan, dtype=np.float64)
    head = np.zeros((1, 4, 3), dtype=np.float64)
    tail = np.zeros_like(head)
    head[0, 0, 1] = tail[0, 0, 1] = 1
    tail[0, 0, 0] = 4
    # +2 degrees across pi, followed by one full positive turn. No wrapping.
    head[0, 3, 0], tail[0, 3, 0] = np.deg2rad([179, 181 + 360])
    head[0, 3, 1] = tail[0, 3, 1] = (tail[0, 3, 0] - head[0, 3, 0]) / 4
    waypoints = np.zeros((1, 3, 4), dtype=np.float64)
    waypoints[0, :, 0] = [1, 3, 3]
    waypoints[0, :, 3] = head[0, 3, 0] + head[0, 3, 1] * np.arange(1, 4)
    initial_durations = np.array([[1.0, 2.0, 3.0, 4.0]])

    inputs = plan.inputs(
        head=head,
        tail=tail,
        total=initial_durations.sum(axis=1),
        initial=plan.initial_guess(waypoints=waypoints, durations=initial_durations),
    )
    result = solver.solve(inputs)
    if np.any(result.status != ForwardStatus.RESULT_AVAILABLE):
        raise RuntimeError(f"solve failed: {result.status}")
    print("coefficient shape [B, K, 2*S, Dim]:", result.coefficients.shape)
    print("waypoints [x, y, z, yaw in radians]:\n", result.waypoints)
    print("durations: \n", result.durations)


if __name__ == "__main__":
    main()
