"""Solve with the NumPy CPU frontend: python examples/numpy_solve.py.

This forward-only example also runs in the NumPy light installation.
See trajectory_problem.py for the explicit model and data declarations.
"""

import argparse

import numpy as np
from constraints import add_constraint_argument
from trajectory_problem import make_problem

from pivopt import ForwardStatus, OptimizerStatus
from pivopt import numpy as cpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_constraint_argument(parser)
    args = parser.parse_args()
    plan, inputs = make_problem(constraints=args.constraints)
    solver = cpu.build_solver(plan, dtype=np.float64)
    result = solver.solve(inputs)

    print("backend: NumPy CPU")
    print("forward status:", ForwardStatus(int(result.status[0])).name)
    print(
        "optimizer status:",
        OptimizerStatus(int(result.diagnostics.optimizer_status[0])).name,
    )
    print("cost:", float(result.cost[0]))
    print("stationarity:", float(result.diagnostics.stationarity[0]))
    print("durations:", result.durations[0])
    print("waypoint:", result.waypoints[0, 0])
    print("coefficients shape:", result.coefficients.shape)


if __name__ == "__main__":
    main()
