"""Solve with Torch arrays on CPU or through the cu12 CPU transport."""

import argparse

import numpy as np
import torch
from constraints import add_constraint_argument
from trajectory_problem import make_problem

from pivopt import ForwardStatus, OptimizerStatus
from pivopt import problem as p
from pivopt import torch as cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    add_constraint_argument(parser)
    args = parser.parse_args()
    device = torch.device(args.device)
    plan, source = make_problem(np.float32, constraints=args.constraints)
    inputs = p.Inputs(
        plan=plan,
        values=tuple(
            (handle, torch.from_numpy(value).to(device))
            for handle, value in source.values
        ),
    )
    solver = cpu.build_solver(
        plan,
        dtype=torch.float32,
        compute_device="cpu" if device.type == "cuda" else None,
    )
    with torch.no_grad():
        result = solver.solve(inputs)

    print("device:", device)
    print("forward status:", ForwardStatus(int(result.status[0].item())).name)
    print(
        "optimizer status:",
        OptimizerStatus(int(result.diagnostics.optimizer_status[0].item())).name,
    )
    print("cost:", float(result.cost[0].item()))
    print("stationarity:", float(result.diagnostics.stationarity[0].item()))
    print("durations:", result.durations[0].cpu().tolist())
    print("waypoint:", result.waypoints[0, 0].cpu().tolist())
    print("coefficients shape:", tuple(result.coefficients.shape))


if __name__ == "__main__":
    main()
