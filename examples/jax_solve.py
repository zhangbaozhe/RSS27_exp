"""Solve with JAX arrays on CPU or through the cu12 CPU transport."""

import argparse

import jax
import jax.numpy as jnp
from constraints import add_constraint_argument
from trajectory_problem import make_problem

from pivopt import ForwardStatus, OptimizerStatus
from pivopt import jax as cpu
from pivopt import problem as p

jax.config.update("jax_enable_x64", True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    add_constraint_argument(parser)
    args = parser.parse_args()
    device = jax.devices("gpu" if args.device == "cuda" else "cpu")[0]
    plan, source = make_problem(gradient_tolerance=1e-2, constraints=args.constraints)
    inputs = p.Inputs(
        plan=plan,
        values=tuple(
            (handle, jax.device_put(value, device)) for handle, value in source.values
        ),
    )
    solver = cpu.build_solver(
        plan,
        dtype=jnp.float64,
        compute_device="cpu" if args.device == "cuda" else None,
    )
    run = jax.jit(lambda values: solver.solve(values))
    result = jax.device_get(run(inputs))

    print("device:", device)
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
