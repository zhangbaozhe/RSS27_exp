"""Inspect a problem's resolved contract and CPU Solver before binding data.

Run: python examples/cpu_explain.py [--dtype float32|float64]
Works with full and NumPy light installations; backward capability is enabled
when available. This example needs no Inputs and does not run an optimizer.

Plan.explain() describes normalized structure and active input slots;
Solver.explain() adds the compiled CPU configuration. Workspace memory is
measured by separate complete-call resource probes, not a public byte query.
"""

import argparse

import numpy as np
from constraints import Calibration, add_constraint_argument, term

from pivopt import numpy as cpu
from pivopt import problem as p


def make_spec(*, constraints="hinge") -> p.ProblemSpec:
    trajectory = p.MincoSpec(
        p.Space(position=3),
        derivative_order=3,
        pieces=2,
        head_state=p.Input("head", differentiate=True),
        tail_state=p.Input("tail", differentiate=True),
        time=p.FrozenTime(durations=p.Input("times", differentiate=True)),
        waypoints=p.DirectWaypoints(),
    )
    return p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=4),
        objective=(
            p.Energy("jerk_energy"),
            term(
                p.Integral,
                "speed_limit",
                lambda profile: p.NormBelow(trajectory.velocity, 3.0, profile=profile),
                Calibration(
                    hinge_weight=p.Input("speed_weight", differentiate=True),
                    width=0.1,
                    barrier_weight=p.Input("speed_weight", differentiate=True),
                    delta=0.022,
                    activation=0.3,
                ),
                constraints,
            ),
        ),
        forward=p.BFGS(initial_guess=p.Input("initial")),
        backward=p.Implicit(),
    )


def make_problem(*, constraints="hinge"):
    """Bind one transfer for callers that also want to execute this contract.

    speed_weight remains a differentiable input: rho=1 or mu=0.022 is supplied
    explicitly here, while make_spec only selects the static profile.
    """
    plan = p.build_plan(make_spec(constraints=constraints), p.PlanShape(batch_size=1))
    head = np.zeros((1, 3, 3))
    tail = head.copy()
    tail[0, 0, 0] = 2.0
    inputs = plan.inputs(
        head=head,
        tail=tail,
        times=np.ones((1, 2)),
        initial=np.asarray([[1.0, 0.1, 0.0]]),
        speed_weight=np.asarray([1.0 if constraints == "hinge" else 0.022]),
    )
    return plan, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    add_constraint_argument(parser)
    args = parser.parse_args()
    dtype = np.dtype(args.dtype)

    spec = make_spec(constraints=args.constraints)
    shape = p.PlanShape(batch_size=1)  # B=1 also works with Single / NumPy light.
    plan = p.build_plan(spec, shape)

    print("1. Plan: normalized structure and active input slots")
    print(
        "S=derivative order, Dim=dimension, B=batch, K=pieces, Q=intervals, N=decisions"
    )
    print(plan.explain())

    print("2. Requirements: active input shapes and mathematical operations")
    requirements = plan.requirements()
    for item in requirements.inputs:
        gradient = "enabled" if item.differentiate else "stopped"
        print(
            f"  input[{item.input.index}] {item.name}: "
            f"shape={item.shape}, outer_gradient={gradient}"
        )
    for component, operation in requirements.operations:
        print(f"  {component}: {operation}")

    print("\n3. Current build capabilities (no execution workspace allocated)")
    for operation in p.Operation:
        report = cpu.check_support(plan, operation)
        print(
            f"  {operation.name}: {'supported' if report.supported else 'unavailable'}"
        )
        for reason in report.reasons:
            print(f"    {reason}")

    # A legal Implicit Plan can still build a forward-only Solver in light mode.
    # The capability query does not certify per-call values or convergence.
    state_support = cpu.check_support(plan, p.Operation.SOLVE_WITH_STATE)
    print(f"\n4. Solver: public dtype={dtype.name}, internal solving=float64")
    solver = cpu.build_solver(plan, dtype=dtype, backward=state_support.supported)
    print(solver.explain())
    if not state_support.supported:
        print("  Backward unavailable: " + "; ".join(state_support.reasons))

    # Removing the only consumer leaves the declaration present but inactive.
    # The original Plan is immutable. The changed Plan needs a new Solver.
    print(
        "\n5. Remove speed_limit: compare contracts and inspect inactive speed_weight"
    )
    energy_only = p.build_plan(spec.without_term("speed_limit"), shape)
    print("  plan.diff(energy_only):", plan.diff(energy_only))
    print(energy_only.explain())

    print("Reuse each immutable Plan and Solver when only numerical values change.")
    print("Each solve receives new Inputs; the Solver reuses its owned Workspace pool.")


if __name__ == "__main__":
    main()
