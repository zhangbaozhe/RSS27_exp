"""Small CPU training/diagnostic example, not a gradient-quality benchmark.

Run ``python examples/cpu_training.py --framework jax --compiled``
or replace ``jax`` with ``torch``. Use ``--dtype float32`` for f32 learning
parameters, outputs and gradients with native f64 computation. No application
or test helpers are imported.
Strict IFT returns NaN gradients for rejected endpoints; the example stops before
updating parameters if that occurs.
"""

import argparse

import numpy as np
from constraints import Calibration, add_constraint_argument, term

from pivopt import problem as p

OPTIONS = p.OptimizerSettings(
    maximum_iterations=500,
    maximum_line_searches=256,
    gradient_tolerance=1e-5,
)


def physical_loss(coefficients, durations, xp):
    """Position loss at fixed global time 0.5 s, with fixed metre-scale target."""
    ends = durations.cumsum(-1)
    starts = ends - durations
    local_time = 0.5 - starts
    position = xp.zeros_like(coefficients[:, :, 0])
    for degree in range(coefficients.shape[2] - 1, -1, -1):
        position = position * local_time[:, :, None] + coefficients[:, :, degree]
    selected = ((starts <= 0.5) & (0.5 < ends))[:, :, None]
    position = (position * selected).sum(1)
    # Four unit-duration pieces keep the timestamp in range.
    return (
        (position[:, 0] - 0.45) ** 2
        + (position[:, 1] - 0.10) ** 2
        + position[:, 2] ** 2
    ).sum()


def base_problem(dtype=np.float64, *, optimizer="bfgs", constraints="hinge"):
    """Prepare a Frozen transfer with the complete eight-term objective.

    With inactive constraints, optimizing internal times and waypoints together
    can reparameterize one global polynomial without changing its energy. That
    FixedTotal null direction fails strict IFT. This example chooses Frozen
    explicitly; the solver does not repair or regularize such a Hessian.
    """
    trajectory = p.MincoSpec(
        p.Space(position=3),
        pieces=4,
        head_state=p.Input("head", differentiate=True),
        tail_state=p.Input("tail", differentiate=True),
        time=p.FrozenTime(durations=p.Input("durations", differentiate=True)),
        waypoints=p.DirectWaypoints(),
    )
    quad = p.Quadrotor()
    spec = p.ProblemSpec(
        trajectory=trajectory,
        sampling=p.Trapezoidal(intervals_per_piece=16),
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
                    hinge_weight=1000,
                    width=0.01,
                    barrier_weight=2.2,
                    delta=0.0022,
                    activation=0.5,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "velocity",
                lambda profile: p.NormBelow(trajectory.velocity, 3, profile=profile),
                Calibration(
                    hinge_weight=1000,
                    width=0.01,
                    barrier_weight=2.2,
                    delta=0.0022,
                    activation=0.3,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "acceleration",
                lambda profile: p.NormBelow(
                    trajectory.acceleration, 20, profile=profile
                ),
                Calibration(
                    hinge_weight=1000,
                    width=0.01,
                    barrier_weight=2.2,
                    delta=0.0022,
                    activation=2.0,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "jerk",
                lambda profile: p.NormBelow(trajectory.jerk, 100, profile=profile),
                Calibration(
                    hinge_weight=1000,
                    width=0.01,
                    barrier_weight=2.2,
                    delta=0.0022,
                    activation=10.0,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "body_rate",
                lambda profile: p.NormBelow(quad.body_rate, 10, profile=profile),
                Calibration(
                    hinge_weight=1000,
                    width=0.01,
                    barrier_weight=2.2,
                    delta=0.0022,
                    activation=1.0,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "tilt",
                lambda profile: p.TiltBelow(quad.attitude, 1, profile=profile),
                Calibration(
                    hinge_weight=1000,
                    width=0.01,
                    barrier_weight=2.2,
                    delta=0.0022,
                    activation=0.1,
                ),
                constraints,
            ),
            term(
                p.Integral,
                "thrust",
                lambda profile: p.Within(quad.thrust, 0, 30, profile=profile),
                Calibration(
                    hinge_weight=1000,
                    width=0.01,
                    barrier_weight=2.2,
                    delta=0.0022,
                    activation=1.5,
                ),
                constraints,
            ),
        ),
        forward=(p.BFGS if optimizer == "bfgs" else p.LBFGS)(
            initial_guess=p.Input("initial"), settings=OPTIONS
        ),
        backward=p.Implicit(),
    )
    plan = p.build_plan(
        spec,
        p.PlanShape(
            batch_size=1,
            input_shapes=(p.InputShape(p.InputRef("corridor"), (1, 4, 6, 4)),),
        ),
    )
    head = np.zeros((1, 3, 3), dtype=dtype)
    tail = head.copy()
    tail[0, 0, 0] = 2
    halfspaces = np.zeros((1, 4, 6, 4), dtype=dtype)
    for face in range(6):
        halfspaces[..., face, face // 2] = -1 if face % 2 else 1
        halfspaces[..., face, 3] = -5
    initial = np.zeros((1, 9), dtype=dtype)
    initial[0, ::3] = [0.5, 1, 1.5]
    return plan, p.Inputs(
        plan=plan,
        values=(
            (plan.input("head"), head),
            (plan.input("tail"), tail),
            (plan.input("corridor"), halfspaces),
            (plan.input("initial"), initial),
            (plan.input("durations"), np.ones((1, 4), dtype=dtype)),
        ),
    )


def run_jax(compiled, steps, dtype="float64", *, optimizer="bfgs", constraints="hinge"):
    import jax
    import jax.numpy as jnp

    from pivopt import jax as api

    jax.config.update("jax_enable_x64", True)
    outer_dtype = getattr(jnp, dtype)
    plan, arrays = base_problem(
        getattr(np, dtype), optimizer=optimizer, constraints=constraints
    )
    base = p.Inputs(
        plan=plan,
        values=tuple((handle, jnp.asarray(value)) for handle, value in arrays.values),
    )
    tail_handle = plan.input("tail")
    tail = next(value for handle, value in base.values if handle == tail_handle)
    forward = api.build_solver(plan, dtype=dtype, backward=True)

    def problem(control):
        return base.with_value(
            tail_handle, jnp.concatenate((control[..., None], tail[:, :, 1:]), -1)
        )

    def objective(control):
        result = forward.solve(problem(control))
        return physical_loss(result.coefficients, result.durations, jnp), result.status

    step = jax.value_and_grad(objective, has_aux=True)
    if compiled:
        step = jax.jit(step)
    control = tail[:, :, 0]
    for index in range(steps):
        (loss, error), gradient = step(control)
        assert control.dtype == gradient.dtype == outer_dtype
        print(index, "loss:", float(loss), "forward error:", error.tolist())
        if not bool(jnp.all(jnp.isfinite(gradient))):
            raise RuntimeError("strict IFT rejected this update; inspect pullback")
        control = control - 0.1 * gradient

    # One fresh solve at the final control. Diagnostics describe this forward;
    # a backward has not happened yet, so its status is not among them.
    result, state = forward.solve_with_state(problem(control))
    coefficient_seed, duration_seed = jax.grad(
        lambda coefficients, durations: physical_loss(coefficients, durations, jnp),
        argnums=(0, 1),
    )(result.coefficients, result.durations)
    status = forward.pullback(
        state, p.ResultSeeds(coefficients=coefficient_seed, durations=duration_seed)
    ).diagnostics
    print("strict backward status:", status.status.tolist())
    print(
        "final loss:", float(physical_loss(result.coefficients, result.durations, jnp))
    )
    print("fresh forward status:", result.status.tolist())
    print("stationarity:", result.diagnostics.stationarity.tolist())


def run_torch(
    compiled, steps, dtype="float64", *, optimizer="bfgs", constraints="hinge"
):
    import torch

    from pivopt import torch as api

    outer_dtype = getattr(torch, dtype)
    plan, arrays = base_problem(
        getattr(np, dtype), optimizer=optimizer, constraints=constraints
    )
    base = p.Inputs(
        plan=plan,
        values=tuple(
            (handle, torch.from_numpy(value)) for handle, value in arrays.values
        ),
    )
    tail_handle = plan.input("tail")
    tail = next(value for handle, value in base.values if handle == tail_handle)
    forward = api.build_solver(plan, dtype=dtype, backward=True)

    def problem(control):
        return base.with_value(
            tail_handle, torch.cat((control[..., None], tail[:, :, 1:]), -1)
        )

    def objective(control):
        result = forward.solve(problem(control))
        return physical_loss(
            result.coefficients, result.durations, torch
        ), result.status

    evaluate = torch.compile(objective, fullgraph=True) if compiled else objective
    control = tail[:, :, 0].detach().clone().requires_grad_()
    for index in range(steps):
        loss, error = evaluate(control)
        (gradient,) = torch.autograd.grad(loss, control)
        assert control.dtype == gradient.dtype == outer_dtype
        print(index, "loss:", float(loss.detach()), "forward error:", error.tolist())
        if not bool(torch.all(torch.isfinite(gradient))):
            raise RuntimeError("strict IFT rejected this update; inspect pullback")
        control = (control - 0.1 * gradient).detach().requires_grad_()

    # One fresh solve at the final control. Diagnostics describe this forward;
    # a backward has not happened yet, so its status is not among them.
    with torch.no_grad():
        result, state = forward.solve_with_state(problem(control.detach()))
    coefficients = result.coefficients.detach().requires_grad_()
    durations = result.durations.detach().requires_grad_()
    coefficient_seed, duration_seed = torch.autograd.grad(
        physical_loss(coefficients, durations, torch), (coefficients, durations)
    )
    status = forward.pullback(
        state, p.ResultSeeds(coefficients=coefficient_seed, durations=duration_seed)
    ).diagnostics
    print("strict backward status:", status.status.tolist())
    print(
        "final loss:",
        float(physical_loss(result.coefficients, result.durations, torch)),
    )
    print("fresh forward status:", result.status.tolist())
    print("stationarity:", result.diagnostics.stationarity.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("jax", "torch"), required=True)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--optimizer", choices=("bfgs", "lbfgs"), default="bfgs")
    add_constraint_argument(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    print(f"Python {args.dtype}; CPU native float64; constraints: {args.constraints}")
    print("A fresh diagnostic pullback is explicit; it is not the last AD status.")
    (run_jax if args.framework == "jax" else run_torch)(
        args.compiled,
        args.steps,
        args.dtype,
        optimizer=args.optimizer,
        constraints=args.constraints,
    )


if __name__ == "__main__":
    main()
