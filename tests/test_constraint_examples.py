"""Every inequality example builds and solves both profiles in this process."""

import importlib

import numpy as np
import pytest

from pivopt import ForwardStatus
from pivopt import numpy as cpu
from pivopt import problem as p


@pytest.mark.parametrize("mode", ("hinge", "relaxed-barrier"))
@pytest.mark.parametrize(
    "name",
    (
        "trajectory_problem",
        "cpu_explain",
        "cpu_plan_shapes",
        "planar_base",
        "unicycle",
        "arm",
        "cpu_training",
        "cpu_polytope_training",
    ),
)
def test_constraint_example_solve(name, mode, record_property):
    if name == "cpu_polytope_training":
        if not cpu.check_support(
            p.build_plan(
                importlib.import_module("cpu_explain").make_spec(),
                p.PlanShape(batch_size=1),
            ),
            p.Operation.SOLVE_WITH_STATE,
        ).supported:
            pytest.skip("polytope training requires backward support")
        import torch

        example = importlib.import_module(name)
        torch.manual_seed(7)
        scenes = example.sample_scenes(1, generator=torch.Generator().manual_seed(7))
        policy = example.PolytopePolicy().double()
        with torch.no_grad():
            decoded = example.decode_action(policy(scenes.features()), scenes)
            setup = example.prepare_problem(1, constraints=mode)
            plan = setup.plan
            result = setup.solver.solve(setup.bind(scenes, decoded))
        status = result.status.numpy()
        cost = result.cost.numpy()
    else:
        example = importlib.import_module(name)
        if name == "cpu_plan_shapes":
            solver, inputs = example.build_case(
                example.make_spec(constraints=mode),
                example.make_data(pieces=4, faces=6),
                intervals=16,
            )
            plan = solver.plan
        else:
            if name == "unicycle":
                plan, inputs = example._problem(example.CORNER, constraints=mode)
            elif name == "cpu_training":
                plan, inputs = example.base_problem(constraints=mode)
            else:
                plan, inputs = example.make_problem(constraints=mode)
            solver = cpu.build_solver(plan, dtype=np.float64)
        result = solver.solve(inputs)
        status, cost = result.status, result.cost

    expected = p.Hinge if mode == "hinge" else p.RelaxedBarrier
    inequalities = [
        item
        for item in plan.spec.objective
        if isinstance(item, (p.Integral, p.AtEnd))
        and not isinstance(item.cost, p.Quadratic)
    ]
    assert inequalities
    assert all(isinstance(item.cost.profile, expected) for item in inequalities)
    assert np.all(status == ForwardStatus.RESULT_AVAILABLE), result.diagnostics
    assert np.isfinite(cost).all()
    record_property("forward_status", np.asarray(status).tolist())
    record_property("cost", np.asarray(cost).tolist())
