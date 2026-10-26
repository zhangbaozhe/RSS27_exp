"""Public example smoke: real multi-step AD, not a quality acceptance corpus."""

import re

import numpy as np
import pytest
from cpu_training import base_problem, run_jax, run_torch

from pivopt import problem as p


@pytest.mark.parametrize("framework", ("jax", "torch"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("compiled", (False, True))
def test_public_cpu_training_example(framework, dtype, compiled, capsys):
    if framework == "jax":
        import jax

        with jax.enable_x64():
            run_jax(compiled, 3, dtype)
    else:
        import torch

        if compiled:
            torch.compiler.reset()
        default_dtype = torch.get_default_dtype()
        run_torch(compiled, 3, dtype)
        assert torch.get_default_dtype() == default_dtype

    output = capsys.readouterr().out
    losses = [float(value) for value in re.findall(r"loss: ([^\s]+)", output)]
    assert len(losses) == 4  # Three AD/update calls and the final fresh inspection.
    assert np.isfinite(losses).all()
    # The Frozen transfer is eligible for strict IFT; the example raises before
    # any update if a nonfinite gradient is returned.
    assert losses[-1] <= losses[0] + 1e-7
    assert "strict backward status:" in output
    assert output.count("forward error: [0]") == 3
    assert "fresh forward status: [0]" in output
    # The example itself checks each actual control and received gradient dtype.


def test_training_model_preserves_actual_smoothing_width():
    # The old example printed 0.1 but inherited the native 0.01 default.
    # Its inactive penalties do not expose this difference in smoke-test losses.
    plan, _ = base_problem()
    assert len(plan.spec.objective) == 8
    for term in plan.spec.objective[1:]:
        assert term.cost.profile == p.Hinge(0.01)
