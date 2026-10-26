"""Public-boundary checks for the certified gate follow-up driver."""

import ast
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
study = importlib.import_module("cpu_gate_certified_training")


def test_certified_gate_driver_uses_only_public_pivopt_imports():
    path = Path(study.__file__)
    source = path.read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("pivopt._") for name in imported)
    assert "_response_trace" not in source
    assert "from tools" not in source
    assert "import tools" not in source


def test_certified_gate_sampler_and_decoder_stay_inside_the_safe_envelope():
    task = study.CertifiedGateTask(4)
    first = task.sample(torch.Generator().manual_seed(1021))
    second = task.sample(torch.Generator().manual_seed(1021))
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
    assert first.features.shape == (4, 23)
    assert task.certify(first)["all_boxes_free"]
    torch.manual_seed(7)
    policy = study.LearningPolicy(task.features, task.outputs, geometry=task.geometry)
    corridor, caps, width = task.decode(policy(first.features), first)
    assert corridor.shape == (4, 8, 4, 3)
    assert caps.shape == (4, 3)
    assert bool((width >= task.width_min).all())
    assert bool((width <= task.width_max).all())


def test_certified_gate_failed_lane_is_preserved_in_strict_json(tmp_path):
    class OneFailedLane(torch.nn.Module):
        def forward(self, features):
            raw = torch.zeros((len(features), task.outputs), dtype=features.dtype)
            raw[0, 0] = torch.nan
            return raw

    task = study.CertifiedGateTask(2, ("A",))
    batch = task.sample(torch.Generator().manual_seed(1021))
    metrics = study.evaluate(task, OneFailedLane(), batch, mode="A")
    assert metrics["status"].tolist() == [
        int(study.ForwardStatus.INVALID_PROBLEM),
        int(study.ForwardStatus.RESULT_AVAILABLE),
    ]
    assert metrics["utility"][0] == 0.0
    path = tmp_path / "failed-lane.json"
    study.write_json(path, metrics)
    saved = study.json.loads(path.read_text())
    assert saved["status"] == metrics["status"].tolist()
    assert saved["task"][0] is None
    assert saved["utility"][0] == 0.0


@pytest.mark.skipif(
    os.environ.get("PIVOPT_BUILD_MODE") == "numpy",
    reason="light build excludes implicit backward",
)
def test_certified_gate_public_numpy_step_updates_a_and_rs():
    torch.set_num_threads(1)
    task = study.CertifiedGateTask(2)
    batch = task.sample(torch.Generator().manual_seed(123))
    initial = None
    for mode in ("A", "RS"):
        torch.manual_seed(9)
        model = study.LearningPolicy(
            task.features, task.outputs, geometry=task.geometry
        )
        before = torch.cat([value.detach().flatten() for value in model.parameters()])
        if initial is None:
            initial = before.clone()
        else:
            torch.testing.assert_close(before, initial, rtol=0, atol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        solver = study.cpu.build_solver(
            task.plans[mode], dtype=np.float64, backward=True
        )
        row = study.training_step(
            task,
            solver,
            model,
            optimizer,
            batch,
            mode=mode,
            task_scale=1.0,
        )
        assert row["updated"]
        after = torch.cat([value.detach().flatten() for value in model.parameters()])
        assert not torch.equal(before, after)
        if mode == "A":
            assert row["response_displacement"] is None
        else:
            assert row["response_displacement"] > 0
