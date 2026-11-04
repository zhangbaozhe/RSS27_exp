"""Scenarios and planner recording shared by the re-pin equivalence check.

Each scenario drives the examples through the same public entry points that the
example tests use. While a scenario runs, the NumPy and Torch ``build_solver``
functions are wrapped, so every planner call is captured with its named inputs,
its optimizer, backward and time records, its public result and, for pullbacks,
its seeds. The same code runs at the old pin, where ``generate_reference.py``
stores the records, and at the new pin, where ``test_repin_equivalence.py``
compares against them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import importlib
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco"
SAME_POINT_TOLERANCE = 1e300


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def flatten(prefix: str, value, out: dict, *, record_types: bool = False) -> None:
    """Store every array or scalar reachable from a public record.

    With ``record_types``, each record also stores its type name under
    ``<prefix>.@type``, enum members are stored by name and empty optional
    fields are stored as ``"None"``. Spec records use it, so a record without
    fields and a disabled gate are still visible.
    """
    if value is None:
        if record_types:
            out[prefix] = np.asarray("None")
        return
    if callable(value) and not dataclasses.is_dataclass(value):
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if record_types:
            out[f"{prefix}.@type"] = np.asarray(type(value).__name__)
        for item in dataclasses.fields(value):
            flatten(
                f"{prefix}.{item.name}",
                getattr(value, item.name),
                out,
                record_types=record_types,
            )
    elif isinstance(value, enum.Enum) and record_types:
        out[prefix] = np.asarray(f"{type(value).__name__}.{value.name}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            flatten(f"{prefix}[{index}]", item, out, record_types=record_types)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            flatten(f"{prefix}.{key}", item, out, record_types=record_types)
    else:
        array = to_numpy(value)
        if array.dtype == object:
            array = np.asarray(str(value))
        out[prefix] = array


def input_arrays(plan, inputs) -> dict[str, np.ndarray]:
    names = {item.input.index: item.name for item in plan.requirements().inputs}
    return {names[handle.index]: to_numpy(value) for handle, value in inputs.values}


class RecordingSolver:
    def __init__(self, solver, plan, calls: list, backend: str):
        self._solver = solver
        self._plan = plan
        self._calls = calls
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def _record(self, kind, **items):
        self._calls.append({"kind": kind, "plan": self._plan, **items})

    def solve(self, inputs):
        result = self._solver.solve(inputs)
        self._record("solve", inputs=inputs, result=result, backend=self._backend)
        return result

    def solve_with_state(self, inputs):
        result, state = self._solver.solve_with_state(inputs)
        self._record(
            "solve_with_state",
            inputs=inputs,
            result=result,
            state=state,
            backend=self._backend,
        )
        return result, state

    def pullback(self, state, seeds):
        pulled = self._solver.pullback(state, seeds)
        self._record(
            "pullback", seeds=seeds, result=pulled, state=state, backend=self._backend
        )
        return pulled


_ACTIVE: list[list[dict]] = []


def _wrap(build, backend: str):
    """Return a build_solver that records into the active Recorder, if any."""

    def build_solver(plan, *args, **kwargs):
        solver = build(plan, *args, **kwargs)
        if not _ACTIVE:
            return solver
        return RecordingSolver(solver, plan, _ACTIVE[-1], backend)

    build_solver.recording_original = build
    return build_solver


def _frontends():
    import pivopt.numpy as numpy_frontend
    import pivopt.torch as torch_frontend

    return ((numpy_frontend, "numpy"), (torch_frontend, "torch"))


def install() -> None:
    """Wrap the frontends once; the wrappers pass through when nothing records."""
    for module, backend in _frontends():
        if not hasattr(module.build_solver, "recording_original"):
            module.build_solver = _wrap(module.build_solver, backend)


class Recorder:
    """Collect every planner call made while the context is open."""

    def __init__(self):
        install()
        self.calls: list[dict] = []

    @staticmethod
    def patch(module) -> None:
        """Rebind a ``build_solver`` that a module imported before install()."""
        current = getattr(module, "build_solver", None)
        for frontend, _ in _frontends():
            if current is frontend.build_solver.recording_original:
                module.build_solver = frontend.build_solver

    def __enter__(self):
        _ACTIVE.append(self.calls)
        return self

    def __exit__(self, *exc):
        _ACTIVE.remove(self.calls)


def has_same_point(call) -> bool:
    names = {item.name for item in call["plan"].requirements().inputs}
    return "initial" in names and hasattr(call["plan"].spec.forward, "settings")


def stopped_plan(plan):
    """Rebuild a plan whose optimizer stops at its starting point.

    The gradient tolerance is one that every point satisfies, so a solve makes
    zero iterations and returns the objective, trajectory and stationarity at
    its ``initial`` input, independent of the optimizer path.
    """
    from pivopt import problem as p

    forward = plan.spec.forward
    settings = dataclasses.replace(
        forward.settings, gradient_tolerance=SAME_POINT_TOLERANCE
    )
    spec = dataclasses.replace(
        plan.spec, forward=dataclasses.replace(forward, settings=settings)
    )
    return p.build_plan(spec, plan.shape)


def same_point_inputs(call, plan, point: np.ndarray):
    """The call's inputs with ``initial`` replaced by a decision point."""
    arrays = input_arrays(call["plan"], call["inputs"])
    arrays["initial"] = np.asarray(point, dtype=np.float64)
    return plan.inputs(**arrays)


def same_point(call, point: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate a call's problem at a decision point with an immediate stop."""
    from pivopt import numpy as cpu

    stopped = stopped_plan(call["plan"])
    solver = cpu.build_solver(stopped, dtype=np.float64)
    result = solver.solve(same_point_inputs(call, stopped, point))
    out: dict[str, np.ndarray] = {}
    flatten("same_point", result, out)
    return out


def same_point_pullback(solve_call, point: np.ndarray, seeds) -> dict[str, np.ndarray]:
    """Pull seeds back from a zero-iteration solve at a decision point.

    The State comes from ``solve_with_state`` on the stopped plan, started at
    ``point``. A FiniteResponse ``Resolve`` correction re-minimizes with the
    forward settings, so on the stopped plan it also makes zero iterations and
    this record covers its linear predictor, not its correction path.
    """
    from pivopt import numpy as cpu

    stopped = stopped_plan(solve_call["plan"])
    solver = cpu.build_solver(stopped, dtype=np.float64, backward=True)
    _, state = solver.solve_with_state(same_point_inputs(solve_call, stopped, point))
    out: dict[str, np.ndarray] = {}
    flatten("same_point", solver.pullback(state, seeds), out)
    return out


def result_seeds(source) -> object:
    """Float64 ResultSeeds from a seeds record or from ``{field: array}``."""
    from pivopt import problem as p

    def field(name):
        value = source[name] if isinstance(source, Mapping) else getattr(source, name)
        return to_numpy(value).astype(np.float64)

    return p.ResultSeeds(
        **{item.name: field(item.name) for item in dataclasses.fields(p.ResultSeeds)}
    )


def state_source(calls: list[dict], call) -> int:
    """Index of the ``solve_with_state`` call that produced a pullback's State."""
    for index, other in enumerate(calls):
        if other["kind"] == "solve_with_state" and other["state"] is call["state"]:
            return index
    raise AssertionError("a pullback used a State that no recorded call produced")


def call_prefix(calls: list[dict], index: int) -> str:
    return f"call{index:02d}.{calls[index]['kind']}"


def record_calls(calls: list[dict], reference=None) -> dict[str, np.ndarray]:
    """Flatten recorded calls into named arrays.

    Without ``reference`` the same-point records start at each call's own
    returned point and reuse its own seeds; ``generate_reference.py`` stores
    them. With ``reference`` (the stored arrays) they start at the stored
    point and use the stored seeds, so both pins evaluate the same point.
    Calls that have no stored counterpart get no same-point record.
    """
    out: dict[str, np.ndarray] = {}
    for index, call in enumerate(calls):
        prefix = call_prefix(calls, index)
        out[f"{prefix}.backend"] = np.asarray(call["backend"])
        names = [item.name for item in call["plan"].requirements().inputs]
        out[f"{prefix}.input_names"] = np.asarray(names)
        flatten(f"{prefix}.spec", call["plan"].spec, out, record_types=True)
        flatten(f"{prefix}.shape", call["plan"].shape, out, record_types=True)
        if "inputs" in call:
            for name, value in input_arrays(call["plan"], call["inputs"]).items():
                out[f"{prefix}.inputs.{name}"] = value
            flatten(f"{prefix}.result", call["result"], out)
            point_key = f"{prefix}.result.x"
            if not has_same_point(call):
                continue
            if reference is None:
                point = to_numpy(call["result"].x)
            elif point_key in reference:
                point = reference[point_key]
            else:
                continue
            for name, value in same_point(call, point).items():
                out[f"{prefix}.{name}"] = value
        else:
            source = state_source(calls, call)
            out[f"{prefix}.state_call"] = np.asarray(source)
            flatten(f"{prefix}.seeds", call["seeds"], out)
            flatten(f"{prefix}.pullback", call["result"], out)
            if not has_same_point(calls[source]):
                continue
            point_key = f"{call_prefix(calls, source)}.result.x"
            if reference is None:
                point = to_numpy(calls[source]["result"].x)
                seeds = result_seeds(call["seeds"])
            elif point_key in reference:
                point = reference[point_key]
                stored = {
                    key[len(f"{prefix}.seeds.") :]: reference[key]
                    for key in reference
                    if key.startswith(f"{prefix}.seeds.")
                }
                seeds = result_seeds(stored)
            else:
                continue
            pulled = same_point_pullback(calls[source], point, seeds)
            for name, value in pulled.items():
                out[f"{prefix}.{name}"] = value
    return out


@contextlib.contextmanager
def mujoco_demos():
    """Put the MuJoCo demos first on ``sys.path`` and restore it afterwards.

    The demos share module names with top-level examples (``arm``,
    ``planar_base``, ``unicycle``) and import their helpers by bare name, so
    the path and every module loaded from the demo directory are removed again.
    """
    path = list(sys.path)
    modules = set(sys.modules)
    sys.path.insert(0, str(MUJOCO))
    try:
        yield
    finally:
        sys.path[:] = path
        for name in set(sys.modules) - modules:
            source = getattr(sys.modules[name], "__file__", None) or ""
            if Path(source).resolve().is_relative_to(MUJOCO):
                del sys.modules[name]


def load_mujoco(name: str):
    """Load one demo; call it inside ``mujoco_demos()``."""
    spec = importlib.util.spec_from_file_location(
        "mujoco_demo_" + name, MUJOCO / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Scenarios ---------------------------------------------------------------------
# Each scenario returns extra values computed by the example itself; the planner
# calls are captured by the recorder.


def case_arm(recorder, mode):
    from pivopt import numpy as cpu

    arm = importlib.import_module("arm")
    plan, inputs = arm.make_problem(constraints=mode)
    cpu.build_solver(plan, dtype=np.float64).solve(inputs)
    return {}


def case_mujoco_arm(recorder):
    with mujoco_demos():
        demo = load_mujoco("arm").Arm()
        return {"reference_checks": dict(demo.reference_checks)}


def case_mujoco_unicycle(recorder):
    with mujoco_demos():
        demo = load_mujoco("unicycle").Unicycle()
        return {"reference_checks": dict(getattr(demo, "reference_checks", {}))}


def case_gate_certified(recorder, mode):
    import torch

    study = importlib.import_module("cpu_gate_certified_training")
    torch.set_num_threads(1)
    task = study.CertifiedGateTask(2)
    batch = task.sample(torch.Generator().manual_seed(123))
    torch.manual_seed(9)
    model = study.LearningPolicy(task.features, task.outputs, geometry=task.geometry)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    solver = study.cpu.build_solver(task.plans[mode], dtype=np.float64, backward=True)
    row = study.training_step(
        task, solver, model, optimizer, batch, mode=mode, task_scale=1.0
    )
    return {"row": {key: row[key] for key in sorted(row) if key != "mode"}}


def case_gate_training(recorder, mode):
    import torch

    gate = importlib.import_module("cpu_gate_training")
    recorder.patch(gate)
    torch.manual_seed(7)
    scenes = gate.sample_scenes(2, generator=torch.Generator().manual_seed(7))
    policy = gate.GatePolicy().double()
    with torch.no_grad():
        decoded = gate.decode_action(policy(scenes.features()), scenes)
        setup = gate.prepare_problem(2, constraints=mode)
        setup.solver.solve(setup.bind(scenes, decoded))
    return {}


def case_gate_probe(recorder, mode):
    probe = importlib.import_module("cpu_gate_probe")
    recorder.patch(probe.T)
    calibration = probe.T.task_calibrations(probe.T.GATE2)[0]
    slacks = np.asarray((-0.02, 0.0, 0.01, 0.03, 0.05, 0.08, 0.12))
    boundaries = np.linspace(0.10, 0.60, 11)
    return {
        "native": probe.native_profile_samples(calibration, mode, slacks),
        "toy": probe.toy_sensitivity(calibration, mode, boundaries),
    }


CASES = {
    "arm_hinge": (case_arm, "hinge"),
    "arm_relaxed_barrier": (case_arm, "relaxed-barrier"),
    "mujoco_arm": (case_mujoco_arm, None),
    "mujoco_unicycle": (case_mujoco_unicycle, None),
    "gate_certified_A": (case_gate_certified, "A"),
    "gate_certified_RS": (case_gate_certified, "RS"),
    "gate_training_hinge": (case_gate_training, "hinge"),
    "gate_training_relaxed_barrier": (case_gate_training, "relaxed-barrier"),
    "gate_probe_hinge": (case_gate_probe, "hinge"),
    "gate_probe_relaxed_barrier": (case_gate_probe, "relaxed-barrier"),
}


def run_case(name: str, reference=None) -> dict[str, np.ndarray]:
    """Run one scenario and flatten its calls; see ``record_calls``."""
    function, mode = CASES[name]
    with Recorder() as recorder:
        extra = function(recorder) if mode is None else function(recorder, mode)
    # Record after unpatching, so the same-point solves are not recorded as calls.
    arrays = record_calls(recorder.calls, reference)
    flatten("extra", extra, arrays)
    return arrays
