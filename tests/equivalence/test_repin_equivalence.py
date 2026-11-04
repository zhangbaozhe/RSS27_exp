"""Migrated legacy configurations against the records stored at pivopt 94f7c1cd.

Each scenario in ``cases.CASES`` runs the migrated examples at the installed
pin and is compared, call by call, with the reference that
``generate_reference.py`` stored at 94f7c1cd. The criteria below were fixed
before the first run at the new pin.

- **Calls.** The same sequence of call kinds and backends.
- **Declarations.** Every normalized spec field and plan shape is bitwise
  equal, including all optimizer settings and backward checks, except the time
  record: ``FixedTotalTime`` must have become ``AllocatedTime`` with
  ``FreeFractions(logit_penalty=0)``, the same minimum piece duration and the
  same total. A total that was an input keeps its input; a constant total
  becomes one new input whose every lane equals that constant. The input
  declaration and input shape lists are compared by input name, because that
  new input shifts their positions: stored entries are unchanged and in order,
  and the new input has no gradient and one value per lane.
- **Inputs.** Bitwise equal on every shared name.
- **Status.** Lane status, cause and publication codes are identical.
- **Same point.** Zero-iteration solves and pullbacks, started at the stored
  point with the stored seeds, keep that point bitwise and make zero
  iterations; their float outputs lie inside the envelope.
- **Endpoints.** Compared with the envelope on lanes where both pins succeeded
  and reached the stationarity qualification line; other lanes are reported.
- **Example values.** Compared with the envelope when every solve lane of the
  scenario qualified; otherwise reported.
- **Work.** Iteration and evaluation counts and the optimizer stopping reason
  are reported, never compared. Wall-clock times are ignored.

The envelope is pivopt's approved endpoint envelope for complete problems:
``|new - old| <= atol + rtol * |old|`` for arrays and
``|new - old| <= cost * max(1, |old|)`` for objective values. NaN and infinite
entries must occur at the same positions with the same values. pivopt records
only pure energy and quadratic problems as bitwise unchanged across the pins;
every scenario here has inequality costs, so every output uses the envelope.

For a FiniteResponse with a ``Resolve`` correction, the same-point pullback
covers the linear predictor only: the correction re-minimizes with the forward
settings, which the stopped plan replaces.

Set ``REPIN_REPORT_DIR`` to write one JSON report per scenario.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from equivalence import cases

FIXTURES = cases.ROOT / "tests" / "fixtures" / "repin_94f7c1cd"

ENVELOPE = {
    np.dtype(np.float64): {"rtol": 2e-6, "atol": 2e-7, "cost": 1e-8},
    np.dtype(np.float32): {"rtol": 8e-4, "atol": 4e-5, "cost": 2e-5},
}
QUALIFICATION = 1e-7
WORK = {
    "iterations",
    "objective_evaluations",
    "optimizer_status",
    "hard_selector_qp_calls",
    "hard_selector_qp_iterations",
    "hard_selector_fallback_count",
    "hard_selector_max_candidate_count",
    "response_correction_steps",
    "forward_objective_evaluations",
}
IGNORED = {"wall_seconds"}
NEW_FIELDS = {"hessian_null_dimension"}
TIME = ".spec.trajectory.time"


def leaf(key: str) -> str:
    return re.sub(r"\[\d+\]$", "", key.rsplit(".", 1)[-1])


def is_float(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.floating)


def exact(old: np.ndarray, new: np.ndarray) -> tuple[bool, str]:
    """Bitwise equality; NaN entries only need the same positions."""
    if old.dtype != new.dtype or old.shape != new.shape:
        return False, f"{old.dtype}{old.shape} -> {new.dtype}{new.shape}"
    if is_float(old):
        nan = np.isnan(old)
        if not np.array_equal(nan, np.isnan(new)):
            return False, "NaN positions differ"
        same = old[~nan].tobytes() == new[~nan].tobytes()
        if same:
            return True, ""
        return False, f"max abs change {float(np.max(np.abs(new - old)[~nan])):.3g}"
    return bool(old.tobytes() == new.tobytes()), ""


def envelope_ratio(old: np.ndarray, new: np.ndarray, *, cost: bool) -> float:
    """Largest change relative to the envelope; at most 1 passes."""
    if old.dtype != new.dtype or old.shape != new.shape:
        return float("inf")
    finite = np.isfinite(old)
    if not np.array_equal(finite, np.isfinite(new)):
        return float("inf")
    if not np.array_equal(old[~finite], new[~finite], equal_nan=True):
        return float("inf")
    if not finite.any():
        return 0.0
    tolerance = ENVELOPE[old.dtype]
    a, b = old[finite].astype(np.float64), new[finite].astype(np.float64)
    if cost:
        scale = tolerance["cost"] * np.maximum(1.0, np.abs(a))
    else:
        scale = tolerance["atol"] + tolerance["rtol"] * np.abs(a)
    return float(np.max(np.abs(b - a) / scale))


class Comparison:
    """Collects one record per checked key and the failures among them."""

    def __init__(self, name: str):
        self.name = name
        self.records: list[dict] = []
        self.qualified: dict[str, list[bool]] = {}

    def add(self, key: str, rule: str, passed, detail="") -> None:
        """Record a check; ``passed`` is None for a reported value."""
        passed = None if passed is None else bool(passed)
        self.records.append(
            {"key": key, "rule": rule, "passed": passed, "detail": detail}
        )

    def failures(self) -> list[dict]:
        return [record for record in self.records if record["passed"] is False]

    def exact(self, key, old, new) -> None:
        passed, detail = exact(old, new)
        self.add(key, "exact", passed, detail)

    def envelope(self, key, old, new, *, gate: bool, cost: bool = False) -> None:
        """Envelope check for floats, exact check for other values.

        Without ``gate`` the same measure is reported and never fails.
        """
        if not is_float(old):
            passed, detail = exact(old, new)
            if gate:
                self.add(key, "exact", passed, detail)
            else:
                self.add(
                    key, "report", None, "equal" if passed else detail or "differs"
                )
            return
        ratio = envelope_ratio(old, new, cost=cost)
        self.add(
            key, "envelope" if gate else "report", ratio <= 1 if gate else None, ratio
        )

    def summary(self) -> dict:
        rules: dict[str, dict[str, int]] = {}
        for record in self.records:
            counts = rules.setdefault(
                record["rule"], {"passed": 0, "failed": 0, "reported": 0}
            )
            if record["passed"] is None:
                counts["reported"] += 1
            else:
                counts["passed" if record["passed"] else "failed"] += 1
        return rules


def calls_of(arrays: dict) -> list[str]:
    """Call prefixes such as ``call00.solve`` in call order."""
    prefixes = {
        ".".join(key.split(".")[:2]) for key in arrays if key.startswith("call")
    }
    return sorted(prefixes)


def subtree(arrays: dict, prefix: str) -> dict:
    return {
        key[len(prefix) :]: value
        for key, value in arrays.items()
        if key.startswith(prefix)
    }


def lane_mask(old: dict, new: dict, call: str) -> np.ndarray | None:
    """Lanes of a solve call that succeeded and qualified on both pins."""
    status = f"{call}.result.status"
    stationarity = f"{call}.result.diagnostics.stationarity"
    if not all(
        key in arrays for key in (status, stationarity) for arrays in (old, new)
    ):
        return None
    if old[status].shape != new[status].shape:
        return None
    return (
        (old[status] == 0)
        & (new[status] == 0)
        & (old[stationarity] <= QUALIFICATION)
        & (new[stationarity] <= QUALIFICATION)
    )


def compare_time(check: Comparison, call: str, old: dict, new: dict) -> str | None:
    """Check the FixedTotalTime -> AllocatedTime mapping; return a new input name."""
    before = subtree(old, call + TIME)
    after = subtree(new, call + TIME)
    key = call + TIME
    check.add(
        key + ".@type",
        "time",
        before.get(".@type") == np.asarray("FixedTotalTime")
        and after.get(".@type") == np.asarray("AllocatedTime"),
        f"{before.get('.@type')} -> {after.get('.@type')}",
    )
    minimum_old = subtree(before, ".minimum_piece_duration")
    minimum_new = subtree(after, ".minimum_piece_duration")
    same_minimum = minimum_old.keys() == minimum_new.keys() and all(
        exact(minimum_old[k], minimum_new[k])[0] for k in minimum_old
    )
    check.add(
        key + ".minimum_piece_duration", "time", same_minimum and bool(minimum_old)
    )
    fractions = subtree(after, ".fractions")
    penalty = [
        v
        for k, v in fractions.items()
        if k.startswith(".logit_penalty") and not k.endswith("@type")
    ]
    check.add(
        key + ".fractions",
        "time",
        fractions.get(".@type") == np.asarray("FreeFractions")
        and bool(penalty)
        and all(np.issubdtype(v.dtype, np.number) and np.all(v == 0) for v in penalty),
        {k: str(v) for k, v in fractions.items()},
    )
    total_old = subtree(before, ".total_duration")
    total_new = subtree(after, ".total")
    if total_new.get(".@type") != np.asarray("InputRef"):
        check.add(
            key + ".total", "time", False, f"new total is {total_new.get('.@type')}"
        )
        return None
    name = str(total_new[".name"])
    if total_old.get(".@type") == np.asarray("InputRef"):
        check.add(
            key + ".total",
            "time",
            str(total_old[".name"]) == name,
            f"{total_old['.name']} -> {name}",
        )
        return None
    if total_old.get(".@type") != np.asarray("Const"):
        check.add(
            key + ".total", "time", False, f"old total is {total_old.get('.@type')}"
        )
        return None
    constant = float(total_old[".value"])
    values = new.get(f"{call}.inputs.{name}")
    old_names = old[f"{call}.input_names"].tolist()
    passed = (
        name not in old_names
        and values is not None
        and values.dtype == np.float64
        and bool(np.all(values == constant))
    )
    check.add(key + ".total", "time", passed, f"constant {constant} -> input {name}")
    return name


INPUT_LISTS = {".spec.inputs[": ".name", ".shape.input_shapes[": ".input.name"}


def input_entries(arrays: dict, prefix: str, name_field: str) -> dict[str, dict]:
    """Entries of a flattened per-input list, keyed by input name, in order."""
    groups: dict[int, dict] = {}
    for key, value in arrays.items():
        if key.startswith(prefix):
            position, field = key[len(prefix) :].split("]", 1)
            groups.setdefault(int(position), {})[field] = value
    return {str(fields[name_field]): fields for _, fields in sorted(groups.items())}


def compare_input_lists(check: Comparison, call: str, old: dict, new: dict, added):
    """Compare input declarations and shapes by input name.

    Every stored entry must be present, unchanged and in the same order. The
    only additional entry is the input that replaced a constant total: a
    declaration without gradient and a shape of one value per lane.
    """
    batch = int(old[f"{call}.shape.batch_size"])
    expected = {
        ".spec.inputs[": {
            ".@type": "InputDecl",
            ".name": added,
            ".differentiate": False,
        },
        ".shape.input_shapes[": {
            ".@type": "InputShape",
            ".input.@type": "InputRef",
            ".input.name": added,
            ".shape[0]": batch,
        },
    }
    for item, name_field in INPUT_LISTS.items():
        key = call + item.rstrip("[")
        before = input_entries(old, call + item, name_field)
        after = input_entries(new, call + item, name_field)
        order = [name for name in after if name != added] == list(before)
        same = all(
            name in after
            and fields.keys() == after[name].keys()
            and all(
                exact(value, after[name][field])[0] for field, value in fields.items()
            )
            for name, fields in before.items()
        )
        check.add(
            key, "inputs by name", order and same, f"{list(before)} -> {list(after)}"
        )
        if added is None:
            continue
        fields = after.get(added, {})
        check.add(
            f"{key}.{added}",
            "inputs by name",
            fields.keys() == expected[item].keys()
            and all(fields[field] == value for field, value in expected[item].items()),
            {field: str(value) for field, value in fields.items()},
        )


def compare_call(
    check: Comparison, call: str, old: dict, new: dict, masks: dict
) -> None:
    kind = call.split(".")[1]
    check.exact(f"{call}.backend", old[f"{call}.backend"], new[f"{call}.backend"])
    added = compare_time(check, call, old, new)
    old_names = old[f"{call}.input_names"].tolist()
    new_names = new[f"{call}.input_names"].tolist()
    check.add(
        f"{call}.input_names",
        "names",
        [n for n in new_names if n != added] == old_names,
        f"{old_names} -> {new_names}",
    )
    compare_input_lists(check, call, old, new, added)
    lists = tuple(call + item for item in INPUT_LISTS)
    old_keys = {k for k in old if k.startswith(call + ".") and not k.startswith(lists)}
    new_keys = {k for k in new if k.startswith(call + ".") and not k.startswith(lists)}
    for key in sorted(old_keys - new_keys):
        if not key.startswith(call + TIME):
            check.add(key, "missing", False)
    for key in sorted(new_keys - old_keys):
        allowed = key.startswith(call + TIME) or key == f"{call}.inputs.{added}"
        if leaf(key) in NEW_FIELDS:
            check.add(key, "new field", None, str(new[key].tolist()))
        elif not allowed:
            check.add(key, "unexpected", False)
    source = None
    if kind == "pullback":
        source = int(old[f"{call}.state_call"])
        source = calls_of(old)[source] if source < len(calls_of(old)) else None
    mask = masks.get(source if kind == "pullback" else call)
    for key in sorted(old_keys & new_keys):
        section = key[len(call) + 1 :].split(".")[0]
        a, b = old[key], new[key]
        name = leaf(key)
        cost = name == "cost" and section in {"result", "same_point"}
        if key.startswith(call + TIME) or section in {"backend", "input_names"}:
            continue
        if name in IGNORED:
            continue
        if section in {"spec", "shape", "inputs", "state_call"}:
            check.exact(key, a, b)
        elif name in WORK and not (section == "same_point" and name == "iterations"):
            check.add(key, "work", None, {"old": a.tolist(), "new": b.tolist()})
        elif section == "same_point":
            if name == "iterations":
                check.add(key, "zero", np.all(a == 0) and np.all(b == 0), b.tolist())
            elif name == "x":
                start = old[f"{call}.result.x"]
                check.add(key, "start", exact(start, b)[0] and exact(a, b)[0])
            else:
                check.envelope(key, a, b, gate=True, cost=cost)
        elif not is_float(a):
            check.exact(key, a, b)
        elif mask is None or a.ndim == 0 or a.shape[:1] != mask.shape:
            check.envelope(key, a, b, gate=False, cost=cost)
        else:
            check.envelope(key, a[mask], b[mask], gate=bool(mask.any()), cost=cost)
            if not mask.all():
                check.envelope(
                    key + "[unqualified lanes]",
                    a[~mask],
                    b[~mask],
                    gate=False,
                    cost=cost,
                )


def compare_case(name: str, old: dict, new: dict) -> Comparison:
    check = Comparison(name)
    old_calls, new_calls = calls_of(old), calls_of(new)
    check.add("calls", "calls", old_calls == new_calls, f"{old_calls} -> {new_calls}")
    masks = {}
    for call in old_calls:
        if call in new_calls and not call.endswith(".pullback"):
            mask = lane_mask(old, new, call)
            if mask is not None:
                masks[call] = mask
    for call in old_calls:
        if call in new_calls:
            compare_call(check, call, old, new, masks)
    qualified = bool(masks) and all(mask.all() for mask in masks.values())
    old_extra = {k for k in old if k.startswith("extra")}
    new_extra = {k for k in new if k.startswith("extra")}
    for key in sorted(old_extra ^ new_extra):
        check.add(key, "missing" if key in old_extra else "unexpected", False)
    for key in sorted(old_extra & new_extra):
        if leaf(key) in IGNORED:
            continue
        if leaf(key) in WORK:
            check.add(
                key, "work", None, {"old": old[key].tolist(), "new": new[key].tolist()}
            )
        else:
            check.envelope(key, old[key], new[key], gate=qualified)
    check.qualified = {call: mask.tolist() for call, mask in masks.items()}
    check.add("extra", "example values", None, "compared" if qualified else "reported")
    return check


def write_report(check: Comparison) -> None:
    directory = os.environ.get("REPIN_REPORT_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    report = {
        "case": check.name,
        "qualification": QUALIFICATION,
        "envelope": {str(k): v for k, v in ENVELOPE.items()},
        "qualified_lanes": check.qualified,
        "summary": check.summary(),
        "failures": check.failures(),
        "records": check.records,
    }
    (path / f"{check.name}.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n"
    )


@pytest.mark.parametrize("name", list(cases.CASES))
def test_repin_equivalence(name):
    with np.load(FIXTURES / f"{name}.npz") as stored:
        old = {key: stored[key] for key in stored.files}
    new = cases.run_case(name, reference=old)
    check = compare_case(name, old, new)
    write_report(check)
    failures = check.failures()
    assert not failures, json.dumps(failures[:20], indent=1, default=str)
