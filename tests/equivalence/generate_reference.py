"""Generate the old-pin reference records for the re-pin equivalence tests.

Run this only in a tree whose environment is pinned to pivopt 94f7c1cd and
whose examples are the unmigrated ones. Since the re-pin, this repository's
own tree has neither, so export the last old-pin commit and copy this
directory into it, from the repository root:

    git archive a9e6f4a | tar -x -C <old-tree>
    cp tests/equivalence/*.py <old-tree>/tests/equivalence/
    cd <old-tree>
    PIVOPT_BUILD_MODE=full uv sync --locked
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        uv run --no-sync python -m tests.equivalence.generate_reference

It writes one ``.npz`` file per scenario to ``tests/fixtures/repin_94f7c1cd/``
of that tree and a ``manifest.json`` with hashes, versions, the example sources
that ran, and a check that every same-point solve stopped at its starting
point and every same-point pullback reproduced the recorded pullback. Copy the
directory back into this repository.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "examples", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from equivalence import cases  # noqa: E402

PIN = "94f7c1cd249d4c4fd1d3d64423971e8c786f551a"
OUTPUT = ROOT / "tests" / "fixtures" / "repin_94f7c1cd"


def installed_commit() -> str:
    text = importlib.metadata.distribution("pivopt").read_text("direct_url.json")
    return json.loads(text)["vcs_info"]["commit_id"]


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def same_point_check(arrays: dict[str, np.ndarray]) -> dict:
    """Max differences between each call and its same-point repetition."""
    report = {}
    for key in arrays:
        if key.endswith(".same_point.x"):
            call = key[: -len(".same_point.x")]
            report[call] = {
                "x_max_abs_change": max_abs(arrays[key], arrays[f"{call}.result.x"]),
                "cost_max_abs_change": max_abs(
                    arrays[f"{call}.same_point.cost"], arrays[f"{call}.result.cost"]
                ),
                "iterations": arrays[
                    f"{call}.same_point.diagnostics.iterations"
                ].tolist(),
            }
        marker = ".same_point.parameters."
        if marker in key:
            call, name = key.split(marker)
            entry = report.setdefault(call, {"parameters_max_abs_change": {}})
            entry["parameters_max_abs_change"][name] = max_abs(
                arrays[key], arrays[f"{call}.pullback.parameters.{name}"]
            )
    return report


def example_sources() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "examples").rglob("*.py"))
    }


def main() -> None:
    commit = installed_commit()
    if commit != PIN:
        raise SystemExit(f"installed pivopt is {commit}, expected {PIN}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pivopt_commit": commit,
        "pivopt_version": importlib.metadata.version("pivopt"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "mujoco", "jax")
        },
        "command": "uv run --no-sync python -m tests.equivalence.generate_reference",
        "example_sources_sha256": example_sources(),
        "cases": {},
    }
    for name in cases.CASES:
        arrays = cases.run_case(name)
        path = OUTPUT / f"{name}.npz"
        np.savez(path, **arrays)
        statuses = {
            key: arrays[key].tolist()
            for key in arrays
            if key.endswith(".result.status") or key.endswith(".diagnostics.status")
        }
        manifest["cases"][name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": len(arrays),
            "status": statuses,
            "same_point": same_point_check(arrays),
        }
        print(f"{name}: {len(arrays)} arrays, status {statuses}")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
