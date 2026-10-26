# RSS27_exp

Independent experiments for RSS 2027 research on trajectory-planning
sensitivities. This initial scaffold carries the current Python and MuJoCo
examples from [pivopt](https://github.com/zhangbaozhe/pivopt), with their
mathematics and solver settings preserved. It provides runnable starting points;
it does not yet implement the paper's complete G/D/T protocol.

## Install

Use Python 3.12, [uv](https://docs.astral.sh/uv/), a C++20 compiler, and SSH access
to `git@github.com:zhangbaozhe/pivopt.git`. Run from this repository root:

```bash
PIVOPT_BUILD_MODE=full CMAKE_BUILD_PARALLEL_LEVEL=8 uv sync --locked
```

`pyproject.toml` pins pivopt to commit
`94f7c1cd249d4c4fd1d3d64423971e8c786f551a`; `uv.lock` pins the resolved Python
environment. The first install builds the full CPU library with backward,
NumPy, Torch and JAX support. CMake and Ninja are obtained through pivopt's
isolated build dependencies; the build also downloads pinned C++ dependencies.
Lower `CMAKE_BUILD_PARALLEL_LEVEL` on machines with limited memory.

MuJoCo **3.9.0 is a direct dependency**. NumPy, CPU Torch, JAX, Matplotlib and
SciPy are also declared directly because the copied scripts use them. The
PyTorch CPU index is selected on Linux x86_64; macOS uses its normal CPU wheel.
The development group supplies pytest and Ruff. This is a script-based project,
so there is no additional Python package to install from this repository.

## First runs

```bash
uv run --no-sync python examples/numpy_solve.py
uv run --no-sync python examples/torch_solve.py
JAX_PLATFORMS=cpu uv run --no-sync python examples/jax_solve.py
uv run --no-sync python examples/cpu_training.py --framework torch --steps 3
```

The last command exercises actual implicit backward and parameter updates.
To check both LinearResponse and FiniteResponse on the certified gate task:

```bash
uv run --no-sync pytest -q tests/test_cpu_gate_certified_training_example.py
```

## Examples

| Entry point under `examples/` | Purpose |
| --- | --- |
| `numpy_solve.py`, `torch_solve.py`, `jax_solve.py` | Basic public frontend solves |
| `cpu_explain.py`, `cpu_plan_shapes.py`, `cpu_xyz_yaw.py` | Problem structure, shapes and xyz+yaw |
| `arm.py`, `planar_base.py`, `unicycle.py` | Robot-specific planning examples |
| `cpu_training.py` | Small Torch/JAX learning and diagnostic example |
| `cpu_polytope_training.py` | Polytope policy training |
| `cpu_gate_training.py` | Gate geometry warm-up and policy training |
| `cpu_gate_probe.py` | Calibration, paired updates and mechanism probes |
| `cpu_gate_robust_search.py`, `cpu_gate_robust_report.py` | Robust-gate search and analysis |
| `cpu_gate_certified_training.py` | Certified-gate Linear/Finite response study |
| `mujoco/` | Optimized-trajectory tracking for four robot models |

Run scripts directly from the repository root so their existing local imports
and relative output paths resolve consistently. Use `--help` on a study driver
to inspect its subcommands. Historical studies may require frozen datasets and
checkpoints that are not included here; copying their driver does not reproduce
their historical results.

## MuJoCo

All four examples simulate physical tracking without opening a viewer:

```bash
uv run --no-sync python examples/mujoco/planar_base.py --headless
uv run --no-sync python examples/mujoco/arm.py --headless
uv run --no-sync python examples/mujoco/unicycle.py --headless
uv run --no-sync python examples/mujoco/quadrotor.py --headless
```

For interactive viewing on Linux, omit `--headless`. On macOS, use the launcher
provided by MuJoCo, for example:

```bash
.venv/bin/mjpython examples/mujoco/quadrotor.py
```

See [the MuJoCo guide](examples/mujoco/README.md) for model units, controllers and
reported metrics. Keep `examples/mujoco/` without an `__init__.py` so it does not
shadow the installed simulator. Model licenses and pinned asset hashes are in
[`examples/mujoco/assets/`](examples/mujoco/assets/SOURCES.md).

## Outputs and verification

Generated runs go under the ignored `artifacts/` directory. Defaults are
`artifacts/gate2/`, `artifacts/sgate3/` and `artifacts/cpu_polytope_training/`;
drivers with a required output option should receive an explicit path there.
Use a new run directory for new experiments. Keep configurations, seeds,
checkpoints, failures and raw metrics together; large historical outputs are
not part of this initial scaffold.

The migrated tests cover both constraint profiles, Torch/JAX learning,
certified-gate updates, independent kinematics and the four headless rollouts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  uv run --no-sync pytest -q
uv run --no-sync ruff check examples tests
```

The environment is CPU-only. CUDA transport and interactive viewers are outside
the initial verification scope. [The migration report](docs/scaffold_2026-10-26.md)
records the actual checks and any limitations. [The source manifest](docs/upstream_examples.json)
identifies every imported file and its original hash.

## Repository boundaries

pivopt owns the numerical library and its core tests. RSS27_exp owns these
experiment drivers, their evaluation and results. The sibling
`ResearchPaper-sensitivities` repository owns the paper and research protocol;
it is not a runtime dependency. The initial copy excludes pivopt's maintainer
tutorials, retired adaptive scripts and build/experiment artifacts.
