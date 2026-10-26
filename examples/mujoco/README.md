# Optimized trajectory tracking in MuJoCo

Each script builds a pivopt problem, optimizes a continuous trajectory once,
then lets a controller track it through MuJoCo dynamics. The examples are
self-contained: they import pivopt, NumPy, MuJoCo and helpers in this directory.

## Run

Use this repository's Python 3.12 environment. MuJoCo is a direct dependency;
install the locked environment from the repository root:

```bash
PIVOPT_BUILD_MODE=full CMAKE_BUILD_PARALLEL_LEVEL=8 uv sync --locked
```

On **macOS**, interactive viewing requires MuJoCo's `mjpython` launcher:

```bash
.venv/bin/mjpython examples/mujoco/planar_base.py
.venv/bin/mjpython examples/mujoco/arm.py
.venv/bin/mjpython examples/mujoco/unicycle.py
.venv/bin/mjpython examples/mujoco/quadrotor.py
```

On **Linux/Windows**, use the regular Python interpreter:

```bash
uv run --no-sync python examples/mujoco/planar_base.py
uv run --no-sync python examples/mujoco/arm.py
uv run --no-sync python examples/mujoco/unicycle.py
uv run --no-sync python examples/mujoco/quadrotor.py
```

Run these commands from the repository root. The viewer opens after optimization
and closes after tracking and settling. Closing the window early stops the run
and prints the partial result. macOS requires rendering on the process's main
thread; `mjpython` arranges this while the simulation runs in Python. A regular
`python` interactive invocation exits with the required command before building
the model or optimizing. This applies to all four scripts, which share the same
viewer helper. See the [MuJoCo passive viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

On **all platforms**, add `--headless` to run the same physics and controller
without a window, using regular Python:

```bash
uv run --no-sync python examples/mujoco/quadrotor.py --headless
```

The directory deliberately has no `__init__.py`: a local top-level `mujoco`
package would interfere with importing the third-party simulator. Scripts are
direct entrypoints rather than a `python -m` package.

## What is displayed

- Blue: the optimized reference path and current reference marker.
- Orange: the robot's actual path. Close tracking makes the two lines overlap.
- Green marker: the scenario goal. Wire boxes: planning corridor/workspace.
- For UR5e, paths and markers refer to `attachment_site`; the workspace bounds
  a 2 cm tip sphere. For X2, trajectory position refers to the center of mass.

All scenarios use third-order MINCO (quintic pieces, minimum jerk), fixed total
time with optimized piece durations, and smooth Hinge inequality penalties.
The sampler uses the result's **piece-local seconds and ascending powers**, and
evaluates derivatives analytically. Physics time selects the reference; viewer
refresh and wall-clock pacing do not change the trajectory's time scale.

| Script | Default scenario | Model and tracking controller |
| --- | --- | --- |
| `planar_base.py` | 8 s planar turn with changing yaw | Example-defined x/y/yaw force-driven base; PD + acceleration feedforward |
| `arm.py` | 6 s motion to a task-space tip position | Menagerie UR5e; joint PD + mass-matrix acceleration feedforward + bias compensation |
| `unicycle.py` | 8 s, 90-degree corner and parking alignment | Example-defined contact-driven differential drive; body-frame pose feedback + wheel-speed PI with anti-windup |
| `quadrotor.py` | 12 s 3D curved path with height/yaw changes | Menagerie Skydio X2; position PD + acceleration feedforward, SO(3) attitude feedback, PX4-style rotor allocation |

Physics runs at 500 Hz. UR5e and X2 control runs at 500 Hz; planar-base and
differential-drive control runs at 100 Hz. The viewer refreshes at 60 Hz.
Planar base and arm hold the final target for 2 s, X2 for 3 s. The differential
drive has a bounded 3 s parking phase, allowing low-speed forward/reverse
correction and turning in place. Its rest headings are explicit; one-sided
trajectory derivatives handle the zero-speed endpoints. The report distinguishes
successful parking from a parking timeout (1.5 cm position, 0.01 rad heading).

## Models, units and limits

SI units and world +z up are used throughout. Unwrapped trajectory yaw is in
radians; feedback errors use the shortest angular difference. Controllers only
write actuator commands during tracking; MuJoCo advances the robot state.

- The planar base's mass, yaw inertia and force/moment bounds are read from
  `assets/planar_base.xml`. They are example-defined simulation parameters.
- The differential drive's wheel radius, track width, speed operating range and
  motor torque bounds are read from `assets/unicycle.xml`. Its 8 rad/s wheel
  speed is a chosen operating range; motor torque is bounded at 1.5 N m.
  Planning wheel signals use **rim linear speed**, so the planner and controller
  explicitly convert using the same wheel radius.
- UR5e is loaded from its upstream MJCF. At runtime its position servos become
  unit-gain torque actuators, preserving the original force bounds (150 N m for
  the first three joints, 28 N m for the wrists), link dynamics and joint ranges.
  The planning chain is built from that same model's zero-configuration frames.
  The 1.5 rad/s planning speed bound is a chosen operating range.
- X2 retains the MJCF's mass, COM, principal inertia, rotor positions, reaction
  moments, fluid properties and four 0–13 N force bounds. The controller rotates
  principal inertia into the body frame. The planning flatness model uses the
  loaded mass and gravity, with zero explicit rotor drag; MuJoCo's generic fluid
  forces remain active. The 2 m/s, 0.6 rad tilt and 2 rad/s body-rate bounds are
  chosen planning operating ranges, not manufacturer limits.
- The imported models' integration timestep is set to 0.002 s. Their source
  files are unchanged; model and adaptation provenance is in `assets/SOURCES.md`.

`px4_allocation.py` references **PX4 v1.17.0** source files and functions in its
comments. It implements pseudo-inverse mixing and sequential desaturation for
Roll/Pitch airmode (`MC_AIRMODE=1`), including yaw margin and final clipping.
The X2 allocation matrix uses moments about its COM. SI body-FLU wrench requests
are converted to normalized FRD requests, then returned to rotor forces in N.
This preserves the physical model's authority while matching PX4's normalized
desaturation logic. PX4 supplies this local allocation algorithm; the tracking
controllers execute locally without a flight-controller runtime.

## Results and verification

The terminal JSON reports dense reference checks, RMS/max Euclidean tracking
errors during the optimized trajectory, errors at its end, errors after settling,
and actuator saturation fraction across all control cycles. X2 also reports
maximum allocation residuals ordered `[thrust, roll, pitch, yaw]` in N/N m.
UR5e joint error is the norm of the six joint errors in radians.

Smooth penalties can leave small residual violations, which are printed rather
than interpreted as feasibility certificates. Dense sampling does not prove
continuous-time feasibility, and a collective thrust bound alone does not
establish per-rotor wrench feasibility. Physical actuator limits are applied
during every control update. These examples demonstrate normal trajectory
tracking with the selected simulation models.

```bash
uv run --no-sync python -m pytest -q tests/test_mujoco_examples.py
uv run --no-sync ruff check examples/mujoco tests/test_mujoco_examples.py
```

Tests cover arbitrary-time polynomial derivatives, planning/MuJoCo UR5e FK,
the X2 actuator wrench mapping at different attitudes, frozen outputs from the
original PX4 C++ method bodies, closed-form X2 normalization, overlay construction, import independence and
all four complete normal physics rollouts. The optional test module skips when
MuJoCo is not installed.
