"""The shared fixed-step simulation loop and native MuJoCo overlays."""

import argparse
import json
import shlex
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
BLUE = (0.12, 0.45, 1.0, 1.0)
ORANGE = (1.0, 0.45, 0.06, 1.0)


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--headless", action="store_true", help="run physics without opening a viewer"
    )
    args = parser.parse_args()
    if sys.platform == "darwin" and not args.headless:
        from mujoco import viewer as mujoco_viewer

        # MuJoCo 3.9's launcher initializes this dispatcher on macOS.
        if mujoco_viewer._MJPYTHON is None:
            command = shlex.join(
                [str(Path(sys.executable).with_name("mjpython")), *sys.argv]
            )
            parser.error(
                "interactive viewing on macOS requires MuJoCo's mjpython launcher "
                f"to render on the main thread. Run:\n  {command}\n"
                "Or add --headless to run without a window."
            )
    return args


def line(scene, start, end, color, width=0.008):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        color,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
    geom.emission = 0.35
    scene.ngeom += 1


def overlay(scene, reference, actual, target, boxes, goal):
    """Render-only geoms: they do not enter dynamics or collision detection."""
    for points, color in ((reference, BLUE), (actual, ORANGE)):
        for start, end in zip(points[:-1], points[1:], strict=True):
            if np.linalg.norm(end - start) > 1e-8:
                line(scene, start, end, color)
    for lower, upper in boxes:
        vertices = np.array(list(product(*zip(lower, upper, strict=True))))
        for i in range(8):
            for axis in (1, 2, 4):
                j = i ^ axis
                if i < j:
                    line(
                        scene, vertices[i], vertices[j], (0.4, 0.65, 0.45, 0.45), 0.003
                    )
    for point, color, radius in (
        (target, BLUE, 0.035),
        (goal, (0.2, 0.9, 0.3, 1), 0.045),
    ):
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [radius, 0, 0],
            point,
            np.eye(3).ravel(),
            color,
        )
        scene.ngeom += 1


def camera(demo, reference):
    """Fit the robot, path and planning boxes in the initial view."""
    bounds = np.vstack((reference, demo.data.xpos[1:], *demo.boxes))
    result = mujoco.MjvCamera()
    result.lookat[:] = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    result.distance = max(2.0, 1.6 * np.linalg.norm(np.ptp(bounds, axis=0)))
    result.azimuth, result.elevation = 125, -30
    return result


@contextmanager
def native_viewer(model, data):
    from mujoco import viewer as mujoco_viewer

    before = set(threading.enumerate())
    handle = mujoco_viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    )
    viewer_threads = set(threading.enumerate()) - before
    try:
        with handle:
            yield handle
    finally:
        # Outside macOS, close() only requests exit from launch_passive's
        # rendering/loading threads. Wait before Python's atexit glfw.terminate
        # to avoid destroying a drawable still used by the viewer (NVIDIA/GLX).
        # On macOS, mjpython owns rendering on the native main thread; Python
        # may expose it as a dummy thread, which cannot be joined.
        if sys.platform != "darwin":
            for thread in viewer_threads:
                thread.join(timeout=5)


def run(demo, *, headless=False):
    """The demos own model, control, observation and reference geometry."""
    model, data, trajectory = demo.model, demo.data, demo.trajectory
    reference = np.stack(
        [demo.reference_position(t) for t in np.linspace(0, trajectory.duration, 240)]
    )
    if headless:
        context = nullcontext(None)
    else:
        context = native_viewer(model, data)
    errors, actual = {}, []
    end_errors = None
    saturated, control_calls = 0, 0
    dt = model.opt.timestep
    stride = round(demo.control_period / dt)
    assert stride >= 1 and np.isclose(stride * dt, demo.control_period)
    next_frame = 0.0
    step = 0
    wall_start = time.monotonic()
    with context as viewer:
        if viewer is not None:
            view = camera(demo, reference)
            with viewer.lock():
                viewer.cam.lookat[:] = view.lookat
                viewer.cam.distance = view.distance
                viewer.cam.azimuth, viewer.cam.elevation = view.azimuth, view.elevation
        while data.time < trajectory.duration + demo.settle_time - 0.5 * dt:
            if viewer is not None and not viewer.is_running():
                break
            if step % stride == 0:
                demo.control(data.time)
                control_calls += 1
                saturated += int(np.any(demo.saturated))
            mujoco.mj_step(model, data)
            # mj_step integrates qpos after computing poses; refresh before observation/control.
            mujoco.mj_forward(model, data)
            position, target, sample_errors = demo.observe(data.time)
            if not np.isfinite(np.r_[data.qpos, data.qvel, data.ctrl]).all():
                raise RuntimeError("nonfinite simulation state")
            if data.time <= trajectory.duration + 0.5 * dt:
                for name, value in sample_errors.items():
                    errors.setdefault(name, []).append(float(np.linalg.norm(value)))
            if end_errors is None and data.time >= trajectory.duration - 0.5 * dt:
                end_errors = {
                    name: float(np.linalg.norm(value))
                    for name, value in sample_errors.items()
                }
            if data.time >= next_frame:
                actual.append(position.copy())
                next_frame += 1 / 60
                if viewer is not None:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        overlay(
                            viewer.user_scn,
                            reference,
                            np.asarray(actual),
                            target,
                            demo.boxes,
                            demo.goal,
                        )
                    viewer.sync()
                    time.sleep(max(0, data.time - (time.monotonic() - wall_start)))
            step += 1
    final = {
        name: float(np.linalg.norm(value))
        for name, value in demo.observe(data.time)[2].items()
    }
    report = {
        "robot": demo.name,
        "trajectory_seconds": trajectory.duration,
        "simulated_seconds": data.time,
        "reference_checks": demo.reference_checks,
        "tracking": {
            name: {
                "rms": float(np.sqrt(np.mean(np.square(values)))),
                "max": max(values),
            }
            for name, values in errors.items()
        },
        "at_trajectory_end": end_errors,
        "after_settling": final,
        "actuator_saturation_fraction": saturated / max(control_calls, 1),
    }
    if hasattr(demo, "allocation_residual_max"):
        report["allocation_residual_max_N_Nm"] = demo.allocation_residual_max.tolist()
    if hasattr(demo, "terminal_status"):
        report["terminal_status"] = demo.terminal_status()
    print(json.dumps(report, indent=2))
    return report
