"""Numerical boundaries and the four normal optimized-trajectory tracking demos."""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Without the optional wheel, examples/mujoco can resolve as a namespace package.
# Requiring the binding's version distinguishes that namespace from the engine.
mujoco = pytest.importorskip("mujoco", minversion="3.9.0")
ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "examples" / "mujoco"


def load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(DEMOS))
    spec = importlib.util.spec_from_file_location(
        "mujoco_demo_" + name, DEMOS / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_piece_local_sampling_matches_polynomial_oracle(monkeypatch):
    trajectory = load("trajectory", monkeypatch).Trajectory
    rng = np.random.default_rng(5)
    coefficients = rng.normal(size=(3, 6, 4))
    durations = np.array((0.4, 1.3, 0.7))
    curve = trajectory(coefficients, durations)
    times = np.array((-0.1, 0, 0.2, 0.4, 1.2, 1.7, 2.1, 2.4, 3.0))
    for order in range(4):
        expected = []
        for time in times:
            time = np.clip(time, 0, durations.sum())
            piece = min(np.searchsorted(np.cumsum(durations), time, side="right"), 2)
            local = time - np.r_[0, np.cumsum(durations)][piece]
            expected.append(
                [
                    np.polynomial.Polynomial(coefficients[piece, :, dim]).deriv(order)(
                        local
                    )
                    for dim in range(4)
                ]
            )
        np.testing.assert_allclose(curve.sample(times, order), expected, atol=3e-13)
        np.testing.assert_allclose(curve.sample(1.2, order), expected[4], atol=3e-13)
    _, velocity, acceleration = curve.reference(3.0)
    np.testing.assert_array_equal(velocity, np.zeros(4))
    np.testing.assert_array_equal(acceleration, np.zeros(4))


def test_ur5e_planning_chain_matches_mujoco(monkeypatch):
    import jax
    from reference_chain import forward

    jax.config.update("jax_enable_x64", True)
    arm = load("arm", monkeypatch)
    model = mujoco.MjModel.from_xml_path(str(DEMOS / "assets" / "ur5e" / "ur5e.xml"))
    chain, data = arm.serial_chain(model), mujoco.MjData(model)
    rng = np.random.default_rng(9)
    for q in rng.uniform(-2, 2, size=(12, 6)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        sphere, tip = forward(chain, q)
        np.testing.assert_allclose(tip, data.site("attachment_site").xpos, atol=2e-12)
        np.testing.assert_allclose(sphere[0], tip, atol=2e-12)


def test_x2_allocation_matches_actual_mujoco_actuator_wrench(monkeypatch):
    allocation = load("px4_allocation", monkeypatch).Allocation
    model = mujoco.MjModel.from_xml_path(str(DEMOS / "assets" / "skydio_x2" / "x2.xml"))
    data = mujoco.MjData(model)
    body = model.body("x2").id
    allocator = allocation(model, body)
    for angle in (0.0, 0.6):
        mujoco.mju_axisAngle2Quat(data.qpos[3:7], np.array((0, 1.0, 0)), angle)
        for index in range(4):
            data.ctrl[:] = 0
            data.ctrl[index] = 1
            mujoco.mj_forward(model, data)
            rotation = data.xmat[body].reshape(3, 3)
            force = rotation.T @ data.qfrc_actuator[:3]
            torque = data.qfrc_actuator[3:6] - np.cross(model.body_ipos[body], force)
            np.testing.assert_allclose(
                np.r_[torque, force[2]], allocator.effectiveness[:, index], atol=2e-14
            )


def test_x2_normalized_mixer_matches_symmetric_geometry_closed_form(monkeypatch):
    allocation = load("px4_allocation", monkeypatch).Allocation
    model = mujoco.MjModel.from_xml_path(str(DEMOS / "assets" / "skydio_x2" / "x2.xml"))
    allocator = allocation(model, model.body("x2").id)
    # Fixed upstream X2: four equal motors, x=+/-0.14, y=+/-0.18 m.
    # PX4 equalizes roll/pitch scale; pitch's shorter arm sets that scale.
    roll, pitch = 0.14 / (np.sqrt(2) * 0.18), 1 / np.sqrt(2)
    expected = np.array(
        (
            (-roll, -pitch, 1, -1),
            (roll, -pitch, -1, -1),
            (roll, pitch, 1, -1),
            (-roll, pitch, -1, -1),
        )
    )
    np.testing.assert_allclose(allocator.mix, expected, atol=2e-14)
    np.testing.assert_allclose(
        allocator.scale,
        (
            np.sqrt(2) / (4 * 13 * 0.14),
            np.sqrt(2) / (4 * 13 * 0.14),
            1 / (4 * 13 * 0.0201),
            1 / (4 * 13),
        ),
        atol=2e-14,
    )


def test_px4_allocation_matches_compiled_upstream_method_bodies(monkeypatch):
    allocation = load("px4_allocation", monkeypatch).Allocation
    model = mujoco.MjModel.from_xml_path(str(DEMOS / "assets" / "skydio_x2" / "x2.xml"))
    allocator = allocation(model, model.body("x2").id)
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "mujoco" / "px4_allocation.json").read_text()
    )
    for (thrust, torque), expected in zip(
        fixture["requests_T_tau"], fixture["expected_rotor_forces_N"], strict=True
    ):
        forces, residual = allocator.allocate(thrust, torque)
        np.testing.assert_allclose(forces, expected, atol=2e-6, rtol=2e-6)
        assert np.all(forces >= allocator.lower)
        assert np.all(forces <= allocator.upper)
        np.testing.assert_allclose(
            allocator.effectiveness @ forces,
            np.r_[torque, thrust] + residual,
            atol=1e-12,
        )


def test_vendored_model_files_match_pinned_upstream_bytes():
    assets = DEMOS / "assets"
    manifest = json.loads((assets / "upstream.json").read_text())
    assert manifest["commit"] == "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
    for item in manifest["files"]:
        assert (
            hashlib.sha256((assets / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


def test_demos_are_self_contained_and_do_not_shadow_mujoco():
    assert not (DEMOS / "__init__.py").exists()
    local_modules = {path.stem for path in DEMOS.glob("*.py")}
    allowed = local_modules | {"mujoco", "numpy", "pivopt"} | sys.stdlib_module_names
    for path in DEMOS.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            else:
                continue
            assert all(name.split(".")[0] in allowed for name in names), (path, names)


@pytest.mark.parametrize(
    "name,error_key,max_error",
    (
        ("planar_base", "position_m", 0.001),
        ("arm", "tip_position_m", 0.001),
        ("unicycle", "position_m", 0.05),
        ("quadrotor", "position_m", 0.03),
    ),
)
def test_normal_optimized_trajectory_tracks_in_physics(name, error_key, max_error):
    process = subprocess.run(
        [sys.executable, str(DEMOS / (name + ".py")), "--headless"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    report = json.loads(process.stdout[process.stdout.index("{") :])
    assert report["tracking"][error_key]["max"] < max_error, report
    assert report["simulated_seconds"] > report["trajectory_seconds"]
    assert report["actuator_saturation_fraction"] == 0, report
    if name == "unicycle":
        assert report["terminal_status"] == {
            "parking_complete": True,
            "parking_timed_out": False,
        }
    for key, value in report["reference_checks"].items():
        assert value < (0.001 if "violation_m" in key else 0.01), (key, value)
    assert all(np.isfinite(value) for value in report["after_settling"].values())


def test_overlay_adds_geometries_without_removing_robot(monkeypatch):
    common = load("common", monkeypatch)
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = mujoco.MjvScene(model, maxgeom=100)
    scene.ngeom = 1  # An existing scene geom must survive offscreen append too.
    points = np.array(((0, 0, 1.0), (1.0, 0, 1.0), (1.0, 1.0, 1.0)))
    common.overlay(
        scene, points, points + 0.01, points[1], [(np.zeros(3), np.ones(3))], points[-1]
    )
    assert scene.ngeom == 19
