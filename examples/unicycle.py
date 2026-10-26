"""Plan four forward-only unicycle maneuvers and report physical metrics.

The trajectory is xy only; the Unicycle readout takes heading as the velocity
direction, so there is no lateral slip. Rest endpoints fix their heading in
the boundary acceleration built by p.unicycle_state. Wheel limits use the
left/right rim linear speed, so no wheel radius is required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from constraints import Calibration, add_constraint_argument, term

from pivopt import ForwardStatus
from pivopt import numpy as cpu
from pivopt import problem as p

TRACK_WIDTH = 0.5
WHEEL_SPEED_LIMIT = 1.2
REST_ACCELERATION = 0.3
EVALUATION_SPEED = 0.05


@dataclass(frozen=True)
class Maneuver:
    name: str
    head_pose: tuple[float, float, float]
    tail_pose: tuple[float, float, float]
    waypoints: tuple[tuple[float, float], ...]
    total_duration: float
    corridor: bool = False


STRAIGHT = Maneuver(
    "straight_start_stop",
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    ((2 / 3, 0.0), (4 / 3, 0.0)),
    4.0,
)
LANE_CHANGE = Maneuver(
    "lane_change",
    (0.0, 0.0, 0.0),
    (3.0, 0.6, 0.0),
    ((1.0, 0.05), (2.0, 0.55)),
    5.0,
)
CORNER = Maneuver(
    "corner_90",
    (0.0, 0.0, 0.0),
    (1.5, 1.5, np.pi / 2),
    ((0.9, 0.0), (1.35, 0.35), (1.5, 0.9)),
    6.0,
    corridor=True,
)
U_TURN = Maneuver(
    "u_turn",
    (0.0, 0.0, 0.0),
    (0.0, 1.2, np.pi),
    ((1.0, 0.0), (1.6, 0.6), (1.0, 1.2)),
    7.0,
)
MANEUVERS = (STRAIGHT, LANE_CHANGE, CORNER, U_TURN)


def _polyline_corridor(maneuver: Maneuver) -> np.ndarray:
    vertices = np.asarray(
        (maneuver.head_pose[:2], *maneuver.waypoints, maneuver.tail_pose[:2]),
        dtype=np.float64,
    )
    rows = []
    for start, end in zip(vertices[:-1], vertices[1:], strict=True):
        tangent = end - start
        tangent /= np.linalg.norm(tangent)
        normal = np.asarray((-tangent[1], tangent[0]))
        rows.append(
            (
                (*tangent, -tangent @ end - 0.1),
                (*-tangent, tangent @ start - 0.1),
                (*normal, -normal @ start - 0.15),
                (*-normal, normal @ start - 0.15),
            )
        )
    return np.asarray(rows, dtype=np.float64)[None]


def _problem(maneuver: Maneuver, *, constraints="hinge"):
    pieces = len(maneuver.waypoints) + 1
    trajectory = p.MincoSpec(
        p.Space(position=2),
        derivative_order=3,
        pieces=pieces,
        head_state=p.Input("head"),
        tail_state=p.Input("tail"),
        time=p.FrozenTime(durations=p.Input("times")),
        waypoints=p.DirectWaypoints(),
    )
    car = p.Unicycle(track_width=TRACK_WIDTH)
    objective = [
        p.Energy("energy"),
        term(
            p.Integral,
            "wheel_speeds",
            lambda profile: p.Within(
                car.wheel_speeds, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT, profile=profile
            ),
            Calibration(
                hinge_weight=2000.0,
                width=0.05,
                barrier_weight=22.0,
                delta=0.011,
                activation=0.12,
            ),
            constraints,
        ),
        term(
            p.Integral,
            "forward_acceleration",
            lambda profile: p.NormBelow(car.forward_acceleration, 2.0, profile=profile),
            Calibration(
                hinge_weight=50.0,
                width=0.1,
                barrier_weight=1.1,
                delta=0.022,
                activation=0.2,
            ),
            constraints,
        ),
        term(
            p.Integral,
            "centripetal_acceleration",
            lambda profile: p.NormBelow(
                car.centripetal_acceleration, 1.5, profile=profile
            ),
            Calibration(
                hinge_weight=50.0,
                width=0.1,
                barrier_weight=1.1,
                delta=0.022,
                activation=0.15,
            ),
            constraints,
        ),
    ]
    corridor = None
    if maneuver.corridor:
        corridor = _polyline_corridor(maneuver)
        objective.append(
            term(
                p.Integral,
                "polyline_corridor",
                lambda profile: p.Inside(
                    trajectory.position, p.Input("corridor"), profile=profile
                ),
                Calibration(
                    hinge_weight=100000.0,
                    width=0.05,
                    barrier_weight=1100.0,
                    delta=0.011,
                    activation=0.033,
                ),
                constraints,
            )
        )
    spec = p.ProblemSpec(
        trajectory=trajectory,
        objective=tuple(objective),
        sampling=p.Trapezoidal(intervals_per_piece=24),
        forward=p.LBFGS(
            initial_guess=p.Input("initial"),
            memory_size=16,
            settings=p.OptimizerSettings(
                maximum_iterations=1000,
                gradient_tolerance=1e-7,
                past=0,
            ),
        ),
        backward=p.ForwardOnly(),
    )
    shapes = (
        ()
        if corridor is None
        else (p.InputShape(p.InputRef("corridor"), corridor.shape),)
    )
    plan = p.build_plan(spec, p.PlanShape(batch_size=1, input_shapes=shapes))
    head = p.unicycle_state(
        np.asarray([maneuver.head_pose[:2]]),
        yaw=maneuver.head_pose[2],
        acceleration=REST_ACCELERATION,
    )
    tail = p.unicycle_state(
        np.asarray([maneuver.tail_pose[:2]]),
        yaw=maneuver.tail_pose[2],
        acceleration=-REST_ACCELERATION,
    )
    durations = np.full((1, pieces), maneuver.total_duration / pieces)
    values = {
        "head": head,
        "tail": tail,
        "times": durations,
        "initial": plan.initial_guess(waypoints=np.asarray([maneuver.waypoints])),
    }
    if corridor is not None:
        values["corridor"] = corridor
    return plan, plan.inputs(**values)


def _derivatives(result, nodes_per_piece=201):
    """Exact position, velocity and acceleration samples, piece by piece."""
    coefficients = result.coefficients[0]
    durations = result.durations[0]
    unit_time = np.linspace(0.0, 1.0, nodes_per_piece)
    samples = [[], [], []]
    for piece, duration in enumerate(durations):
        time = unit_time * duration
        for order in range(3):
            samples[order].append(
                sum(
                    np.prod(range(power - order + 1, power + 1))
                    * coefficients[piece, power][None, :]
                    * time[:, None] ** (power - order)
                    for power in range(order, coefficients.shape[1])
                )
            )
    return tuple(np.concatenate(items) for items in samples)


def _boundary_heading(result, *, start: bool, offset=1e-6) -> float:
    """Velocity direction just after departure or just before arrival."""
    coefficients = result.coefficients[0, 0 if start else -1]
    time = offset if start else result.durations[0, -1] - offset
    velocity = sum(
        power * coefficients[power] * time ** (power - 1)
        for power in range(1, coefficients.shape[0])
    )
    return float(np.arctan2(velocity[1], velocity[0]))


def _heading_error(actual: float, expected: float) -> float:
    return float(np.angle(np.exp(1j * (actual - expected))))


def _report(maneuver: Maneuver, result) -> dict:
    position, velocity, acceleration = _derivatives(result)
    speed = np.linalg.norm(velocity, axis=-1)
    moving = speed >= EVALUATION_SPEED
    cross = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
    yaw_rate = cross[moving] / speed[moving] ** 2
    tangential = (velocity * acceleration).sum(-1)[moving] / speed[moving]
    wheels = speed[moving, None] + np.outer(yaw_rate, (-0.5, 0.5)) * TRACK_WIDTH
    heading = np.unwrap(np.arctan2(velocity[moving, 1], velocity[moving, 0]))
    return {
        "name": maneuver.name,
        "objective": float(result.cost[0]),
        "iterations": int(result.diagnostics.iterations[0]),
        "max_speed": float(speed.max()),
        "max_abs_yaw_rate": float(np.abs(yaw_rate).max()),
        "wheel_speed_violation": float(
            np.max(np.maximum(np.abs(wheels) - WHEEL_SPEED_LIMIT, 0.0))
        ),
        "max_abs_forward_acceleration": float(np.abs(tangential).max()),
        "max_abs_centripetal_acceleration": float(
            np.abs(speed[moving] * yaw_rate).max()
        ),
        "start_heading_error": _heading_error(
            _boundary_heading(result, start=True), maneuver.head_pose[2]
        ),
        "end_heading_error": _heading_error(
            _boundary_heading(result, start=False), maneuver.tail_pose[2]
        ),
        "heading_change": float(heading[-1] - heading[0]),
        "final_position_error": float(
            np.linalg.norm(position[-1] - maneuver.tail_pose[:2])
        ),
    }


def run_scenarios(*, constraints="hinge"):
    reports = []
    for maneuver in MANEUVERS:
        plan, inputs = _problem(maneuver, constraints=constraints)
        result = cpu.build_solver(plan, dtype=np.float64).solve(inputs)
        if np.any(result.status != ForwardStatus.RESULT_AVAILABLE):
            raise RuntimeError(f"{maneuver.name} failed: {result.diagnostics}")
        reports.append(_report(maneuver, result))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_constraint_argument(parser)
    args = parser.parse_args()
    print(json.dumps(run_scenarios(constraints=args.constraints), indent=2))


if __name__ == "__main__":
    main()
