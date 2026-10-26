"""Optimize and track an xy+yaw trajectory with a force-driven planar base."""

import mujoco
import numpy as np
from common import ASSETS, arguments, run, wrap_angle
from trajectory import box_corridor, corridor_violation, solve

from pivopt import problem as p


class PlanarBase:
    name = "planar_base"
    control_period = 0.01
    settle_time = 2.0

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS / "planar_base.xml"))
        self.data = mujoco.MjData(self.model)
        vertices = np.array(((0, 0), (1.2, 0), (1.8, 0.6), (1.8, 1.8)))
        corridor, boxes = box_corridor(vertices, 0.25)
        self.boxes = [(np.r_[lo, 0.03], np.r_[hi, 0.6]) for lo, hi in boxes]
        pieces = len(vertices) - 1
        trajectory = p.MincoSpec(
            p.Space(position=2, yaw=True),
            derivative_order=3,
            pieces=pieces,
            head_state=p.Input("head"),
            tail_state=p.Input("tail"),
            time=p.FixedTotalTime(total_duration=8.0, minimum_piece_duration=0.2),
            waypoints=p.DirectWaypoints(),
        )
        spec = p.ProblemSpec(
            trajectory=trajectory,
            objective=(
                p.Energy("jerk", weights={"position": 1.0, "yaw": 0.3}),
                p.Integral(
                    "corridor",
                    p.Inside(
                        trajectory.position,
                        p.Input("corridor"),
                        profile=p.Hinge(width=0.03),
                    ),
                    weight=3000,
                ),
                p.Integral(
                    "speed",
                    p.NormBelow(trajectory.velocity, 1.2, profile=p.Hinge(width=0.05)),
                    weight=100,
                ),
                p.Integral(
                    "yaw_rate",
                    p.NormBelow(trajectory.yaw.d(1), 1.5, profile=p.Hinge(width=0.05)),
                    weight=100,
                ),
            ),
            sampling=p.Trapezoidal(intervals_per_piece=24),
            forward=p.LBFGS(
                initial_guess=p.Input("initial"),
                settings=p.OptimizerSettings(maximum_iterations=400, past=0),
            ),
            backward=p.ForwardOnly(),
        )
        plan = p.build_plan(
            spec,
            p.PlanShape(
                batch_size=1,
                input_shapes=(p.InputShape(p.InputRef("corridor"), corridor.shape),),
            ),
        )
        head, tail = np.zeros((1, 3, 3)), np.zeros((1, 3, 3))
        tail[0, :, 0] = (*vertices[-1], np.pi / 2)
        waypoints = np.c_[vertices[1:-1], np.linspace(0, np.pi / 2, pieces + 1)[1:-1]][
            None
        ]
        self.trajectory = solve(
            plan,
            plan.inputs(
                head=head,
                tail=tail,
                corridor=corridor,
                initial=plan.initial_guess(
                    waypoints=waypoints, durations=np.full((1, pieces), 8 / pieces)
                ),
            ),
        )
        body = self.model.body("base")
        self.inertia = np.array((body.mass[0], body.mass[0], body.inertia[2]))
        self.saturated = np.zeros(3, dtype=bool)
        mujoco.mj_forward(self.model, self.data)
        self.goal = self.reference_position(self.trajectory.duration)
        times = np.linspace(0, self.trajectory.duration, 2001)
        self.reference_checks = {
            "corridor_violation_m": corridor_violation(
                self.trajectory.sample(times)[..., :2],
                times,
                self.trajectory.durations,
                corridor,
            ),
            "speed_violation_m_s": float(
                np.maximum(
                    np.linalg.norm(self.trajectory.sample(times, 1)[..., :2], axis=-1)
                    - 1.2,
                    0,
                ).max()
            ),
            "yaw_rate_violation_rad_s": float(
                np.maximum(
                    np.abs(self.trajectory.sample(times, 1)[:, 2]) - 1.5, 0
                ).max()
            ),
        }

    def control(self, time):
        position, velocity, acceleration = self.trajectory.reference(time)
        error = position - self.data.qpos
        error[2] = wrap_angle(error[2])
        requested = self.inertia * (
            acceleration + 10 * error + 6 * (velocity - self.data.qvel)
        )
        self.data.ctrl[:] = np.clip(requested, *self.model.actuator_ctrlrange.T)
        self.saturated = np.abs(self.data.ctrl - requested) > 1e-9

    def reference_position(self, time):
        return np.r_[self.trajectory.sample(time)[:2], 0.16]

    def observe(self, time):
        reference = self.trajectory.sample(time)
        position = self.data.body("base").xpos.copy()
        return (
            position,
            self.reference_position(time),
            {
                "position_m": position[:2] - reference[:2],
                "yaw_rad": wrap_angle(self.data.qpos[2] - reference[2]),
            },
        )


if __name__ == "__main__":
    args = arguments(__doc__)
    run(PlanarBase(), headless=args.headless)
