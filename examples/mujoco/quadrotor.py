"""Optimize xyz+yaw and track it with the Menagerie Skydio X2 and PX4 allocation."""

import mujoco
import numpy as np
from common import ASSETS, arguments, run, wrap_angle
from px4_allocation import Allocation
from trajectory import box_corridor, corridor_violation, solve

from pivopt import problem as p


def desired_rotation(force, yaw):
    z_axis = force / np.linalg.norm(force)
    heading = np.array((np.cos(yaw), np.sin(yaw), 0))
    y_axis = np.cross(z_axis, heading)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((np.cross(y_axis, z_axis), y_axis, z_axis))


def vee(matrix):
    return np.array((matrix[2, 1], matrix[0, 2], matrix[1, 0]))


class Quadrotor:
    name = "quadrotor"
    control_period = 0.002
    settle_time = 3.0

    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(ASSETS / "skydio_x2" / "x2.xml"))
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[5, 5, 0.1],
            rgba=[0.85, 0.86, 0.89, 1],
        )
        self.model = spec.compile()
        # Adapt only integration resolution; retain X2 mass, COM, inertia,
        # rotor geometry, reaction torque, force bounds and fluid properties.
        self.model.opt.timestep = self.control_period
        self.data = mujoco.MjData(self.model)
        body = self.model.body("x2")
        self.body_id, self.mass = body.id, body.mass[0]
        principal_rotation = np.empty(9)
        mujoco.mju_quat2Mat(principal_rotation, body.iquat)
        principal_rotation = principal_rotation.reshape(3, 3)
        self.inertia = principal_rotation @ np.diag(body.inertia) @ principal_rotation.T
        self.gravity = -self.model.opt.gravity.copy()
        self.allocation = Allocation(self.model, body.id)
        self.saturated = np.zeros(4, dtype=bool)
        self.allocation_residual_max = np.zeros(4)
        vertices = np.array(
            ((0, 0, 1), (1, 0, 1.5), (1.5, 1, 1.8), (0.5, 1.6, 1.2), (0, 0.8, 1.2))
        )
        pieces = len(vertices) - 1
        corridor, self.boxes = box_corridor(vertices, 0.25)
        trajectory = p.MincoSpec(
            p.Space(position=3, yaw=True),
            derivative_order=3,
            pieces=pieces,
            head_state=p.Input("head"),
            tail_state=p.Input("tail"),
            time=p.FixedTotalTime(total_duration=12.0, minimum_piece_duration=0.3),
            waypoints=p.DirectWaypoints(),
        )
        quad = p.Quadrotor(mass=self.mass, gravity=self.gravity[2])
        self.thrust_limit = self.model.actuator_ctrlrange[:, 1].sum()
        problem = p.ProblemSpec(
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
                    p.NormBelow(trajectory.velocity, 2.0, profile=p.Hinge(width=0.05)),
                    weight=200,
                ),
                p.Integral(
                    "thrust",
                    p.Within(
                        quad.thrust, 0, self.thrust_limit, profile=p.Hinge(width=0.1)
                    ),
                    weight=200,
                ),
                p.Integral(
                    "tilt",
                    p.TiltBelow(quad.attitude, 0.6, profile=p.Hinge(width=0.03)),
                    weight=200,
                ),
                p.Integral(
                    "body_rate",
                    p.NormBelow(quad.body_rate, 2.0, profile=p.Hinge(width=0.03)),
                    weight=200,
                ),
            ),
            sampling=p.Trapezoidal(intervals_per_piece=32),
            forward=p.LBFGS(
                initial_guess=p.Input("initial"),
                settings=p.OptimizerSettings(maximum_iterations=600, past=0),
            ),
            backward=p.ForwardOnly(),
        )
        plan = p.build_plan(
            problem,
            p.PlanShape(
                batch_size=1,
                input_shapes=(p.InputShape(p.InputRef("corridor"), corridor.shape),),
            ),
        )
        head, tail = np.zeros((1, 4, 3)), np.zeros((1, 4, 3))
        head[0, :3, 0], tail[0, :3, 0] = vertices[0], vertices[-1]
        tail[0, 3, 0] = 1.2
        waypoints = np.c_[vertices[1:-1], (0.3, 0.7, 1.0)][None]
        self.trajectory = solve(
            plan,
            plan.inputs(
                head=head,
                tail=tail,
                corridor=corridor,
                initial=plan.initial_guess(
                    waypoints=waypoints, durations=np.full((1, pieces), 3.0)
                ),
            ),
        )
        # Reference position is center of mass, not the MJCF body's origin.
        self.data.qpos[:3] = vertices[0] - body.ipos
        mujoco.mj_forward(self.model, self.data)
        self.goal = vertices[-1].copy()
        times = np.linspace(0, self.trajectory.duration, 2001)
        acceleration = self.trajectory.sample(times, 2)[:, :3] + self.gravity
        thrust = self.mass * np.linalg.norm(acceleration, axis=-1)
        tilt = np.arctan2(
            np.linalg.norm(acceleration[:, :2], axis=-1), acceleration[:, 2]
        )
        rotations = np.stack(
            [
                desired_rotation(a, yaw)
                for a, yaw in zip(
                    acceleration, self.trajectory.sample(times)[:, 3], strict=True
                )
            ]
        )
        rotation_rates = np.transpose(rotations, (0, 2, 1)) @ np.gradient(
            rotations, times, axis=0
        )
        body_rates = np.stack([0.5 * vee(rate - rate.T) for rate in rotation_rates])
        self.reference_checks = {
            "corridor_violation_m": corridor_violation(
                self.trajectory.sample(times)[:, :3],
                times,
                self.trajectory.durations,
                corridor,
            ),
            "speed_violation_m_s": float(
                np.maximum(
                    np.linalg.norm(self.trajectory.sample(times, 1)[:, :3], axis=-1)
                    - 2.0,
                    0,
                ).max()
            ),
            "collective_thrust_violation_N": float(
                np.maximum(thrust - self.thrust_limit, 0).max()
            ),
            "tilt_violation_rad": float(np.maximum(tilt - 0.6, 0).max()),
            "body_rate_violation_rad_s": float(
                np.maximum(np.linalg.norm(body_rates, axis=-1) - 2.0, 0).max()
            ),
        }

    def control(self, time):
        position, velocity, acceleration = self.trajectory.reference(time)
        rotation = self.data.body("x2").xmat.reshape(3, 3)
        spatial_velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.body_id,
            spatial_velocity,
            0,
        )
        omega = rotation.T @ spatial_velocity[:3]
        force = self.mass * (
            acceleration[:3]
            + self.gravity
            + 6 * (position[:3] - self.data.xipos[self.body_id])
            + 4 * (velocity[:3] - spatial_velocity[3:])
        )
        target_rotation = desired_rotation(force, position[3])
        attitude_error = 0.5 * vee(
            target_rotation.T @ rotation - rotation.T @ target_rotation
        )
        yaw_rate_body = rotation.T @ np.array((0, 0, velocity[3]))
        torque = (
            -np.array((4, 4, 2)) * attitude_error
            - np.array((1, 1, 0.6)) * (omega - yaw_rate_body)
            + np.cross(omega, self.inertia @ omega)
        )
        thrust = max(0.0, force @ rotation[:, 2])
        forces, residual = self.allocation.allocate(thrust, torque)
        self.data.ctrl[:] = forces
        self.saturated = (forces <= self.allocation.lower + 1e-9) | (
            forces >= self.allocation.upper - 1e-9
        )
        self.allocation_residual_max = np.maximum(
            self.allocation_residual_max, np.abs(residual[[3, 0, 1, 2]])
        )

    def reference_position(self, time):
        return self.trajectory.sample(time)[:3]

    def observe(self, time):
        position = self.data.xipos[self.body_id].copy()
        reference = self.trajectory.sample(time)
        rotation = self.data.body("x2").xmat.reshape(3, 3)
        return (
            position,
            reference[:3],
            {
                "position_m": position - reference[:3],
                "yaw_rad": wrap_angle(
                    np.arctan2(rotation[1, 0], rotation[0, 0]) - reference[3]
                ),
            },
        )


if __name__ == "__main__":
    args = arguments(__doc__)
    run(Quadrotor(), headless=args.headless)
