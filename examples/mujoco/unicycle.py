"""Optimize an xy corner trajectory and track with a contact-driven differential drive."""

import mujoco
import numpy as np
from common import ASSETS, arguments, run, wrap_angle
from trajectory import box_corridor, corridor_violation, solve

from pivopt import problem as p


class Unicycle:
    name = "unicycle"
    control_period = 0.01
    settle_time = 3.0

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS / "unicycle.xml"))
        self.data = mujoco.MjData(self.model)
        self.radius = self.model.geom("left_tire").size[0]
        self.track_width = abs(
            self.model.body("left_wheel").pos[1] - self.model.body("right_wheel").pos[1]
        )
        self.wheel_limit = self.model.numeric("wheel_speed_limit").data[0]
        self.wheel_dofs = [
            self.model.joint(name).dofadr[0] for name in ("left_wheel", "right_wheel")
        ]
        self.integral = np.zeros(2)
        self.saturated = np.zeros(2, dtype=bool)
        vertices = np.array(((0, 0), (0.9, 0), (1.35, 0.35), (1.5, 0.9), (1.5, 1.5)))
        pieces = len(vertices) - 1
        corridor, boxes = box_corridor(vertices, 0.18)
        self.boxes = [(np.r_[lo, 0.03], np.r_[hi, 0.6]) for lo, hi in boxes]
        trajectory = p.MincoSpec(
            p.Space(position=2),
            derivative_order=3,
            pieces=pieces,
            head_state=p.Input("head"),
            tail_state=p.Input("tail"),
            time=p.FixedTotalTime(total_duration=8.0, minimum_piece_duration=0.3),
            waypoints=p.DirectWaypoints(),
        )
        car = p.Unicycle(track_width=self.track_width)
        problem = p.ProblemSpec(
            trajectory=trajectory,
            objective=(
                p.Energy("jerk"),
                p.Integral(
                    "corridor",
                    p.Inside(
                        trajectory.position,
                        p.Input("corridor"),
                        profile=p.Hinge(width=0.03),
                    ),
                    weight=10000,
                ),
                p.Integral(
                    "wheel_speed",
                    p.Within(
                        car.wheel_speeds,
                        -self.radius * self.wheel_limit,
                        self.radius * self.wheel_limit,
                        profile=p.Hinge(width=0.03),
                    ),
                    weight=2000,
                ),
                p.Integral(
                    "forward_acceleration",
                    p.NormBelow(
                        car.forward_acceleration, 1.2, profile=p.Hinge(width=0.05)
                    ),
                    weight=100,
                ),
                p.Integral(
                    "centripetal_acceleration",
                    p.NormBelow(
                        car.centripetal_acceleration, 1.0, profile=p.Hinge(width=0.05)
                    ),
                    weight=100,
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
        # At rest, nonzero boundary acceleration fixes the one-sided heading.
        head = p.unicycle_state(vertices[0][None], yaw=0.0, acceleration=0.25)
        tail = p.unicycle_state(vertices[-1][None], yaw=np.pi / 2, acceleration=-0.25)
        self.trajectory = solve(
            plan,
            plan.inputs(
                head=head,
                tail=tail,
                corridor=corridor,
                initial=plan.initial_guess(
                    waypoints=vertices[1:-1][None], durations=np.full((1, pieces), 2.0)
                ),
            ),
        )
        mujoco.mj_forward(self.model, self.data)
        self.goal = self.reference_position(self.trajectory.duration)
        times = np.linspace(0, self.trajectory.duration, 2001)
        velocity, acceleration = (
            self.trajectory.sample(times, 1),
            self.trajectory.sample(times, 2),
        )
        speed = np.linalg.norm(velocity, axis=-1)
        moving = speed > 1e-4
        yaw_rate = np.zeros_like(speed)
        yaw_rate[moving] = (
            velocity[moving, 0] * acceleration[moving, 1]
            - velocity[moving, 1] * acceleration[moving, 0]
        ) / speed[moving] ** 2
        rim_speeds = (
            speed[:, None]
            + yaw_rate[:, None] * np.array((-0.5, 0.5)) * self.track_width
        )
        forward_acceleration = (velocity[moving] * acceleration[moving]).sum(
            axis=-1
        ) / speed[moving]
        self.reference_checks = {
            "corridor_violation_m": corridor_violation(
                self.trajectory.sample(times),
                times,
                self.trajectory.durations,
                corridor,
            ),
            "wheel_speed_violation_rad_s": float(
                np.maximum(np.abs(rim_speeds) / self.radius - self.wheel_limit, 0).max()
            ),
            "forward_acceleration_violation_m_s2": float(
                np.maximum(np.abs(forward_acceleration) - 1.2, 0).max()
            ),
            "centripetal_acceleration_violation_m_s2": float(
                np.maximum(np.abs(speed * yaw_rate) - 1.0, 0).max()
            ),
        }

    def motion_reference(self, time):
        position = self.trajectory.sample(time)
        # Use the neighboring one-sided velocity at rest, not atan2(0, 0).
        inside = np.clip(time, 1e-3, self.trajectory.duration - 1e-3)
        velocity, acceleration = (
            self.trajectory.sample(inside, 1),
            self.trajectory.sample(inside, 2),
        )
        speed_squared = velocity @ velocity
        heading = np.arctan2(velocity[1], velocity[0])
        yaw_rate = (
            velocity[0] * acceleration[1] - velocity[1] * acceleration[0]
        ) / speed_squared
        speed = np.linalg.norm(self.trajectory.sample(time, 1))
        if time <= 0:
            heading = 0.0
        if time >= self.trajectory.duration:
            heading, speed, yaw_rate = np.pi / 2, 0.0, 0.0
        return position, heading, speed, yaw_rate

    def control(self, time):
        position, heading, speed, yaw_rate = self.motion_reference(time)
        rotation = self.data.body("base").xmat.reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        error_world = position - self.data.body("base").xpos[:2]
        error_body = rotation[:2, :2].T @ error_world
        angle_error = wrap_angle(heading - yaw)
        # Body-frame pose feedback plus time-indexed v/omega feedforward.
        linear = speed * np.cos(angle_error) + 1.6 * error_body[0]
        angular = yaw_rate + 3.0 * speed * error_body[1] + 3.0 * np.sin(angle_error)
        if time >= self.trajectory.duration:
            distance = np.linalg.norm(error_world)
            if distance > 0.015:
                bearing = wrap_angle(np.arctan2(error_world[1], error_world[0]) - yaw)
                direction = 1.0
                if abs(bearing) > np.pi / 2:
                    bearing = wrap_angle(bearing + np.pi)
                    direction = -1.0
                linear = direction * min(0.15, 0.8 * distance) * np.cos(bearing)
                angular = 2.5 * np.sin(bearing)
            else:
                linear = 0.0
                angular = 2.5 * angle_error if abs(angle_error) > 0.01 else 0.0
        wheel_target = (
            np.array(
                (
                    linear - self.track_width * angular / 2,
                    linear + self.track_width * angular / 2,
                )
            )
            / self.radius
        )
        wheel_target /= max(1.0, np.max(np.abs(wheel_target)) / self.wheel_limit)
        error = wheel_target - self.data.qvel[self.wheel_dofs]
        requested = 0.25 * error + self.integral
        self.data.ctrl[:] = np.clip(requested, *self.model.actuator_ctrlrange.T)
        self.saturated = np.abs(self.data.ctrl - requested) > 1e-9
        # Conditional integration: allow unwinding, block growth into a torque limit.
        integrate = (~self.saturated) | (error * (requested - self.data.ctrl) <= 0)
        self.integral[integrate] += 0.8 * self.control_period * error[integrate]

    def terminal_status(self):
        rotation = self.data.body("base").xmat.reshape(3, 3)
        position_error = np.linalg.norm(self.data.body("base").xpos[:2] - self.goal[:2])
        heading_error = wrap_angle(
            np.arctan2(rotation[1, 0], rotation[0, 0]) - np.pi / 2
        )
        complete = bool(position_error <= 0.015 and abs(heading_error) <= 0.01)
        timeout = (
            self.data.time
            >= self.trajectory.duration + self.settle_time - self.model.opt.timestep
        )
        return {
            "parking_complete": complete,
            "parking_timed_out": bool(timeout and not complete),
        }

    def reference_position(self, time):
        return np.r_[self.trajectory.sample(time), 0.15]

    def observe(self, time):
        position = self.data.body("base").xpos.copy()
        rotation = self.data.body("base").xmat.reshape(3, 3)
        target, heading, _, _ = self.motion_reference(time)
        return (
            position,
            self.reference_position(time),
            {
                "position_m": position[:2] - target,
                "yaw_rad": wrap_angle(
                    np.arctan2(rotation[1, 0], rotation[0, 0]) - heading
                ),
            },
        )


if __name__ == "__main__":
    args = arguments(__doc__)
    run(Unicycle(), headless=args.headless)
