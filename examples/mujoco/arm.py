"""Optimize a UR5e joint trajectory to a tip target, then track with torque PD."""

import mujoco
import numpy as np
from common import ASSETS, arguments, run
from trajectory import box_corridor, corridor_violation, solve

from pivopt import problem as p


def quaternion(rotation):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(rotation).ravel())
    return tuple(result)


def serial_chain(model):
    """Build the planning chain from the same MJCF's zero-configuration frames."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    previous_rotation, previous_position = np.eye(3), np.zeros(3)
    joints = []
    for joint in range(model.njnt):
        body = model.jnt_bodyid[joint]
        rotation = data.xmat[body].reshape(3, 3)
        position = data.xanchor[joint]
        joints.append(
            p.Joint(
                origin=p.Pose(
                    position=tuple(
                        previous_rotation.T @ (position - previous_position)
                    ),
                    orientation=quaternion(previous_rotation.T @ rotation),
                ),
                axis=tuple(model.jnt_axis[joint]),
            )
        )
        previous_rotation, previous_position = rotation.copy(), position.copy()
    tip = data.site("attachment_site")
    tip_position = tuple(previous_rotation.T @ (tip.xpos - previous_position))
    return p.SerialChain(
        joints=tuple(joints),
        collision_spheres=(p.Sphere(link=6, center=tip_position, radius=0.02),),
        tip=p.Pose(
            position=tip_position,
            orientation=quaternion(previous_rotation.T @ tip.xmat.reshape(3, 3)),
        ),
    )


class Arm:
    name = "arm"
    control_period = 0.002
    settle_time = 2.0

    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(ASSETS / "ur5e" / "ur5e.xml"))
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3, 3, 0.1],
            rgba=[0.85, 0.86, 0.89, 1],
        )
        self.model = spec.compile()
        self.model.opt.timestep = self.control_period
        # Menagerie ur5e.xml: position-controlled general actuators, +/-150/28 Nm.
        # https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/universal_robots_ur5e/ur5e.xml
        # Adaptation: unit-gain torque motors; preserve joint/force limits and dynamics.
        self.model.actuator_ctrlrange[:] = self.model.actuator_forcerange
        self.model.actuator_gainprm[:] = 0
        self.model.actuator_gainprm[:, 0] = 1
        self.model.actuator_biasprm[:] = 0
        self.model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        self.model.key_ctrl[:] = 0
        self.data, self.reference_data = (
            mujoco.MjData(self.model),
            mujoco.MjData(self.model),
        )
        self.chain = serial_chain(self.model)
        initial_q = self.model.key("home").qpos.copy()
        goal_q = initial_q + np.array((0.7, 0.25, -0.3, 0.25, 0.3, 0.2))
        self.reference_data.qpos[:] = goal_q
        mujoco.mj_forward(self.model, self.reference_data)
        self.goal = self.reference_data.site("attachment_site").xpos.copy()
        pieces = 4
        trajectory = p.MincoSpec(
            p.Space(joints=6),
            derivative_order=3,
            pieces=pieces,
            head_state=p.Input("head"),
            tail_state=p.FreeTail(derivatives=p.Input("tail_derivatives")),
            time=p.FixedTotalTime(total_duration=6.0, minimum_piece_duration=0.2),
            waypoints=p.DirectWaypoints(),
        )
        self.reference_data.qpos[:] = initial_q
        mujoco.mj_forward(self.model, self.reference_data)
        start_tip = self.reference_data.site("attachment_site").xpos.copy()
        corridor, self.boxes = box_corridor((start_tip, self.goal), 0.18)
        corridor = np.repeat(corridor, pieces, axis=1)
        problem = p.ProblemSpec(
            trajectory=trajectory,
            objective=(
                p.Energy("jerk"),
                p.AtEnd(
                    "tip_target",
                    p.Quadratic(self.chain.tip_position, target=tuple(self.goal)),
                    weight=3000,
                ),
                p.Integral(
                    "joint_limits",
                    p.Within(
                        trajectory.joints,
                        self.model.jnt_range[:, 0],
                        self.model.jnt_range[:, 1],
                        profile=p.Hinge(width=0.03),
                    ),
                    weight=2000,
                ),
                p.Integral(
                    "joint_speed",
                    p.Within(
                        trajectory.joints.d(1), -1.5, 1.5, profile=p.Hinge(width=0.05)
                    ),
                    weight=100,
                ),
                p.Integral(
                    "workspace",
                    p.Inside(
                        self.chain.spheres,
                        p.Input("workspace"),
                        profile=p.Hinge(width=0.03),
                    ),
                    weight=2000,
                ),
            ),
            sampling=p.Trapezoidal(intervals_per_piece=24),
            forward=p.LBFGS(
                initial_guess=p.Input("initial"),
                settings=p.OptimizerSettings(maximum_iterations=500, past=0),
            ),
            backward=p.ForwardOnly(),
        )
        plan = p.build_plan(
            problem,
            p.PlanShape(
                batch_size=1,
                input_shapes=(p.InputShape(p.InputRef("workspace"), corridor.shape),),
            ),
        )
        head = np.zeros((1, 6, 3))
        head[0, :, 0] = initial_q
        waypoints = np.array(
            [initial_q + (goal_q - initial_q) * i / pieces for i in range(1, pieces)]
        )[None]
        self.trajectory = solve(
            plan,
            plan.inputs(
                head=head,
                tail_derivatives=np.zeros((1, 6, 2)),
                workspace=corridor,
                initial=plan.initial_guess(
                    waypoints=waypoints,
                    terminal=goal_q[None],
                    durations=np.full((1, pieces), 1.5),
                ),
            ),
        )
        self.data.qpos[:] = initial_q
        mujoco.mj_forward(self.model, self.data)
        self.mass_matrix = np.empty((6, 6))
        self.saturated = np.zeros(6, dtype=bool)
        times = np.linspace(0, self.trajectory.duration, 1001)
        joints = self.trajectory.sample(times)
        tip = np.stack([self.reference_position(t) for t in times])
        sphere_corridor = corridor.copy()
        sphere_corridor[..., -1] += self.chain.collision_spheres[0].radius
        self.reference_checks = {
            "joint_limit_violation_rad": float(
                np.maximum(
                    np.maximum(
                        self.model.jnt_range[:, 0] - joints,
                        joints - self.model.jnt_range[:, 1],
                    ),
                    0,
                ).max()
            ),
            "joint_speed_violation_rad_s": float(
                np.maximum(np.abs(self.trajectory.sample(times, 1)) - 1.5, 0).max()
            ),
            "tip_sphere_workspace_violation_m": corridor_violation(
                tip, times, self.trajectory.durations, sphere_corridor
            ),
            "optimized_tip_target_error_m": float(np.linalg.norm(tip[-1] - self.goal)),
        }

    def control(self, time):
        position, velocity, acceleration = self.trajectory.reference(time)
        mujoco.mj_fullM(self.model, self.mass_matrix, self.data.qM)
        requested = (
            self.mass_matrix @ acceleration
            + self.data.qfrc_bias
            - self.data.qfrc_passive
            + 140 * (position - self.data.qpos)
            + 35 * (velocity - self.data.qvel)
        )
        self.data.ctrl[:] = np.clip(requested, *self.model.actuator_ctrlrange.T)
        self.saturated = np.abs(self.data.ctrl - requested) > 1e-9

    def reference_position(self, time):
        self.reference_data.qpos[:] = self.trajectory.sample(time)
        mujoco.mj_forward(self.model, self.reference_data)
        return self.reference_data.site("attachment_site").xpos.copy()

    def observe(self, time):
        position, target = (
            self.data.site("attachment_site").xpos.copy(),
            self.reference_position(time),
        )
        return (
            position,
            target,
            {
                "tip_position_m": position - target,
                "joint_rad": self.data.qpos - self.trajectory.sample(time),
            },
        )


if __name__ == "__main__":
    args = arguments(__doc__)
    run(Arm(), headless=args.headless)
