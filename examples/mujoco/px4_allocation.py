"""PX4 v1.17.0 sequential desaturation, MC_AIRMODE=1, for the X2's four rotors.

Adapted from src/lib/control_allocation/control_allocation/:
  ControlAllocationPseudoInverse.cpp: updateControlAllocationMatrixScale,
      normalizeControlAllocationMatrix
  ControlAllocationSequentialDesaturation.cpp: mixAirmodeRP, mixYaw,
      desaturateActuators, computeDesaturationGain
https://github.com/PX4/PX4-Autopilot/tree/v1.17.0/src/lib/control_allocation/control_allocation
Copyright (c) 2019 PX4 Development Team. All rights reserved.
Redistributed under BSD-3-Clause; see assets/PX4-LICENSE.

Adaptations: NumPy float64, four active wrench axes, SI inputs/outputs. Convert
the X2's body FLU axes to PX4 FRD and normalize actuator thrust to [0, 1].
The inverse-matrix column scales convert SI wrench requests to PX4 normalized
requests, retaining SI authority and the reference algorithm's 0.2 threshold.
"""

import mujoco
import numpy as np


class Allocation:
    def __init__(self, model, body_id):
        self.lower, self.upper = model.actuator_ctrlrange.T.copy()
        self.span = self.upper - self.lower
        # X2 motors have unit force gain and parallel +body-z thrust directions.
        self.effectiveness = np.empty((4, 4))  # [tau_x, tau_y, tau_z, thrust_z] / N
        for rotor in range(4):
            site = model.actuator_trnid[rotor, 0]
            site_rotation = np.empty(9)
            mujoco.mju_quat2Mat(site_rotation, model.site_quat[site])
            site_rotation = site_rotation.reshape(3, 3)
            force = site_rotation @ model.actuator_gear[rotor, :3]
            reaction = site_rotation @ model.actuator_gear[rotor, 3:]
            lever = model.site_pos[site] - model.body_ipos[body_id]
            self.effectiveness[:3, rotor] = np.cross(lever, force) + reaction
            self.effectiveness[3, rotor] = force[2]
        self.frd_sign = np.array((1, -1, -1, -1))
        inverse = np.linalg.pinv(
            self.frd_sign[:, None] * self.effectiveness * self.span
        )
        # Reference: PX4 ControlAllocationPseudoInverse::updateControlAllocationMatrixScale.
        rp_scale = max(
            np.linalg.norm(inverse[:, 0]) / np.sqrt(2),
            np.linalg.norm(inverse[:, 1]) / np.sqrt(2),
        )
        self.scale = np.array(
            (rp_scale, rp_scale, np.max(inverse[:, 2]), np.mean(np.abs(inverse[:, 3])))
        )
        self.mix = inverse / self.scale
        self.mix[np.abs(self.mix) < 1e-3] = 0

    @staticmethod
    def desaturation_gain(direction, motors, upper):
        # Reference: PX4 computeDesaturationGain, including weak-axis filtering.
        gains = [0.0]
        for value, actuator, maximum in zip(
            direction, motors, np.broadcast_to(upper, motors.shape), strict=True
        ):
            if abs(value) < 0.2:
                continue
            if actuator < 0:
                gains.append(-actuator / value)
            if actuator > maximum:
                gains.append((maximum - actuator) / value)
        return min(gains) + max(gains)

    def desaturate(self, motors, direction, upper=1.0, reduce_thrust_only=False):
        # Reference: PX4 desaturateActuators. In FRD, positive thrust-axis gain
        # reduces upward thrust; preserve that sign for its increase_only flag.
        gain = self.desaturation_gain(direction, motors, upper)
        if reduce_thrust_only and gain < 0:
            return
        motors += gain * direction
        motors += 0.5 * self.desaturation_gain(direction, motors, upper) * direction

    def allocate(self, thrust, torque):
        requested = np.r_[torque, thrust]
        # Remove lower-bound trim, then map SI wrench to normalized FRD requests.
        control = (
            self.scale * self.frd_sign * (requested - self.effectiveness @ self.lower)
        )
        # Reference: PX4 mixAirmodeRP, then mixYaw (MINIMUM_YAW_MARGIN=0.15).
        motors = self.mix[:, (0, 1, 3)] @ control[[0, 1, 3]]
        self.desaturate(motors, self.mix[:, 3])
        motors += self.mix[:, 2] * control[2]
        self.desaturate(motors, self.mix[:, 2], upper=1.15)
        self.desaturate(motors, self.mix[:, 3], reduce_thrust_only=True)
        # PX4 ControlAllocation::clipActuatorSetpoint: the temporary yaw margin
        # is internal only; final physical rotor forces obey the MJCF bounds.
        forces = self.lower + np.clip(motors, 0, 1) * self.span
        return forces, self.effectiveness @ forces - requested
