# Model and algorithm provenance

The imported model files are unchanged snapshots from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), commit
`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`.
`upstream.json` records each upstream path and its downloaded SHA-256.

| Local directory | Upstream directory | License |
| --- | --- | --- |
| `ur5e/` | `universal_robots_ur5e/` | BSD-3-Clause, retained in `ur5e/LICENSE` |
| `skydio_x2/` | `skydio_x2/` | Apache-2.0, retained in `skydio_x2/LICENSE` |

Sources:

- [UR5e MJCF](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/universal_robots_ur5e/ur5e.xml)
- [Skydio X2 MJCF](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/skydio_x2/x2.xml)

Runtime adaptations are visible in `arm.py` and `quadrotor.py`: add a ground
plane, set a 2 ms timestep, and replace UR5e's position servos with unit-gain
torque actuators using the original force bounds. Native mass, inertia,
geometry and collision settings are preserved. The two base XMLs are local
example-defined models.

The allocation port is based on [PX4-Autopilot v1.17.0](https://github.com/PX4/PX4-Autopilot/tree/v1.17.0/src/lib/control_allocation/control_allocation):

- `ControlAllocationPseudoInverse.cpp`: pseudo-inverse column normalization.
- `ControlAllocationSequentialDesaturation.cpp`: `mixAirmodeRP`, `mixYaw`,
  `desaturateActuators`, `computeDesaturationGain`.
- `ControlAllocation.cpp`: final actuator clipping.

Its copyright/license is retained verbatim in `PX4-LICENSE`. Adaptations use
NumPy float64, four active wrench axes, the loaded X2 geometry, normalized
rotor thrust and explicit body-FLU/FRD conversion. Source references also appear
beside the corresponding implementation.
