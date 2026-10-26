"""Independent serial-chain forward-kinematics reference."""

from __future__ import annotations

import jax.numpy as jnp


def _rotation(quaternion, dtype):
    w, x, y, z = jnp.asarray(quaternion, dtype=dtype)
    return jnp.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=dtype,
    )


def _compose(rotation, translation, pose, dtype):
    local_rotation = _rotation(pose.orientation, dtype)
    local_translation = jnp.asarray(pose.position, dtype=dtype)
    return rotation @ local_rotation, translation + rotation @ local_translation


def _axis_angle(axis, angle):
    axis = jnp.asarray(axis, dtype=angle.dtype)
    skew = jnp.asarray(
        (
            (0, -axis[2], axis[1]),
            (axis[2], 0, -axis[0]),
            (-axis[1], axis[0], 0),
        ),
        dtype=angle.dtype,
    )
    identity = jnp.eye(3, dtype=angle.dtype)
    sine, cosine = jnp.sin(angle), jnp.cos(angle)
    return cosine * identity + (1 - cosine) * jnp.outer(axis, axis) + sine * skew


def forward(chain, q):
    """Return collision-sphere centers and tip position in the world frame."""
    dtype = q.dtype
    rotation = _rotation(chain.base.orientation, dtype)
    translation = jnp.asarray(chain.base.position, dtype=dtype)
    centers = [None] * len(chain.collision_spheres)

    def emit(link, rotation, translation):
        for index, sphere in enumerate(chain.collision_spheres):
            if sphere.link == link:
                centers[index] = translation + rotation @ jnp.asarray(
                    sphere.center, dtype=dtype
                )

    emit(0, rotation, translation)
    for index, joint in enumerate(chain.joints):
        rotation, translation = _compose(rotation, translation, joint.origin, dtype)
        axis = jnp.asarray(joint.axis, dtype=dtype)
        if joint.type.name == "PRISMATIC":
            translation = translation + rotation @ (axis * q[index])
        else:
            rotation = rotation @ _axis_angle(axis, q[index])
        emit(index + 1, rotation, translation)
    _, tip = _compose(rotation, translation, chain.tip, dtype)
    if centers:
        return jnp.stack(centers), tip
    return jnp.zeros((0, 3), dtype=dtype), tip
