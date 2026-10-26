"""Explicit example calibrations for the two inequality profiles.

Each caller writes literal values. delta = width / (1.5 * 3) makes the barrier's
curvature at and beyond delta three times the hinge's peak curvature 1.5*rho/width;
barrier weight mu = rho * delta makes the push at delta equal the hinge maximum rho.
Activation is about 10% of the constraint's own scale and at least 3 * delta.
This module selects those values without converting them.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

from pivopt import problem as p

MODES = ("hinge", "relaxed-barrier")
Weight = TypeVar("Weight")


def add_constraint_argument(parser):
    parser.add_argument(
        "--constraints",
        choices=MODES,
        default="hinge",
        help="handle inequalities with the smoothed hinge or relaxed barrier",
    )


@dataclass(frozen=True)
class Calibration(Generic[Weight]):
    hinge_weight: Weight
    width: float
    barrier_weight: Weight
    delta: float
    activation: float | None = None


def term(kind, name, make, calibration, mode):
    if mode == "hinge":
        profile = p.Hinge(width=calibration.width)
        weight = calibration.hinge_weight
    elif mode == "relaxed-barrier":
        profile = p.RelaxedBarrier(
            delta=calibration.delta, activation=calibration.activation
        )
        weight = calibration.barrier_weight
    else:
        raise ValueError(f"unknown constraint mode: {mode!r}")
    return kind(name, cost=make(profile), weight=weight)
