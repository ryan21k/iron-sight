"""Projected sagittal geometry and causal smoothing, independent of video I/O."""
from math import atan2, degrees, hypot, pi

SIDES = {
    "left": (11, 23, 25, 27, 29, 31),
    "right": (12, 24, 26, 28, 30, 32),
}


def vector(start, end):
    # OpenCV coordinates point down; use a Cartesian vertical axis.
    return end[0] - start[0], start[1] - end[1]


def angle(u, v):
    if min(hypot(*u), hypot(*v)) < 1e-6:
        return None
    return degrees(atan2(abs(u[0] * v[1] - u[1] * v[0]), u[0] * v[0] + u[1] * v[1]))


def measure(points, side):
    shoulder, hip, knee, ankle, heel, toe = (
        points.get(index) for index in SIDES[side]
    )
    result = {}
    if hip and knee and ankle:
        included = angle(vector(knee, hip), vector(knee, ankle))
        if included is not None:
            result["knee_flexion_deg"] = 180 - included
    if shoulder and hip:
        trunk = vector(hip, shoulder)
        inclination = angle(trunk, (0, 1))
        if inclination is not None:
            result["torso_inclination_deg"] = inclination
        if knee:
            included = angle(trunk, vector(hip, knee))
            if included is not None:
                result["hip_flexion_proxy_deg"] = 180 - included
    if knee and ankle and heel and toe:
        included = angle(vector(ankle, knee), vector(heel, toe))
        if included is not None:
            result["ankle_dorsiflexion_proxy_deg"] = 90 - included
    return result


class OneEuro:
    """Filter each coordinate using its own smoothed rate of change."""

    def __init__(self, min_cutoff=1.5, beta=0.02):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.previous = None

    @staticmethod
    def alpha(cutoff, dt):
        rate = 2 * pi * cutoff * dt
        return rate / (1 + rate)

    def update(self, value, time):
        filtered, derivative = value, 0.0
        if self.previous is not None:
            previous_time, raw, filtered, derivative = self.previous
            dt = time - previous_time
            derivative += self.alpha(1.0, dt) * ((value - raw) / dt - derivative)
            cutoff = self.min_cutoff + self.beta * abs(derivative)
            filtered += self.alpha(cutoff, dt) * (value - filtered)
        self.previous = time, value, filtered, derivative
        return filtered
