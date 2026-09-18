"""Pure geometry helpers for an X-Z camera orbit around an object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OrbitPoint:
    index: int
    radius_m: float
    angle_deg: float
    position_m: tuple[float, float, float]
    pitch_deg: float
    quat_xyzw: tuple[float, float, float, float]


def float_range(start: float, stop: float, step: float, *, inclusive: bool) -> list[float]:
    """Return a stable floating-point range without cumulative drift."""
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be >= start")
    epsilon = step * 1e-9
    count = math.floor((stop - start + (epsilon if inclusive else -epsilon)) / step) + 1
    return [start + i * step for i in range(max(0, count))]


def quat_to_euler_xyz(quat_xyzw: Sequence[float]) -> tuple[float, float, float]:
    """Quaternion to fixed roll, pitch, yaw (radians)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Fixed roll, pitch, yaw (radians) to xyzw quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_multiply_xyzw(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    """Hamilton product ``a * b`` for xyzw quaternions.

    Multiplying the reference orientation by a rotation on the right applies
    that rotation intrinsically, around the tool's local axes.
    """
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    out = (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
    norm = math.sqrt(sum(v * v for v in out))
    if norm < 1e-12:
        raise ValueError("quaternion product is zero")
    return tuple(v / norm for v in out)


def pointing_pitch_rad(angle_rad: float, forward_axis: str) -> float:
    """Pitch that points the selected camera/TCP axis along camera->object.

    ``angle_rad`` is atan2(dz, dx), i.e. the angle from base-frame +X to
    the vector from the camera to the object in the X-Z plane.
    """
    formulas = {
        "+x": -angle_rad,
        "-x": math.pi - angle_rad,
        "+z": math.pi / 2 - angle_rad,
        "-z": -math.pi / 2 - angle_rad,
    }
    try:
        return formulas[forward_axis]
    except KeyError as exc:
        raise ValueError(f"unsupported forward axis: {forward_axis}") from exc


def build_orbit(
    center_m: Sequence[float],
    reference_quat_xyzw: Sequence[float],
    radii_m: Iterable[float],
    angles_deg: Sequence[float],
    *,
    forward_axis: str = "+z",
    pitch_offset_deg: float = 0.0,
    pitch_frame: str = "base_y",
    reference_angle_deg: float = -90.0,
    reverse_first: bool = False,
) -> list[OrbitPoint]:
    """Build a serpentine set of fixed-Y X-Z semicircles.

    For angle ``a``, camera->object is ``r * [cos(a), 0, sin(a)]`` and
    camera position is therefore ``object - that vector``.
    """
    cx, cy, cz = (float(v) for v in center_m)
    roll, _, yaw = quat_to_euler_xyz(reference_quat_xyzw)
    reference_pointing_pitch = pointing_pitch_rad(
        math.radians(reference_angle_deg), forward_axis
    )
    points: list[OrbitPoint] = []
    for radius_index, radius in enumerate(radii_m):
        if radius <= 0:
            raise ValueError("radius must be positive")
        reverse_this_radius = bool(radius_index % 2) ^ bool(reverse_first)
        ordered_angles = list(reversed(angles_deg)) if reverse_this_radius else angles_deg
        for angle_deg in ordered_angles:
            angle = math.radians(angle_deg)
            pointing_pitch = pointing_pitch_rad(angle, forward_axis)
            pitch = (pointing_pitch - reference_pointing_pitch
                     + math.radians(pitch_offset_deg))
            pitch_quat = euler_xyz_to_quat(0.0, pitch, 0.0)
            if pitch_frame == "base_y":
                # Extrinsic rotation about robot base-frame Y. Pre-multiplying
                # is essential: post-multiplication would rotate about tool Y.
                quat = quat_multiply_xyzw(pitch_quat, reference_quat_xyzw)
            elif pitch_frame == "local":
                # Intrinsic tool/camera Y rotation. At reference_angle_deg the
                # recorded quaternion is used unchanged (apart from offset).
                quat = quat_multiply_xyzw(reference_quat_xyzw, pitch_quat)
            elif pitch_frame == "base_euler":
                # Retained only for explicit compatibility with early plans.
                pitch = pointing_pitch + math.radians(pitch_offset_deg)
                quat = euler_xyz_to_quat(roll, pitch, yaw)
            else:
                raise ValueError(f"unsupported pitch frame: {pitch_frame}")
            points.append(OrbitPoint(
                index=len(points),
                radius_m=float(radius),
                angle_deg=float(angle_deg),
                position_m=(
                    cx - radius * math.cos(angle),
                    cy,
                    cz - radius * math.sin(angle),
                ),
                pitch_deg=math.degrees(pitch),
                quat_xyzw=quat,
            ))
    return points
