import math
import unittest

from path_geometry import (
    build_orbit,
    float_range,
    pointing_pitch_rad,
    quat_multiply_xyzw,
)


class PathGeometryTests(unittest.TestCase):
    def test_default_counts(self):
        radii = float_range(10, 50, 1, inclusive=False)
        angles = float_range(0, 180, 0.5, inclusive=True)
        self.assertEqual(len(radii), 40)
        self.assertEqual(len(angles), 361)

    def test_radius_order_can_be_reversed(self):
        radii = float_range(10, 50, 1, inclusive=False)
        radii.reverse()
        self.assertEqual(len(radii), 40)
        self.assertEqual(radii[0], 49)
        self.assertEqual(radii[-1], 10)

    def test_position_and_fixed_y(self):
        points = build_orbit([1, 2, 3], [0, 0, 0, 1], [0.1], [0, 90, 180])
        self.assertAlmostEqual(points[0].position_m[0], 0.9)
        self.assertAlmostEqual(points[1].position_m[2], 2.9)
        self.assertAlmostEqual(points[2].position_m[0], 1.1)
        self.assertTrue(all(p.position_m[1] == 2 for p in points))

    def test_serpentine_radius_transition(self):
        points = build_orbit([0, 0, 0], [0, 0, 0, 1], [0.1, 0.11], [0, 90, 180])
        self.assertEqual([p.angle_deg for p in points], [0, 90, 180, 180, 90, 0])
        distance = math.dist(points[2].position_m, points[3].position_m)
        self.assertAlmostEqual(distance, 0.01)

    def test_reverse_first_serpentine(self):
        points = build_orbit(
            [0, 0, 0], [0, 0, 0, 1], [0.1, 0.11, 0.12], [0, 90, 180],
            reverse_first=True,
        )
        self.assertEqual(
            [p.angle_deg for p in points],
            [180, 90, 0, 0, 90, 180, 180, 90, 0],
        )
        self.assertAlmostEqual(math.dist(points[2].position_m, points[3].position_m), 0.01)
        self.assertAlmostEqual(math.dist(points[5].position_m, points[6].position_m), 0.01)

    def test_plus_z_pitch_points_at_angle(self):
        self.assertAlmostEqual(pointing_pitch_rad(0, "+z"), math.pi / 2)
        self.assertAlmostEqual(pointing_pitch_rad(math.pi / 2, "+z"), 0)
        self.assertAlmostEqual(pointing_pitch_rad(math.pi, "+z"), -math.pi / 2)

    def test_local_pitch_is_intrinsic_tool_y(self):
        reference = [0.6947532728, 0.7192481419, 0.0, 0.0]
        points = build_orbit(
            [0, 0, 0], reference, [0.1], [-120, -90, -60],
            forward_axis="+z", pitch_frame="local", reference_angle_deg=-90,
        )
        self.assertAlmostEqual(points[0].pitch_deg, 30.0)
        self.assertAlmostEqual(points[1].pitch_deg, 0.0)
        self.assertAlmostEqual(points[2].pitch_deg, -30.0)
        expected = quat_multiply_xyzw(
            reference, [0.0, math.sin(math.radians(15)), 0.0, math.cos(math.radians(15))]
        )
        dot = abs(sum(a * b for a, b in zip(points[0].quat_xyzw, expected)))
        self.assertAlmostEqual(dot, 1.0)

    def test_base_pitch_is_extrinsic_robot_y(self):
        reference = [0.6947532728, 0.7192481419, 0.0, 0.0]
        points = build_orbit(
            [0, 0, 0], reference, [0.1], [-120, -90, -60],
            forward_axis="+z", pitch_frame="base_y", reference_angle_deg=-90,
        )
        base_y_30 = [0.0, math.sin(math.radians(15)), 0.0, math.cos(math.radians(15))]
        expected = quat_multiply_xyzw(base_y_30, reference)
        dot = abs(sum(a * b for a, b in zip(points[0].quat_xyzw, expected)))
        self.assertAlmostEqual(dot, 1.0)
        self.assertAlmostEqual(points[0].pitch_deg, 30.0)


if __name__ == "__main__":
    unittest.main()
