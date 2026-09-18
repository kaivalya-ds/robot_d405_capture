"""Capture raw D405 RGB/depth frames while a UR7e follows X-Z semicircles."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from path_geometry import OrbitPoint, build_orbit, float_range


HERE = Path(__file__).resolve().parent


class RawD405:
    """Unaligned and unfiltered native D405 color/depth stream."""

    def __init__(self, serial: str | None, width: int, height: int, fps: int, warmup: int):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        device = profile.get_device()
        self.model = device.get_info(rs.camera_info.name)
        self.serial = device.get_info(rs.camera_info.serial_number)
        if "D405" not in self.model.upper():
            self.pipeline.stop()
            raise RuntimeError(f"expected a D405, opened {self.model} ({self.serial})")
        self.depth_scale_m = float(device.first_depth_sensor().get_depth_scale())
        self.streams = {}
        for stream_name, stream_kind in (("color", rs.stream.color), ("depth", rs.stream.depth)):
            intr = profile.get_stream(stream_kind).as_video_stream_profile().get_intrinsics()
            self.streams[stream_name] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "distortion_model": str(intr.model),
                "distortion_coeffs": list(intr.coeffs),
            }
        for _ in range(warmup):
            self.pipeline.wait_for_frames()

    def grab(self) -> tuple[np.ndarray, np.ndarray, dict]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("D405 returned an incomplete frameset")
        return (
            np.asanyarray(color_frame.get_data()).copy(),
            np.asanyarray(depth_frame.get_data()).copy(),
            {
                "frameset_number": int(frames.get_frame_number()),
                "color_frame_number": int(color_frame.get_frame_number()),
                "depth_frame_number": int(depth_frame.get_frame_number()),
                "color_timestamp_ms": float(color_frame.get_timestamp()),
                "depth_timestamp_ms": float(depth_frame.get_timestamp()),
            },
        )

    def metadata(self) -> dict:
        return {
            "model": self.model,
            "serial": self.serial,
            "depth_scale_m_per_unit": self.depth_scale_m,
            "streams": self.streams,
            "processing": "none; streams are raw, unaligned and unfiltered",
        }

    def stop(self) -> None:
        self.pipeline.stop()


def point_dict(point: OrbitPoint) -> dict:
    return {
        "index": point.index,
        "radius_m": point.radius_m,
        "angle_deg": point.angle_deg,
        "position_m": list(point.position_m),
        "pitch_deg": point.pitch_deg,
        "quat_xyzw": list(point.quat_xyzw),
    }


def save_frame(output: Path, point: OrbitPoint, rgb: np.ndarray, depth_raw: np.ndarray,
               frame_meta: dict, actual_pose: dict) -> dict:
    stem = f"{point.index:06d}"
    color_rel = Path("color_rgb") / f"{stem}.png"
    depth_rel = Path("depth_raw") / f"{stem}.npy"
    # OpenCV expects BGR for encoding; conversion makes the decoded PNG RGB-correct.
    if not cv2.imwrite(str(output / color_rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to save {color_rel}")
    if depth_raw.dtype != np.uint16:
        raise TypeError(f"expected uint16 raw depth, received {depth_raw.dtype}")
    np.save(output / depth_rel, depth_raw, allow_pickle=False)
    return {
        **point_dict(point),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "color_rgb_path": color_rel.as_posix(),
        "depth_raw_npy_path": depth_rel.as_posix(),
        "camera_frame": frame_meta,
        "actual_tcp_pose": actual_pose,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.101",
                        help="UR7e controller IP")
    parser.add_argument("--dex-bridge-root", type=Path,
                        default=Path(r"C:\Users\asus\dexsent\imp\dex-bridge"))
    parser.add_argument("--camera-serial", default="409122271039")
    parser.add_argument("--output", type=Path,
                        default=HERE / "captures" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="object center in base_link metres; skips jog-and-record")
    parser.add_argument("--reference-quat", nargs=4, type=float,
                        metavar=("QX", "QY", "QZ", "QW"),
                        help="reference TCP quaternion; required with --center")
    parser.add_argument("--center-file", type=Path,
                        help="JSON object-center record; skips jog-and-record")
    parser.add_argument(
        "--jig-config",
        type=Path,
        help="matching pose_gt jig JSON; required with --execute and processed "
             "automatically after a complete capture",
    )
    parser.add_argument("--min-radius-cm", type=float, default=10.0)
    parser.add_argument("--max-radius-cm", type=float, default=50.0)
    parser.add_argument("--radius-step-cm", type=float, default=1.0)
    parser.add_argument("--radius-order", choices=("min-to-max", "max-to-min"),
                        default="min-to-max",
                        help="capture radii in ascending or descending order")
    parser.add_argument("--include-max-radius", action="store_true",
                        help="include 50 cm, producing 41 radii with the defaults")
    parser.add_argument("--angle-start-deg", type=float, default=-180.0)
    parser.add_argument("--angle-end-deg", type=float, default=0.0)
    parser.add_argument("--angle-step-deg", type=float, default=0.5)
    parser.add_argument("--camera-forward-axis", choices=("+x", "-x", "+z", "-z"), default="+z",
                        help="TCP axis parallel to the D405 optical forward axis")
    parser.add_argument("--pitch-frame", choices=("base_y", "local", "base_euler"),
                        default="base_y",
                        help="rotation convention; base_y rotates about robot base Y")
    parser.add_argument("--reference-angle-deg", type=float, default=-90.0,
                        help="orbit angle at which the recorded quaternion is zero pitch")
    parser.add_argument("--start-endpoint", choices=("auto", "angle-start", "angle-end"),
                        default="auto",
                        help="first semicircle direction; auto chooses the nearest endpoint")
    parser.add_argument("--approach-angle-step-deg", type=float, default=5.0,
                        help="maximum angular step while safely approaching the first endpoint")
    parser.add_argument("--pitch-offset-deg", type=float, default=0.0,
                        help="fixed camera-mount pitch calibration offset")
    parser.add_argument("--speed-m-s", type=float, default=0.03)
    parser.add_argument("--accel-m-s2", type=float, default=0.10)
    parser.add_argument("--settle-s", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--execute", action="store_true",
                        help="enable robot movement and camera capture")
    args = parser.parse_args()
    if args.center is not None and args.center_file is not None:
        parser.error("use either --center or --center-file, not both")
    if args.center_file is not None and args.reference_quat is not None:
        parser.error("--reference-quat is read from --center-file")
    if (args.center is None) != (args.reference_quat is None):
        if args.center_file is None:
            parser.error("--center and --reference-quat must be supplied together")
    if args.center is None and not args.robot_ip:
        parser.error("--robot-ip is required to record the jogged center")
    for name in ("radius_step_cm", "angle_step_deg", "speed_m_s", "accel_m_s2"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.settle_s < 0:
        parser.error("--settle-s must be non-negative")
    if args.approach_angle_step_deg <= 0:
        parser.error("--approach-angle-step-deg must be positive")
    if args.jig_config is not None and not args.jig_config.is_file():
        parser.error(f"--jig-config does not exist: {args.jig_config}")
    if args.execute and args.jig_config is None:
        parser.error("--jig-config is required with --execute for automatic GT processing")
    return args


async def read_tcp_pose(driver) -> dict:
    state = await driver.get_state()
    pose = state.cartesian.tcp_pose
    if pose is None:
        raise RuntimeError(f"UR7e TCP pose unavailable; state={state.derived_mode}")
    if state.fault.fault:
        raise RuntimeError(f"UR7e fault: {state.fault.alarm_message}")
    return pose.to_dict()


async def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    driver = None
    camera = None

    if args.center_file is not None:
        center_record = json.loads(args.center_file.resolve().read_text(encoding="utf-8"))
        if center_record.get("frame") != "base":
            raise ValueError("--center-file frame must be 'base'")
        center = [float(v) for v in center_record["position_m"]]
        reference_quat = [float(v) for v in center_record["reference_quat_xyzw"]]
    elif args.center is not None:
        center = list(args.center)
        reference_quat = list(args.reference_quat)
    else:
        sys.path.insert(0, str(args.dex_bridge_root.resolve()))
        from dex_bridge.drivers.ur.ur7e_driver import Ur7eDriver

        driver = Ur7eDriver("ur7e_capture", {
            "ip": args.robot_ip,
            "driver_config": {"motion_timeout_s": 120.0},
        })
        await driver.connect()
        await driver.enable()
        await asyncio.to_thread(input,
            "Jog the TCP reference point to the OBJECT CENTER, stop jogging, then press Enter... ")
        recorded_pose = await read_tcp_pose(driver)
        center = recorded_pose["position_m"]
        reference_quat = recorded_pose["quat_xyzw"]

    radii_cm = float_range(args.min_radius_cm, args.max_radius_cm,
                           args.radius_step_cm, inclusive=args.include_max_radius)
    if args.radius_order == "max-to-min":
        radii_cm.reverse()
    angles_deg = float_range(args.angle_start_deg, args.angle_end_deg,
                             args.angle_step_deg, inclusive=True)
    build_kwargs = dict(
        forward_axis=args.camera_forward_axis,
        pitch_offset_deg=args.pitch_offset_deg,
        pitch_frame=args.pitch_frame,
        reference_angle_deg=args.reference_angle_deg,
    )
    # A dry plan cannot inspect the live TCP. For auto mode it shows the
    # angle-start-first variant; execution resolves and rewrites the plan.
    reverse_first = args.start_endpoint == "angle-end"
    points = build_orbit(
        center, reference_quat, (r / 100.0 for r in radii_cm), angles_deg,
        reverse_first=reverse_first,
        **build_kwargs,
    )
    config = {
        "schema": "ur7e-d405-semicircle-capture/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "object_center_base_link_m": center,
        "reference_quat_xyzw": reference_quat,
        "fixed_y_m": center[1],
        "radius_values_cm": radii_cm,
        "radius_order": args.radius_order,
        "angle_values_deg": angles_deg,
        "serpentine": True,
        "start_endpoint_requested": args.start_endpoint,
        "first_direction": "reverse" if reverse_first else "normal",
        "camera_forward_axis": args.camera_forward_axis,
        "pitch_frame": args.pitch_frame,
        "reference_angle_deg": args.reference_angle_deg,
        "pitch_offset_deg": args.pitch_offset_deg,
        "point_count": len(points),
        "motion": {"speed_m_s": args.speed_m_s, "accel_m_s2": args.accel_m_s2,
                   "settle_s": args.settle_s},
    }
    if args.jig_config is not None:
        config["jig_config"] = str(args.jig_config.resolve())
    write_json(output / "capture_config.json", config)
    with (output / "planned_poses.jsonl").open("w", encoding="utf-8") as plan_file:
        for point in points:
            plan_file.write(json.dumps(point_dict(point)) + "\n")

    print(f"Plan: {len(radii_cm)} semicircles x {len(angles_deg)} poses = {len(points)} frames")
    print(f"Object center [m]: {[round(v, 6) for v in center]}; fixed Y={center[1]:.6f}")
    print(f"Plan written to {output}")
    if not args.execute:
        print("Dry run complete. Inspect planned_poses.jsonl, then rerun with --execute.")
        if driver is not None:
            await driver.disconnect()
        return
    if driver is None:
        if not args.robot_ip:
            raise RuntimeError("--robot-ip is required with --execute")
        sys.path.insert(0, str(args.dex_bridge_root.resolve()))
        from dex_bridge.drivers.ur.ur7e_driver import Ur7eDriver
        driver = Ur7eDriver("ur7e_capture", {
            "ip": args.robot_ip,
            "driver_config": {"motion_timeout_s": 120.0},
        })
        await driver.connect()
        await driver.enable()

    # Select the closest endpoint, then use the recorded center pose as the
    # single non-capturing approach waypoint. The user has explicitly verified
    # that moving to this shared center pose is safe. From there, the first
    # capture move travels radially outward to the selected endpoint.
    current_pose = await read_tcp_pose(driver)
    normal_points = build_orbit(
        center, reference_quat, (r / 100.0 for r in radii_cm), angles_deg,
        reverse_first=False, **build_kwargs,
    )
    reverse_points = build_orbit(
        center, reference_quat, (r / 100.0 for r in radii_cm), angles_deg,
        reverse_first=True, **build_kwargs,
    )
    current_position = current_pose["position_m"]
    dx_to_object = center[0] - current_position[0]
    dz_to_object = center[2] - current_position[2]
    current_radius_m = math.hypot(dx_to_object, dz_to_object)
    current_angle_deg = (
        math.degrees(math.atan2(dz_to_object, dx_to_object))
        if current_radius_m > 1e-6
        else args.reference_angle_deg
    )
    if current_radius_m > 0.55:
        await driver.disconnect()
        raise RuntimeError("TCP is more than 55 cm from the object center; jog it nearer before capture")

    def angular_delta_deg(source: float, target: float) -> float:
        return (target - source + 180.0) % 360.0 - 180.0

    normal_delta = angular_delta_deg(current_angle_deg, normal_points[0].angle_deg)
    reverse_delta = angular_delta_deg(current_angle_deg, reverse_points[0].angle_deg)
    if args.start_endpoint == "angle-start":
        points, reverse_first = normal_points, False
    elif args.start_endpoint == "angle-end":
        points, reverse_first = reverse_points, True
    elif abs(reverse_delta) < abs(normal_delta):
        points, reverse_first = reverse_points, True
    else:
        points, reverse_first = normal_points, False

    approach_points = [OrbitPoint(
        index=-1,
        radius_m=0.0,
        angle_deg=args.reference_angle_deg,
        position_m=tuple(float(v) for v in center),
        pitch_deg=0.0,
        quat_xyzw=tuple(float(v) for v in reference_quat),
    )]
    config["first_direction"] = "reverse" if reverse_first else "normal"
    config["resolved_start_angle_deg"] = points[0].angle_deg
    config["safe_approach"] = {
        "strategy": "recorded_center_then_first_capture_pose",
        "recorded_center_position_m": center,
        "recorded_center_quat_xyzw": reference_quat,
        "current_radius_m": current_radius_m,
        "current_angle_deg": current_angle_deg,
        "first_capture_radius_m": points[0].radius_m,
        "first_capture_angle_deg": points[0].angle_deg,
        "waypoint_count": len(approach_points),
        "capture_at_approach_waypoints": False,
    }
    write_json(output / "capture_config.json", config)
    with (output / "planned_poses.jsonl").open("w", encoding="utf-8") as plan_file:
        for point in points:
            plan_file.write(json.dumps(point_dict(point)) + "\n")
    print(f"Center-first approach: current TCP -> recorded center "
          f"{[round(v, 6) for v in center]} -> first capture "
          f"r={points[0].radius_m * 100:.1f} cm "
          f"angle={points[0].angle_deg:.1f} deg; "
          f"first direction={'reverse' if reverse_first else 'normal'}")

    answer = await asyncio.to_thread(
        input, "Verify the plan, workspace, TCP, and camera axis. Type MOVE to start: ")
    if answer.strip() != "MOVE":
        print("Capture cancelled; no robot motion was commanded.")
        await driver.disconnect()
        return

    completed = 0
    try:
        from dex_bridge.core.pose import Pose
        (output / "color_rgb").mkdir()
        (output / "depth_raw").mkdir()
        camera = RawD405(args.camera_serial, args.width, args.height, args.fps,
                         args.warmup_frames)
        config["camera"] = camera.metadata()
        config["capture_started_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "capture_config.json", config)
        manifest_path = output / "manifest.jsonl"
        for approach_index, point in enumerate(approach_points, 1):
            print(f"[approach {approach_index}/{len(approach_points)}] "
                  f"r={point.radius_m * 100:.1f} cm angle={point.angle_deg:.1f} deg",
                  flush=True)
            await driver.move_cartesian_pose(
                Pose(position_m=list(point.position_m),
                     quat_xyzw=list(point.quat_xyzw), frame="base"),
                speed=args.speed_m_s, accel=args.accel_m_s2,
                motion_id=f"approach-{approach_index:03d}", timeout_s=120.0,
            )
        print("Safe approach complete; starting frame capture.", flush=True)
        with manifest_path.open("a", encoding="utf-8", buffering=1) as manifest:
            for point in points:
                target = Pose(position_m=list(point.position_m),
                              quat_xyzw=list(point.quat_xyzw), frame="base")
                await driver.move_cartesian_pose(
                    target, speed=args.speed_m_s, accel=args.accel_m_s2,
                    motion_id=f"capture-{point.index:06d}", timeout_s=120.0,
                )
                await asyncio.sleep(args.settle_s)
                rgb, depth_raw, frame_meta = await asyncio.to_thread(camera.grab)
                actual_pose = await read_tcp_pose(driver)
                record = save_frame(output, point, rgb, depth_raw, frame_meta, actual_pose)
                manifest.write(json.dumps(record) + "\n")
                completed += 1
                print(f"[{completed}/{len(points)}] r={point.radius_m * 100:.0f} cm "
                      f"angle={point.angle_deg:.1f} deg", flush=True)
    finally:
        config["completed_frames"] = completed
        config["capture_finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "capture_config.json", config)
        if driver is not None:
            try:
                await driver.stop()
            finally:
                await driver.disconnect()
        if camera is not None:
            camera.stop()

    if completed == len(points):
        print(
            f"Capture complete ({completed}/{len(points)}); calculating AprilTag GT "
            f"with {args.jig_config}...",
            flush=True,
        )
        from process_pose_gt import process_capture
        gt_summary = await asyncio.to_thread(
            process_capture,
            output,
            args.jig_config,
        )
        print("Automatic GT processing complete:")
        print(json.dumps(gt_summary, indent=2))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
