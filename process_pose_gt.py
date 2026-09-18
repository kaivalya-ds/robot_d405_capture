"""Batch AprilTag object GT for a UR7e + D405 semicircle capture.

For every RGB frame:
  * one or more tags and a solved pose -> pose_gt/NNNNNN.json
  * zero detected AprilTags -> move RGB/depth pair under archive/
  * tags detected but pose unsolved -> leave pair in place and log the failure

The pose math and JSON pose representation are imported directly from
pose_gt/live_overlay.py so offline and live GT use the same implementation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
POSE_GT_ROOT = REPO_ROOT / "pose_gt"
sys.path.insert(0, str(POSE_GT_ROOT))
import live_overlay as GT  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_capture_manifest(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["index"])
            if index in records:
                raise ValueError(f"duplicate frame index {index} in {path}:{line_number}")
            records[index] = record
    return records


def camera_calibration(capture_config: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    stream = capture_config["camera"]["streams"]["color"]
    K = np.array([
        [stream["fx"], 0.0, stream["ppx"]],
        [0.0, stream["fy"], stream["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.asarray(stream["distortion_coeffs"], dtype=np.float64)
    return K, dist, [int(stream["width"]), int(stream["height"])]


def capture_metadata(record: dict) -> dict:
    return {
        key: record.get(key)
        for key in (
            "index",
            "radius_m",
            "angle_deg",
            "position_m",
            "pitch_deg",
            "quat_xyzw",
            "captured_utc",
            "camera_frame",
            "actual_tcp_pose",
        )
    }


def move_pair_to_archive(
    capture_root: Path,
    rgb_path: Path,
    depth_path: Path,
    archive_log,
) -> None:
    archive_rgb = capture_root / "archive" / "color_rgb" / rgb_path.name
    archive_depth = capture_root / "archive" / "depth_raw" / depth_path.name
    for destination in (archive_rgb, archive_depth):
        if destination.exists():
            raise FileExistsError(f"archive destination already exists: {destination}")

    # Move depth first and RGB last. Enumeration is driven by active RGB files,
    # so a completed pair cannot be selected again on a resumed run.
    shutil.move(str(depth_path), str(archive_depth))
    try:
        shutil.move(str(rgb_path), str(archive_rgb))
    except Exception:
        # Restore the depth half if the second move fails.
        shutil.move(str(archive_depth), str(depth_path))
        raise
    archive_log.write(json.dumps({
        "index": int(rgb_path.stem),
        "reason": "zero_apriltags_detected",
        "original_color_rgb": f"color_rgb/{rgb_path.name}",
        "original_depth_raw": f"depth_raw/{depth_path.name}",
        "archived_color_rgb": f"archive/color_rgb/{rgb_path.name}",
        "archived_depth_raw": f"archive/depth_raw/{depth_path.name}",
        "archived_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    archive_log.flush()


def process_capture(capture_root: Path, jig_config_path: Path) -> dict:
    capture_root = capture_root.resolve()
    jig_config_path = jig_config_path.resolve()
    if not capture_root.is_dir():
        raise NotADirectoryError(capture_root)

    rgb_dir = capture_root / "color_rgb"
    depth_dir = capture_root / "depth_raw"
    pose_dir = capture_root / "pose_gt"
    archive_dir = capture_root / "archive"
    archive_rgb_dir = archive_dir / "color_rgb"
    archive_depth_dir = archive_dir / "depth_raw"
    for required in (rgb_dir, depth_dir):
        if not required.is_dir():
            raise NotADirectoryError(required)
    pose_dir.mkdir(exist_ok=True)
    archive_rgb_dir.mkdir(parents=True, exist_ok=True)
    archive_depth_dir.mkdir(parents=True, exist_ok=True)

    cfg = GT.JigConfig(jig_config_path)
    corners_obj = cfg.tag_corners_obj()
    detector = GT.make_detector(cfg.dict_name)
    capture_config_path = capture_root / "capture_config.json"
    capture_config = json.loads(capture_config_path.read_text(encoding="utf-8"))
    K, dist, resolution = camera_calibration(capture_config)
    manifest_path = capture_root / "manifest.jsonl"
    capture_records = load_capture_manifest(manifest_path)
    rgb_files = sorted(rgb_dir.glob("*.png"))

    stats = {
        "schema": "offline-apriltag-gt-summary/1",
        "capture": capture_root.name,
        "capture_root": str(capture_root),
        "jig": cfg.name,
        "jig_config": str(jig_config_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_rgb_frames_at_start": len(rgb_files),
        "poses_written": 0,
        "poses_already_present": 0,
        "zero_tag_pairs_archived": 0,
        "tag_detected_pose_failures": 0,
        "missing_depth_pairs": 0,
        "unreadable_rgb_frames": 0,
        "high_reprojection_poses": 0,
    }
    archive_log_path = archive_dir / "archive_manifest.jsonl"
    failure_log_path = pose_dir / "pose_failures.jsonl"
    started = time.monotonic()

    with (
        archive_log_path.open("a", encoding="utf-8", buffering=1) as archive_log,
        failure_log_path.open("a", encoding="utf-8", buffering=1) as failure_log,
    ):
        for ordinal, rgb_path in enumerate(rgb_files, 1):
            index = int(rgb_path.stem)
            depth_path = depth_dir / f"{rgb_path.stem}.npy"
            gt_path = pose_dir / f"{rgb_path.stem}.json"
            if gt_path.exists():
                stats["poses_already_present"] += 1
                continue
            if not depth_path.is_file():
                stats["missing_depth_pairs"] += 1
                failure_log.write(json.dumps({
                    "index": index,
                    "reason": "missing_depth_pair",
                    "color_rgb": str(rgb_path),
                    "expected_depth_raw": str(depth_path),
                }) + "\n")
                continue

            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None:
                stats["unreadable_rgb_frames"] += 1
                failure_log.write(json.dumps({
                    "index": index,
                    "reason": "unreadable_rgb",
                    "color_rgb": str(rgb_path),
                }) + "\n")
                continue
            if [image.shape[1], image.shape[0]] != resolution:
                raise ValueError(
                    f"{rgb_path} resolution {image.shape[1]}x{image.shape[0]} "
                    f"does not match calibration {resolution[0]}x{resolution[1]}"
                )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = GT.detect_tags(gray, detector)
            if not detections:
                move_pair_to_archive(capture_root, rgb_path, depth_path, archive_log)
                stats["zero_tag_pairs_archived"] += 1
            else:
                solved = GT.solve_object_pose(
                    detections,
                    corners_obj,
                    K,
                    dist,
                    cfg.corner_shift,
                )
                if solved is None:
                    stats["tag_detected_pose_failures"] += 1
                    failure_log.write(json.dumps({
                        "index": index,
                        "reason": "tags_detected_pose_unsolved",
                        "tags_detected": [tag_id for tag_id, _ in detections],
                    }) + "\n")
                else:
                    R, t_mm, reproj_px, ids_used = solved
                    t_stl_mm = t_mm + R @ cfg.stl_to_object_mm
                    per_tag = {}
                    for tag_id, corners in detections:
                        if tag_id not in corners_obj:
                            continue
                        tag_solution = GT.solve_object_pose(
                            [(tag_id, corners)],
                            corners_obj,
                            K,
                            dist,
                            cfg.corner_shift,
                        )
                        if tag_solution is None:
                            continue
                        R_tag, t_tag_mm, tag_reproj_px, _ = tag_solution
                        tag_pose = GT.pose_to_dict(R_tag, t_tag_mm)
                        tag_pose["reproj_px"] = tag_reproj_px
                        per_tag[str(tag_id)] = tag_pose

                    quality_ok = bool(reproj_px <= cfg.reproj_err_max_px)
                    if not quality_ok:
                        stats["high_reprojection_poses"] += 1
                    record = {
                        "schema": "offline-apriltag-object-gt/1",
                        "frame_index": index,
                        "color_rgb_path": f"color_rgb/{rgb_path.name}",
                        "depth_raw_npy_path": f"depth_raw/{depth_path.name}",
                        "jig": cfg.name,
                        "jig_config": str(jig_config_path),
                        "resolution": resolution,
                        "K": K.tolist(),
                        "dist": dist.tolist(),
                        "tags_detected": [tag_id for tag_id, _ in detections],
                        "tags_used": ids_used,
                        "reproj_px": reproj_px,
                        "reproj_err_max_px": cfg.reproj_err_max_px,
                        "quality_ok": quality_ok,
                        "cam_to_object": GT.pose_to_dict(R, t_mm),
                        "cam_to_stl": GT.pose_to_dict(R, t_stl_mm),
                        "cam_to_object_per_tag": per_tag,
                        "capture": capture_metadata(capture_records.get(index, {"index": index})),
                    }
                    atomic_json(gt_path, record)
                    stats["poses_written"] += 1

            if ordinal % 100 == 0 or ordinal == len(rgb_files):
                elapsed = time.monotonic() - started
                rate = ordinal / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{capture_root.name}] {ordinal}/{len(rgb_files)} "
                    f"GT={stats['poses_written']} "
                    f"archived={stats['zero_tag_pairs_archived']} "
                    f"failed={stats['tag_detected_pose_failures']} "
                    f"rate={rate:.1f} fps",
                    flush=True,
                )

    stats["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stats["elapsed_s"] = time.monotonic() - started
    stats["pose_json_files_total"] = len([
        path for path in pose_dir.glob("*.json")
        if path.name != "batch_summary.json"
    ])
    stats["archived_rgb_files_total"] = len(list(archive_rgb_dir.glob("*.png")))
    stats["archived_depth_files_total"] = len(list(archive_depth_dir.glob("*.npy")))
    stats["active_rgb_files_total"] = len(list(rgb_dir.glob("*.png")))
    stats["active_depth_files_total"] = len(list(depth_dir.glob("*.npy")))
    atomic_json(pose_dir / "batch_summary.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--jig-config", type=Path, required=True)
    args = parser.parse_args()
    summary = process_capture(args.capture, args.jig_config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
