"""Export evenly distributed RGB/depth/GT subsets and ZIP them."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def spread_extra_indices(group_count: int, extra_count: int) -> set[int]:
    if extra_count <= 0:
        return set()
    if extra_count == 1:
        return {group_count // 2}
    return {
        round(i * (group_count - 1) / (extra_count - 1))
        for i in range(extra_count)
    }


def evenly_spaced(records: list[dict], count: int) -> list[dict]:
    records = sorted(records, key=lambda record: (record["angle_deg"], record["index"]))
    if len(records) < count:
        raise ValueError(f"need {count} records but group contains only {len(records)}")
    if count == 1:
        return [records[len(records) // 2]]
    indices = [round(i * (len(records) - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("even selection produced duplicate indices")
    return [records[index] for index in indices]


def select_variety(capture_root: Path, count: int) -> list[dict]:
    gt_dir = capture_root / "pose_gt"
    groups: dict[float, list[dict]] = defaultdict(list)
    for record in read_manifest(capture_root / "manifest.jsonl"):
        stem = f"{int(record['index']):06d}"
        if (gt_dir / f"{stem}.json").is_file():
            groups[round(float(record["radius_m"]), 6)].append(record)
    if not groups:
        raise RuntimeError(f"no GT frames found in {capture_root}")
    if sum(len(records) for records in groups.values()) < count:
        raise ValueError(f"{capture_root} contains fewer than {count} GT frames")

    radii = sorted(groups)
    base_count, extras = divmod(count, len(radii))
    extra_indices = spread_extra_indices(len(radii), extras)
    selected = []
    for radius_index, radius in enumerate(radii):
        target = base_count + (1 if radius_index in extra_indices else 0)
        selected.extend(evenly_spaced(groups[radius], target))
    selected.sort(key=lambda record: int(record["index"]))
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} records, expected {count}")
    return selected


def copy_subset(
    capture_root: Path,
    jig_config: Path,
    destination: Path,
    count: int,
) -> dict:
    selected = select_variety(capture_root, count)
    for subdirectory in ("color_rgb", "depth_raw", "pose_gt"):
        (destination / subdirectory).mkdir(parents=True, exist_ok=False)

    jobs: list[tuple[Path, Path]] = []
    for record in selected:
        stem = f"{int(record['index']):06d}"
        jobs.extend([
            (capture_root / "color_rgb" / f"{stem}.png",
             destination / "color_rgb" / f"{stem}.png"),
            (capture_root / "depth_raw" / f"{stem}.npy",
             destination / "depth_raw" / f"{stem}.npy"),
            (capture_root / "pose_gt" / f"{stem}.json",
             destination / "pose_gt" / f"{stem}.json"),
        ])
    missing = [source for source, _ in jobs if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing subset source files, first: {missing[0]}")

    def copy_job(job: tuple[Path, Path]) -> None:
        shutil.copy2(*job)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(copy_job, jobs))

    selected_manifest = destination / "manifest.jsonl"
    with selected_manifest.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record) + "\n")
    shutil.copy2(capture_root / "capture_config.json", destination / "capture_config.json")
    shutil.copy2(jig_config, destination / jig_config.name)

    radius_counts: dict[str, int] = defaultdict(int)
    for record in selected:
        radius_counts[f"{float(record['radius_m']):.2f}"] += 1
    summary = {
        "schema": "gt-variety-subset/1",
        "source_capture": str(capture_root.resolve()),
        "jig_config": jig_config.name,
        "sample_count": len(selected),
        "selection": (
            "all available radii represented; allocation balanced across radii; "
            "angles sampled evenly within each radius; valid GT frames only"
        ),
        "source_frame_indices": [int(record["index"]) for record in selected],
        "radius_sample_counts": dict(sorted(radius_counts.items(), key=lambda item: float(item[0]))),
        "angle_min_deg": min(float(record["angle_deg"]) for record in selected),
        "angle_max_deg": max(float(record["angle_deg"]) for record in selected),
        "files": {
            "rgb_png": len(selected),
            "depth_npy": len(selected),
            "pose_gt_json": len(selected),
        },
    }
    (destination / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def zip_folder(folder: Path, output_zip: Path) -> None:
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for ordinal, path in enumerate(files, 1):
            archive.write(path, arcname=path.relative_to(folder.parent))
            if ordinal % 250 == 0 or ordinal == len(files):
                print(f"[{folder.name}] zipped {ordinal}/{len(files)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    specs = [
        (
            "jig2_1000",
            Path("robot_d405_capture/captures/20260722_225850"),
            Path("pose_gt/jig2_config.json"),
        ),
        (
            "jig1_1000",
            Path("robot_d405_capture/captures/20260723_103829"),
            Path("pose_gt/jig1_config.json"),
        ),
    ]
    summaries = {}
    for label, capture_root, jig_config in specs:
        print(f"[{label}] selecting and copying {args.count} frames", flush=True)
        destination = output / label
        summaries[label] = copy_subset(
            capture_root.resolve(),
            jig_config.resolve(),
            destination,
            args.count,
        )
        zip_path = output / f"{label}.zip"
        zip_folder(destination, zip_path)
        summaries[label]["zip_path"] = str(zip_path)
        summaries[label]["zip_bytes"] = zip_path.stat().st_size
    (output / "export_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
