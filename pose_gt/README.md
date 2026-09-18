# Pose GT — AprilTag Ground-Truth Overlay

Estimate the 6-DoF pose of an object from a jig of 4 AprilTag (36h11) markers and
overlay the object CAD live on an Intel RealSense D435/D435i RGB stream. Used to
produce ground-truth poses for benchmarking MegaPose.

The pose solver is coplanar-safe: it uses IPPE PnP over the tag corners, a
calibrated corner-order shift, and an out-of-plane flip resolver, so **any single
one of the 4 tags is enough** to recover the object pose (all 4 are redundant).

---

## 1. Environment setup

Requires [conda](https://docs.conda.io/) and a RealSense D435/D435i on a USB 3
port for the live overlay.

```bash
# create the env (named "cv")
conda create -n cv python=3.11 -y
conda activate cv

# install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs: `opencv-contrib-python` (cv2.aruco AprilTag + PnP),
`numpy`, `scipy`, `trimesh` + `pygltflib` + `pillow` (CAD load), and
`pyrealsense2` (camera).

---

## 2. Run the live overlay

```bash
python live_overlay.py --config objects/matka/jig1_config.json
# or
python live_overlay.py --config objects/matka/jig2_config.json
```

It streams the RealSense RGB at the highest resolution it can negotiate (falls
back 1080p → 720p → 480p automatically), detects the tags, and draws:

- each detected tag outline + id;
- a coloured dot at the object origin predicted by **each tag individually**
  (verify every `object → tag` transform — the dots should coincide);
- the fused **object frame** (large RGB triad) and the **STL frame** (small triad);
- the object CAD as a translucent shaded surface.

### Keys

| key | action |
|-----|--------|
| `s` | capture: freeze the pose and save `captured_pose_<jig>.json` |
| `c` | clear the frozen capture (back to live) |
| `m` | toggle the CAD overlay on/off |
| `+` / `-` | CAD opacity up / down |
| `q` / `Esc` | quit |

### Captured output

Pressing `s` writes `captured_pose_<jig>.json` containing the camera intrinsics
and the **camera → object** transform (and the same derived from each tag alone),
each with the rotation expressed as **RPY** (deg + rad), quaternion `[x,y,z,w]`,
Rodrigues `rvec`, and a full 4×4 matrix. RPY uses the ROS-`tf` / scipy `'xyz'`
fixed-axis convention.

---

## 3. Config files (JSON)

All jig geometry lives in a JSON config — the code reads everything from it, so a
new jig or a **different object** needs no code changes. See
[`objects/matka/jig1_config.json`](objects/matka/jig1_config.json).

```jsonc
{
  "jig_name": "jig1",
  "apriltag": {
    "dictionary": "DICT_APRILTAG_36h11",  // any cv2.aruco predefined dict
    "tag_size_mm": 20.0,                  // black-square edge length
    "corner_shift": 0                     // detector corner-order alignment
  },
  "stl_to_object_mm": [0.0, -15.0, 0.0],  // STL geometry origin vs object frame
  "object_to_tag": {                      // full object->tag pose per tag
    "0": { "xyz_mm": [ 32.5, -15.78,  32.5], "rpy_deg": [90.0, 0.0, 0.0] },
    "1": { "xyz_mm": [-32.5, -15.78,  32.5], "rpy_deg": [90.0, 0.0, 0.0] },
    "2": { "xyz_mm": [-32.5, -15.78, -32.5], "rpy_deg": [90.0, 0.0, 0.0] },
    "3": { "xyz_mm": [ 32.5, -15.78, -32.5], "rpy_deg": [90.0, 0.0, 0.0] }
  },
  "reproj_err_max_px": 3.0,
  "min_tags": 1,
  "cad_stl": "matka.stl"                  // resolved beside this config
}
```

Each tag entry under `object_to_tag` is a full **object → tag pose**: `xyz_mm`
(tag centre in the object frame) and `rpy_deg` (tag orientation, ROS-`tf` / scipy
`'xyz'` fixed-axis). `rpy_deg [90,0,0]` lays a tag flat in the object x–z plane
with its normal along object +y.

**To benchmark a different object:** choose its folder under `objects/`, then
edit that object's copied `jig1_config.json` and `jig2_config.json` parameters.
The `cad_stl` path is resolved relative to the config file, so each object folder
is self-contained. Fill in `object_to_tag` poses measured in that object's own
frame. `live_overlay.py` is object-agnostic.

Frames use the OpenCV camera convention (x-right, y-down, z-forward). The object
frame is the STL-native origin; each tag frame is +x = u, +y = v, +z = normal out
of the printed face.

---

## Files

| file | purpose |
|------|---------|
| `live_overlay.py`   | generic live overlay (takes `--config`) |
| `objects/<object>/jig1_config.json` | jig 1 geometry for that object |
| `objects/<object>/jig2_config.json` | jig 2 geometry for that object |
| `objects/<object>/<object>.stl`     | that object's CAD mesh |
| `requirements.txt`  | Python dependencies |
