"""Live RealSense D435/D435i RGB overlay for an AprilTag object jig.

The jig geometry is loaded from a JSON config file:
    python live_overlay.py --config objects/matka/jig1_config.json
    python live_overlay.py --config objects/white/jig2_config.json

Pose solver adapted from classical-pose-estimation-main/ground-truth/apriltag_gt.py
(coplanar-safe): IPPE PnP over all visible tag corners in the object frame, a
calibrated corner-order shift so a SINGLE tag resolves uniquely, and an
out-of-plane flip resolver so the object z axis points toward the camera. Any
subset of the 4 tags (1/2/3/4 -- all redundant) yields the object pose.

Overlay: detected tags + ids, per-tag object-origin markers (verify each
april->object transform), the fused OBJECT frame, the STL frame, and the CAD as
a translucent shaded surface.

Keys: q quit | s capture (-> captured_pose.json) | c clear | m cad on/off |
      +/- opacity
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class JigConfig:
    """Loads jig geometry from a JSON file (single source of truth)."""

    def __init__(self, path):
        path = Path(path)
        if not path.is_absolute():
            # Accept paths relative to either the current working directory or
            # pose_gt, which keeps CLI use convenient from both locations.
            path = path.resolve() if path.exists() else (HERE / path).resolve()
        with open(path) as fp:
            d = json.load(fp)
        self.path = path
        self.name = d.get("jig_name", path.stem)

        ap = d["apriltag"]
        self.dict_name = ap["dictionary"]
        self.tag_size_mm = float(ap["tag_size_mm"])
        self.corner_shift = int(ap.get("corner_shift", 0))

        self.stl_to_object_mm = np.array(d["stl_to_object_mm"], dtype=float)

        # Each tag has a FULL object->tag pose: position (xyz_mm) + orientation
        # (rpy_deg, ROS 'xyz' fixed-axis). Legacy position-only configs
        # ("object_to_tag_centers_mm" + "tag_plane_u/v") are still accepted.
        self.tag_pose = {}          # {id: (t_xyz_mm, R_obj_tag)}
        if "object_to_tag" in d:
            for k, v in d["object_to_tag"].items():
                t = np.array(v["xyz_mm"], dtype=float)
                R = rpy_to_rot(np.radians(v.get("rpy_deg", [90.0, 0.0, 0.0])))
                self.tag_pose[int(k)] = (t, R)
        else:  # legacy schema
            u = np.array(d.get("tag_plane_u", [1, 0, 0]), dtype=float)
            v_ = np.array(d.get("tag_plane_v", [0, 0, 1]), dtype=float)
            R = np.column_stack([u, v_, np.cross(u, v_)])
            for k, c in d["object_to_tag_centers_mm"].items():
                self.tag_pose[int(k)] = (np.array(c, dtype=float), R)

        self.reproj_err_max_px = float(d.get("reproj_err_max_px", 3.0))
        self.min_tags = int(d.get("min_tags", 1))
        self.cad_stl = d.get("cad_stl", "matka.stl")

    def tag_corners_obj(self):
        """{id: (4,3) corner coords in the object frame (mm)}, [bl,br,tr,tl].

        Corners are the tag square's own [bl,br,tr,tl] transformed by the tag's
        full object->tag pose, so BOTH position and orientation come from the
        config."""
        h = self.tag_size_mm / 2.0
        local = np.array([[-h, -h, 0], [+h, -h, 0],
                          [+h, +h, 0], [-h, +h, 0]], dtype=float)  # tag frame
        out = {}
        for tid, (t, R) in self.tag_pose.items():
            out[tid] = (R @ local.T).T + t
        return out


# --------------------------------------------------------------------------- #
# small math helpers
# --------------------------------------------------------------------------- #
def _reorder(c, shift):
    return c[np.roll(np.arange(4), shift)]


def rpy_to_rot(rpy):
    """(roll, pitch, yaw) rad -> rotation matrix, XYZ fixed-axis (ROS tf / scipy
    'xyz'):  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_to_rpy(R):
    """Rotation matrix -> (roll, pitch, yaw) rad, XYZ fixed-axis (ROS tf / scipy
    'xyz'):  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    sy = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def rot_to_quat(R):
    """Rotation matrix -> quaternion [x, y, z, w] (tf convention)."""
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    return q.tolist()


def pose_to_dict(R, t_mm):
    """Full camera->object (or ->tag) TF as JSON-friendly dict: RPY (deg+rad),
    quaternion, rvec, matrix."""
    roll, pitch, yaw = rot_to_rpy(R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_mm
    return {
        "translation_mm": [float(v) for v in t_mm],
        "rpy_deg": [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)],
        "rpy_rad": [roll, pitch, yaw],
        "quaternion_xyzw": rot_to_quat(R),
        "rvec": cv2.Rodrigues(R)[0].flatten().tolist(),
        "matrix_4x4": T.tolist(),
    }


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def make_detector(dict_name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(d, params)


def detect_tags(gray, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    dets = []
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            dets.append((int(i), c.reshape(4, 2).astype(np.float64)))
    return dets


# --------------------------------------------------------------------------- #
# object pose (coplanar-safe)
# --------------------------------------------------------------------------- #
def _reproj(obj, img, R, t, K, dist):
    proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, dist)
    return float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())


def _resolve_planar_flip(obj, img, K, dist):
    """Coplanar tags give two IPPE poses (real + mirror). Disambiguate by:
      (1) object origin in front of camera (t_z > 0)
      (2) tag-plane normal (object +Y) faces the camera (its cam-z < 0)
      (3) tiebreak: reprojection error.
    Criterion (2) is what makes a SINGLE tag resolve correctly."""
    c0 = obj.mean(0)
    local = (obj - c0).astype(np.float64)
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(local, img, K, dist,
                                                 flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        _, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        return cv2.Rodrigues(rvec)[0], tvec.reshape(3)

    cands = []
    for rvec, tvec in zip(rvecs, tvecs):
        R = cv2.Rodrigues(rvec)[0]
        t = tvec.reshape(3) - R @ c0
        if t[2] <= 0:
            continue
        cands.append((R[2, 1], _reproj(obj, img, R, t, K, dist), R, t))
    if not cands:
        R = cv2.Rodrigues(rvecs[0])[0]
        return R, tvecs[0].reshape(3) - R @ c0
    facing = [c for c in cands if c[0] < 0]
    pool = facing if facing else cands
    pool.sort(key=lambda c: c[1])
    return pool[0][2], pool[0][3]


def solve_object_pose(dets, corners_obj, K, dist, corner_shift, sweep=False):
    """-> (R_cam_obj, t_cam_obj_mm, reproj_px, ids_used) or None. Any subset of
    tags works (all redundant)."""
    dist = np.zeros(5) if dist is None else dist
    usable = [(tid, c) for tid, c in dets if tid in corners_obj]
    if len(usable) < 1:
        return None
    shifts = range(4) if sweep else [corner_shift]
    best = None
    for shift in shifts:
        obj = np.concatenate([corners_obj[tid] for tid, _ in usable])
        img = np.concatenate([_reorder(c, shift) for _, c in usable])
        c0 = obj.mean(0)
        local = (obj - c0).astype(np.float64)
        try:
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(local, img, K, dist,
                                                     flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        for rvec, tvec in zip(rvecs, tvecs):
            R = cv2.Rodrigues(rvec)[0]
            t = tvec.reshape(3) - R @ c0
            err = _reproj(obj, img, R, t, K, dist)
            if best is None or err < best[0]:
                best = (err, obj, img)
    if best is None:
        return None
    _, obj, img = best
    R, t = _resolve_planar_flip(obj, img, K, dist)
    return R, t, _reproj(obj, img, R, t, K, dist), [tid for tid, _ in usable]


# --------------------------------------------------------------------------- #
# overlay drawing
# --------------------------------------------------------------------------- #
def draw_axes(img, R, t_mm, K, dist, L=30.0, thick=3):
    rvec = cv2.Rodrigues(R)[0]
    pts = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], np.float64)
    p, _ = cv2.projectPoints(pts, rvec, t_mm, K, dist)
    p = p.reshape(-1, 2).astype(int)
    for i, col in [(1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))]:
        cv2.line(img, tuple(p[0]), tuple(p[i]), col, thick, cv2.LINE_AA)


def draw_model_mesh(img, R, t_mm, K, dist, verts_mm, tris, alpha=0.45,
                    base=(0, 170, 255)):
    """Translucent shaded CAD: project, back-face cull, painter's sort, fill."""
    rvec = cv2.Rodrigues(R)[0]
    Vc = (R @ verts_mm.T).T + t_mm
    P = cv2.projectPoints(verts_mm, rvec, t_mm, K, dist)[0].reshape(-1, 2).astype(np.int32)
    tv = Vc[tris]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    order = np.where(n[:, 2] < 0)[0]
    order = order[np.argsort(tv[order][:, :, 2].mean(1))[::-1]]
    shade = np.clip(-n[:, 2], 0.25, 1.0)
    overlay = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in order:
        pts = P[tris[i]]
        s = shade[i]
        cv2.fillConvexPoly(overlay, pts,
                           (int(base[0]*s), int(base[1]*s), int(base[2]*s)))
        cv2.fillConvexPoly(mask, pts, 255)
    m = mask > 0
    img[m] = (alpha * overlay[m] + (1 - alpha) * img[m]).astype(np.uint8)


def load_cad(cfg):
    import trimesh
    p = Path(cfg.cad_stl)
    if not p.is_absolute():
        # Keep each object's STL and jig configs self-contained in one folder.
        p = cfg.path.parent / p
    mesh = trimesh.load(str(p))
    verts_mm = np.asarray(mesh.vertices, dtype=np.float64) + cfg.stl_to_object_mm
    return verts_mm, np.asarray(mesh.faces)


# --------------------------------------------------------------------------- #
# stream
# --------------------------------------------------------------------------- #
def start_stream():
    import pyrealsense2 as rs
    for (w, h, fps) in [(1920, 1080, 30), (1920, 1080, 15),
                        (1280, 720, 30), (848, 480, 30), (640, 480, 30)]:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            prof = pipe.start(cfg)
            pipe.wait_for_frames(4000)
            intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy], [0, 0, 1.0]])
            dist = np.array(intr.coeffs, dtype=np.float64)
            print(f"streaming {w}x{h}@{fps}  fx={intr.fx:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
            return pipe, K, dist, w, h
        except Exception as e:
            print(f"  {w}x{h}@{fps} unavailable ({e})")
            try:
                pipe.stop()
            except Exception:
                pass
    raise RuntimeError("no RealSense colour stream available")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="jig config JSON (e.g. objects/matka/jig1_config.json)")
    args = ap.parse_args()

    cfg = JigConfig(args.config)
    print(f"loaded config: {cfg.name}  ({cfg.path.name})")
    corners_obj = cfg.tag_corners_obj()

    import pyrealsense2 as rs  # noqa: F401
    verts_mm, tris = load_cad(cfg)
    print(f"CAD: {len(verts_mm)} verts, {len(tris)} tris")

    detector = make_detector(cfg.dict_name)
    pipe, K, dist, W, H = start_stream()

    cad_on = True
    alpha = 0.45
    captured = None
    tag_cols = {0: (255, 80, 80), 1: (80, 255, 80),
                2: (80, 80, 255), 3: (0, 200, 255)}

    win = f"object overlay [{cfg.name}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)

    def solve(d):
        return solve_object_pose(d, corners_obj, K, dist, cfg.corner_shift)

    try:
        while True:
            cf = pipe.wait_for_frames().get_color_frame()
            if not cf:
                continue
            img = np.asanyarray(cf.get_data()).copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detect_tags(gray, detector)

            for tid, c in dets:
                cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
                p = c.mean(0).astype(int)
                cv2.putText(img, f"id{tid}", (p[0] - 20, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

            # per-tag object origin markers (verify each april->object transform)
            per_origins = {}
            for tid, c in dets:
                if tid not in corners_obj:
                    continue
                r = solve([(tid, c)])
                if r:
                    Ri, ti, ei, _ = r
                    per_origins[tid] = (Ri, ti, ei)
                    o, _ = cv2.projectPoints(np.zeros((1, 3)), cv2.Rodrigues(Ri)[0], ti, K, dist)
                    o = tuple(o.reshape(2).astype(int))
                    cv2.circle(img, o, 7, tag_cols.get(tid, (255, 255, 255)), -1)
                    cv2.circle(img, o, 7, (0, 0, 0), 1)
                    cv2.putText(img, str(tid), (o[0] + 8, o[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                tag_cols.get(tid, (255, 255, 255)), 2, cv2.LINE_AA)

            if captured is not None:
                R, t, ids_used, err = captured["R"], captured["t"], captured["ids"], captured["err"]
                res_ok = True
            else:
                res = solve(dets)
                res_ok = res is not None
                if res_ok:
                    R, t, err, ids_used = res

            hud = [f"{cfg.name} | tags {sorted(t for t, _ in dets)} | "
                   f"{'CAPTURED' if captured else 'live'} | cad {'on' if cad_on else 'off'} a{alpha:.2f}"]
            if res_ok and len(ids_used) >= 1:
                if cad_on:
                    draw_model_mesh(img, R, t, K, dist, verts_mm, tris, alpha)
                draw_axes(img, R, t, K, dist, L=30, thick=4)              # object frame
                t_stl = t + R @ cfg.stl_to_object_mm
                draw_axes(img, R, t_stl, K, dist, L=20, thick=2)          # STL frame

                # small x/y/z frame at EACH tag, from its config pose:
                # T_cam_tag = T_cam_obj . T_obj_tag
                for tid, (t_ot, R_ot) in cfg.tag_pose.items():
                    R_ct = R @ R_ot
                    t_ct = R @ t_ot + t
                    draw_axes(img, R_ct, t_ct, K, dist,
                              L=cfg.tag_size_mm * 0.6, thick=2)

                gate = "OK" if err < cfg.reproj_err_max_px else "HIGH-ERR"
                hud.append(f"obj t(mm)=[{t[0]:.0f} {t[1]:.0f} {t[2]:.0f}] "
                           f"reproj={err:.2f}px({gate}) n={len(ids_used)}")

            if res_ok and per_origins:
                spread = " ".join(f"{tid}:{np.linalg.norm(ti - t):.0f}"
                                  for tid, (Ri, ti, ei) in sorted(per_origins.items()))
                hud.append("per-tag origin dev(mm) " + spread)

            hud.append("q quit | s capture | c clear | m cad | +/- opacity")
            for i, line in enumerate(hud):
                y = 34 + 30 * i
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (50, 255, 50), 1, cv2.LINE_AA)

            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                res = solve(dets)
                if res is not None:
                    R, t, err, ids_used = res
                    captured = {"R": R, "t": t, "err": err, "ids": ids_used}
                    per_tag = {}
                    for tid, c in dets:
                        if tid not in corners_obj:
                            continue
                        r = solve([(tid, c)])
                        if r:
                            Rt, tt, et, _ = r
                            dd = pose_to_dict(Rt, tt)
                            dd["reproj_px"] = et
                            per_tag[str(tid)] = dd
                    rec = {
                        "jig": cfg.name,
                        "config_file": cfg.path.name,
                        "resolution": [W, H],
                        "K": K.tolist(),
                        "dist": dist.tolist(),
                        "reproj_px": err,
                        "tags_used": ids_used,
                        "cam_to_object": pose_to_dict(R, t),
                        "cam_to_object_per_tag": per_tag,
                    }
                    out = HERE / f"captured_pose_{cfg.name}.json"
                    with open(out, "w") as fp:
                        json.dump(rec, fp, indent=2)
                    rpy = rec["cam_to_object"]["rpy_deg"]
                    print(f"\n[CAPTURE] {cfg.name} tags {ids_used} reproj {err:.2f}px")
                    print(f"  t(mm)=[{t[0]:.1f} {t[1]:.1f} {t[2]:.1f}]  "
                          f"rpy(deg)=[{rpy[0]:.1f} {rpy[1]:.1f} {rpy[2]:.1f}] -> {out.name}")
                else:
                    print("[CAPTURE] no valid tags")
            elif key == ord('c'):
                captured = None
                print("[capture cleared]")
            elif key == ord('m'):
                cad_on = not cad_on
            elif key in (ord('+'), ord('=')):
                alpha = min(1.0, alpha + 0.1)
            elif key in (ord('-'), ord('_')):
                alpha = max(0.1, alpha - 0.1)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
