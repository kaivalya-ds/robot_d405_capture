# UR7e + D405 semicircle capture

This recorder treats the TCP pose you jog to as the object center. It creates
semicircles in the robot base-frame X-Z plane, keeps base-frame Y constant,
rotates around the robot base-frame Y axis, and captures one raw D405 RGB/depth
frameset at every pose.

With the defaults it uses radii 10 through 49 cm (40 semicircles) and angles
0 through 180 degrees inclusive in 0.5 degree increments (361 poses each), for
14,440 frames. Add `--include-max-radius` if 50 cm must also be included; that
produces 41 semicircles and 14,801 frames.

Radii run from minimum to maximum by default. Use
`--radius-order max-to-min` to begin at 49 cm (or 50 cm with
`--include-max-radius`) and finish at 10 cm.

Successive radii always alternate direction. At a semicircle endpoint the TCP
moves radially outward by 1 cm, then traverses the next semicircle in reverse;
it never crosses through the object to begin a new radius. On execution,
`--start-endpoint auto` chooses the endpoint requiring the least angular travel.
Before frame 0, and only after the operator types `MOVE`, the robot moves from
its current pose to the recorded shared center pose (position and quaternion).
It then moves radially outward from that center to the selected first capture
pose. The center pose is not captured. This center-first path assumes the
workspace between the current pose, recorded center, and first orbit endpoint
has been verified clear. After the final capture the robot stops at that pose
and does not return to the center or start pose.

## Geometry and camera mounting

For angle `a`, measured from base +X to the camera-to-object vector:

```text
camera->object = r [cos(a), 0, sin(a)]
camera position = object center - camera->object
```

The default upper semicircle runs from -180 to 0 degrees. The recorded
quaternion is the zero-pitch orientation at -90 degrees. Every other target is
computed as `q_target = q_base_Y(delta_pitch) * q_reference`. Pre-multiplication
makes the physical rotation occur around robot base Y; directly replacing an
Euler component can appear as roll near this tool orientation. Use
`--reference-angle-deg` to change the zero-pitch point and `--pitch-offset-deg`
for a fixed mount calibration offset.

Angles 0 to 180 put the camera on the negative-Z side of the object. The
defaults use -180 to 0, placing the camera on the positive-Z/upper side.

## Use

Install the camera/robot dependencies in the active Python environment:

```powershell
pip install numpy opencv-python pyrealsense2 ur-rtde
```

First generate a plan. The script connects to the robot, asks you to jog the
TCP reference point to the object center, records that pose, and does not move:

```powershell
python robot_d405_capture/capture_semicircles.py `
  --robot-ip 192.168.1.101 `
  --camera-serial 409122271039
```

Inspect `capture_config.json` and `planned_poses.jsonl` in the printed output
folder. Then run the actual capture into a new output folder:

```powershell
python robot_d405_capture/capture_semicircles.py `
  --robot-ip 192.168.1.101 `
  --camera-serial 409122271039 `
  --jig-config pose_gt/objects/matka/jig1_config.json `
  --output robot_d405_capture/captures/run_001 `
  --execute
```

Every `--execute` run requires its matching `--jig-config`. After all planned
frames are captured and the camera and robot are safely stopped, the script
automatically detects AprilTags and writes `pose_gt/NNNNNN.json`. RGB/depth
pairs with no detected tag are moved under that capture's `archive/`.

To reuse a previously recorded object center without jogging again, pass it
directly to both planning and capture runs:

```powershell
python robot_d405_capture/capture_semicircles.py `
  --center-file robot_d405_capture/common_center.json
```

Actual motion requires typing `MOVE`. Keep the teach pendant and emergency
stop accessible. Test a small, safe subset before the complete run, for example:

```powershell
python robot_d405_capture/capture_semicircles.py `
  --robot-ip 192.168.1.101 --camera-serial 409122271039 `
  --min-radius-cm 30 --max-radius-cm 31 `
  --angle-start-deg 80 --angle-end-deg 100 --angle-step-deg 5 `
  --output robot_d405_capture/captures/safety_test --execute
```

Each frame produces lossless `color_rgb/NNNNNN.png` and raw uint16 NumPy
`depth_raw/NNNNNN.npy`. Load depth with `numpy.load(path, allow_pickle=False)`
and multiply its pixels by
`camera.depth_scale_m_per_unit` in `capture_config.json` to obtain metres.
`manifest.jsonl` records the requested and measured TCP pose for every frame.
