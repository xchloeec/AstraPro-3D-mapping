# Astra Pro RGB-D 3D Mapping

An experimental Python/Open3D pipeline for recording, reconstructing and
inspecting indoor environments with an original Orbbec Astra Pro. Numbered
stages allow camera access, synchronisation, odometry, mapping and measurement
to be tested independently.

> **Accuracy notice:** local RGB-D geometry is usable, but complete-room maps
> currently accumulate camera-pose drift. Stage 17 measurements are distances
> inside the reconstructed map and are **not yet survey/construction-grade**.

## Hardware architecture

The original Astra Pro exposes depth through OpenNI2 and colour through a
separate Windows UVC interface. It has no hardware RGB-depth synchronisation or
integrated IMU. `camera/rgbd_adapter.py` combines both interfaces and pairs
frames by nearest software timestamp. RTAB-Map's native OpenNI2 RGB-D input
cannot directly handle this split-interface configuration.

## Requirements

- Windows 10/11, 64-bit Python 3.11 and an Orbbec Astra Pro
- OpenNI2 runtime with the Orbbec driver
- NumPy, OpenCV and Open3D from `requirements.txt`

```powershell
python -m pip install -r requirements.txt
```

Close Windows Camera, NiViewer, OrbbecViewer and RTAB-Map before capture: only
one application may own the camera.

## Pipeline

```text
OpenNI2 depth + UVC RGB
  -> nearest-timestamp pairing
  -> quality-controlled RGB-D dataset
  -> camera odometry / experimental pose graph
  -> XYZRGB fusion -> PLY -> viewing / measurement
```

## Stage guide

| Stage | Entry point | Purpose | Status |
|---|---|---|---|
| 1 | `01_test_depth_camera.py` | Validate OpenNI2 depth capture | Stable |
| 2 | `02_test_rgb_camera.py` | Locate and test UVC RGB | Stable |
| 3 | `03_test_point_cloud.py` | Generate one depth-only XYZ cloud | Stable |
| 4 | `04_test_rgbd_alignment.py` | Test registered RGB/depth acquisition | Stable |
| 5/5A | `05_test_coloured_point_cloud.py`, `05a_test_dense_coloured_point_cloud.py` | Coloured single-frame cloud | Stable; worker retry may be required |
| 6 | `06_test_point_cloud_filter.py` | Downsample and filter a cloud | Stable |
| 7/7B | `07_test_multiframe_capture.py`, `07b_test_multiframe_point_clouds.py` | Capture and inspect multiple frames | Experimental |
| 8 | `08_export_open3d_dataset.py` | Export an Open3D dataset | Experimental |
| 9 | `09_record_open3d_room_dataset.py` | Legacy room recording/reconstruction | Legacy and slow |
| 10 | `10_live_rgbd_mapping.py` | Initial live mapping | Experimental; drift remains |
| 11 | `11_launch_rtabmap_room_scanner.py` | RTAB-Map investigation | Limited by split RGB/depth interfaces |
| 12/12B | `12_test_rgbd_adapter.py`, `12b_test_rgbd_pairing.py` | Standard frames and timestamp pairing | Stable |
| 13 | `13_record_rgbd_slam_dataset.py` | Quality-controlled timestamped dataset | Stable |
| 14/14B | `14_test_offline_rgbd_odometry.py` | Sequential 6DoF trajectory | Locally stable; cumulative drift remains |
| 15 | `15_fuse_rgbd_room.py` | Fast trajectory-guided XYZRGB PLY | Stable processing; pose-dependent accuracy |
| 16 | `16_optimize_pose_graph.py` | Visual/depth pose-graph experiment | Experimental; validate before use |
| 17 | `17_live_scan_and_measure.py` | One-run live map, PLY and measurement | Experimental; not measurement-grade |

## Recommended offline workflow

```powershell
python 13_record_rgbd_slam_dataset.py
python 14_test_offline_rgbd_odometry.py
python 15_fuse_rgbd_room.py
```

During capture, move slowly, retain 60-80% overlap and revisit a textured
starting area. Stage 15 automatically selects the latest suitable dataset. It
uses Stage 14 by default; experimental Stage 16 poses require `--use-stage16`.

## One-run live workflow

```powershell
python 17_live_scan_and_measure.py
```

Close the live window to stop and save `live_map.ply`. In the measurement
window, Shift+left-click two points and press Q. XYZ coordinates and distance
are printed and written to `point_measurements.csv`.

## Core mathematics

### Depth back-projection

For depth pixel `(u,v)`:

```text
Z = depth_mm / 1000
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Stage 13 stores depth plus `fx, fy, cx, cy` instead of duplicating each frame as
XYZ. Final PLY files already contain XYZRGB.

### World transformation and coordinate frame

```text
P_world = T_world_from_camera * P_camera
```

OpenNI/Open3D camera coordinates (X-right, Y-down, Z-forward) are exported as
X-right, Y-forward, Z-up:

```text
X' = X, Y' = Z, Z' = -Y
```

### Point distance

```text
distance = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
```

This is metric distance inside the reconstructed map; its real-world accuracy
is limited by depth noise, point selection and global pose drift.

## Verified results

- median RGB-depth offset improved from 57.29 ms to about 8-9 ms;
- Stage 14B improved one test from 66/99 to 98/99 tracked steps;
- Stage 15 projected about 13 million points from 200 frames and saved a PLY
  in approximately 10 seconds;
- Stage 17 measured 2.027 m against about 1.80 m ground truth (12.6% error), so
  the current drifted whole-room map is unsuitable for accurate dimensions.

## Known limitations

- Native camera libraries may crash with Windows `0xC0000374`; capture is
  isolated in restartable worker processes.
- Per-frame tracking success does not guarantee globally correct geometry.
- White, reflective and textureless surfaces weaken visual odometry.
- Stage 16 may find local revisits without a valid complete-room closure.
- Filtering depth cannot repair an incorrect camera trajectory.
- Accurate wall spacing should use region/plane fitting, not two noisy points.

## Generated data and Git

`.gitignore` excludes `output/`, `report_evidence/`, `.venv/` and `external/`.
These may contain large recordings, PLY files, evidence and third-party
binaries. The repository stores source code and documentation only.

## Future work

- multi-frame median depth and plane-to-plane measurement;
- stronger loop closure, relocalisation and a live tracking-quality overlay;
- RGB-D + IMU + wheel-odometry fusion for the rescue rover;
- ROS 2 and Raspberry Pi 5 integration;
- quantitative validation against measured room dimensions.
