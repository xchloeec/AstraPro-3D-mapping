# Astra Pro RGB-D 3D Mapping

An experimental Python/Open3D pipeline for recording, reconstructing and
inspecting indoor environments with an original Orbbec Astra Pro. Numbered
stages allow camera access, synchronisation, odometry, mapping and measurement
to be tested independently.

> **Accuracy notice:** local RGB-D geometry is usable, but complete-room maps
> currently accumulate camera-pose drift. Stage 17 measurements are distances
> inside the reconstructed map and are **not yet survey/construction-grade**.

## Current workflow (8 October 2026)

Run `17_live_scan_and_measure.py` for a new scan, including a different room.
Space starts/pauses/resumes; Q finishes and triggers final processing. No fixed
frame limit or command-line arguments are required. Keep overlapping translated
views, capture both sides of corners and repeatedly revisit textured areas,
including the starting area at the end. When paused, return to the last tracked
view before resuming. Each scan creates separate recording and evidence folders.

The latest calibrated offline result can be opened without cameras or new
processing using `28_review_calibrated_room.py`. C toggles coverage and L toggles
interpreted layout. Calibration remains applicable to another room when the
camera, resolution and optical configuration stay the same. Stages 24, 26 and
27 support checkerboard RGB calibration and experimental depth-projection
validation; they do not automatically install a new live calibration.

The current room reconstruction still has missing floor regions and fragmented
geometry. Long-revisit diagnostics found serious trajectory inconsistency in
the existing recording. An independent correction failed its held-out 10 cm
check, so it was retained as diagnostic evidence rather than a replacement map.
Revisit retrieval now reserves distant candidates and retains distributed
historic views; this change still needs a successful full-room validation.
Lighting normalisation helps tracking but cannot recover missing depth from
reflective surfaces. Layout extraction preserves observed recesses/partitions;
it does not force the room into a rectangle or certify traversable free space.

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
| 18 | `18_finalize_room_scan.py` | Validated trajectory pass, surface mesh and coverage review | Automatic after Stage 17; inspect refinement report |
| 19 | `19_refine_room_trajectory.py` | Recompute poses, relocalise and verify revisit constraints | Offline diagnostics; accuracy still requires measured validation |
| 20 | `20_extract_room_layout.py` | Observed wall candidates, fitted surfaces and floor plan | Automatic in Stage 18; boundary completeness remains unverified |
| 22 | `22_compare_lingbot.py` | Optional isolated RGB-only model comparison | Experimental; separate dependencies and weights |
| 23 | `23_audit_recording.py` | Audit existing capture quality | Diagnostic |
| 24 | `24_calibrate_rgb_camera.py` | Checkerboard RGB calibration | Default 17.1 mm matches the measured local print; measure your own paper |
| 25 | `25_check_saved_rgbd_alignment.py` | Saved RGB/depth edge comparison | Diagnostic, not an extrinsic calibration |
| 26 | `26_record_alignment_board.py` | Record paired checkerboard/depth views | RGB isolated from OpenNI; valid-depth capture guard |
| 27 | `27_validate_alignment_board.py` | Compare experimental depth projections | Does not auto-apply calibration |
| 28 | `28_review_calibrated_room.py` | Open latest calibrated model | Read-only review, no camera required |
| 29 | `29_audit_room_drift.py` | Saved graph/revisit coverage audit | Diagnostic |
| 30/31 | `30_find_long_room_revisits.py`, `31_correct_room_revisit_experiment.py` | Long-revisit correction experiment | Specific saved recording; exports only after validation passes |

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

Stage 17 has no fixed frame limit and starts paused. In the camera preview,
press **Space** to start, pause or resume, and **Q/Esc** to finish capture.
The mapper processes remaining frames and saves `live_map.ply`, then opens
measurement. Keep the camera at the paused view, or return to that view before
resuming: automatic relocalisation after moving while paused is not supported.
The 3D view may continue processing queued frames while capture is paused.
In the live 3D window, **C** toggles the colour map and observed-surface
coverage, and **Space** pauses/resumes capture. Red cells were observed from
one camera-position bin, amber from two, and green from at least three.
Pause to review, then return to the last tracked view and resume to revisit
red/amber surfaces with overlap from slightly shifted camera positions.
Coverage uses 10 cm surface cells and 15 cm camera-position bins; repeated
frames from a stationary camera do not increase the score. It is a heuristic,
not an accuracy score or whole-room completion percentage. Unseen surfaces
are unknown, and pose errors can also corrupt coverage estimates.
Long scans use additional disk space and still accumulate pose drift.
The live view uses camera-world -Y as up; the final measurement view and
evidence PLY use +Z as up. Start with the camera upright and approximately
level, since this camera has no IMU to determine gravity automatically.
Closing the live 3D window also stops capture and saves the processed map.
In the measurement
window, Shift+left-click two points and press Q. XYZ coordinates and distance
are printed and written to `point_measurements.csv`.
Stage 17 also saves a separate final-map evidence archive for each scan in
`report_evidence/stage_17_room_scans/`, including the PLY, available capture
statistics, trajectory, calibration, measurements and a provenance manifest.
Raw images remain in the original `output/live_mappings/` recording folder.
After capture, Stage 17 automatically runs Stage 18 and opens a final room
surface review. Press **C** to toggle surface/coverage and **L** for fitted
wall/floor layout, then close that window
to proceed to point measurement. Once capture has finished, this review is
read-only; use live pause/review/resume to add frames to the same scan.

## Final surface processing

Stage 17 runs this automatically. To reprocess a saved scan independently, run:

```powershell
python 18_finalize_room_scan.py --view
```

This processes the latest completed live scan into a TSDF surface mesh and
statistically filtered point cloud, archived separately under
`report_evidence/stage_18_final_room/`. An explicit recording-folder argument
selects an older scan. Originals remain in their recording folders.
Processing uses every saved tracked pose by default. `--max-frames` can reduce
processing cost, at the expense of coverage.
By default this first recomputes poses using depth-supported image features,
robust rigid estimation and checked ICP refinement. It attempts relocalisation
against stored keyframes and verifies revisit constraints before pose-graph
optimisation. Rejected graph proposals retain the recomputed sequential poses.
Frames without a validated pose are excluded rather than fused at an assumed
position. A low validated-frame fraction means the resulting model is partial.
These checks are heuristic; they do not guarantee accurate room dimensions or
fill unobserved areas. `--original-poses` bypasses refinement for comparison;
`--poses PATH` selects an explicitly saved trajectory.
The default final pass now tests every recorded frame (stride 1), rather than
every third frame. There is no fixed post-capture mapper wait timeout. Processing
can take considerably longer; wait for the final evidence path before closing
the process. Live preview still samples frames for responsiveness.

Final evidence also includes `layout_surfaces.ply`, `floorplan.svg` and
`layout_report.json`. Planes are fitted to observed surfaces, with optional
orthogonal regularisation only when wall directions and residuals support it.
Recesses and unobserved gaps are retained rather than replaced with a room box.
Tall furniture can resemble walls; fitted wall envelopes can span openings.
Floor cells represent observed floor support, not a verified closed footprint.
The layout view is an interpretation separate from the measured scan. No complete
or surveyed room layout is claimed until missing structure and dimensions are
independently checked. Plane-fit residuals flag candidates needing a rescan.
After live tracking loss, return to the last successfully tracked view; the
mapper now retains that reference instead of rebasing at an unknown pose.
Final archives also contain `coverage_review.ply` and `coverage_summary.json`.
They record the actual trajectory used, refinement diagnostics and accepted
revisit constraints. Original recordings and live trajectories are retained.
Stage 17 measures the final cleaned model when final processing succeeds;
`measurement_source.txt` identifies that model and CSV rows include both XYZs.

### Capture and tracking safeguards

New live scans use buffered nearest-software-timestamp RGB-D pairing, accept
offsets up to 25 ms and save `pair_timing.csv`. This is not hardware sync.
Tracking applies bounded adaptive gamma and local contrast normalisation to
grayscale RGB, preserving original recorded colours and depth. This software
exposure normalisation is not measured light or camera calibration and cannot
recover missing depth or detail in a black image. The camera preview warns of
low light and saves brightness, clipping and sharpness in `lighting_quality.csv`.
Intrinsics remain estimated from OpenNI field of view; no independently
measured spatial calibration is claimed (`capture_metadata.json`).
Features are detected in regions with valid depth. Tracking requires robust
feature/depth agreement and bidirectional geometric overlap, and retains the
last validated reference on failure. The camera preview displays tracking
status and the last tracked image to return to. Status can lag live capture.
If moving while paused, return to an overlapping tracked view before resuming;
relocalisation is attempted but may fail if no sufficiently similar view exists.
Live status updates tolerate Windows file-sharing locks: an unavailable update
is skipped instead of aborting reconstruction. Each accepted live pose is
checkpointed immediately. A handled processing exception exports the current
partial map and marks `live_mapping_summary.json` accordingly.
If a process terminates before map export, saved RGB-D pairs can be recovered
by passing that recording folder explicitly to `18_finalize_room_scan.py`.

Run diagnostics independently without a camera:

```powershell
python 19_refine_room_trajectory.py output/live_mappings/live_DATE_TIME
python 18_finalize_room_scan.py output/live_mappings/live_DATE_TIME --view
python -m unittest discover -s tests -p test_validated_registration.py
```

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

Current development scope is camera-only. IMU, wheel odometry and other
additional sensors are deferred to a later independent cross-check phase.
Camera priorities are measured RGB-depth spatial alignment, exposure quality,
validated tracking, depth consistency and evidence-backed layout extraction.

Camera pose tracking retains a reference keyframe for small movements instead
of accumulating every adjacent near-identity estimate. Each validated frame
still receives a pose. Offline references advance at 0.08 m, 5 degrees or 15
accepted frames; the live mapper uses the same motion thresholds and a maximum
of five processed frames. This is a drift-reduction strategy, not a guarantee
of correct room shape.
If a retained keyframe no longer matches, tracking first tries the most recent
validated frame under the same visual/depth checks before searching old
relocalisation anchors. A successful recent-frame match becomes the new
reference. Failed measurements are still excluded rather than assigned poses.

Trajectory refinement saves the full pose graph before and after optimisation,
the unverified proposal trajectory, and numerical graph consistency checks.
If correction is rejected, final review explicitly labels the reconstruction
as unverified with failed global correction. High local tracking success does
not imply a successful room map.

Registration and Stage 18 fusion now exclude poorly supported depth pixels and
sharp depth boundaries in 3x3 neighbourhoods. Raw recordings are preserved.
Stage 18 saves `depth_quality.jsonl` and aggregate screening statistics in
`processing_report.json`. This screening does not detect reflective materials,
fill missing surfaces, or prove depth accuracy. Thin structures can be omitted;
missing depth remains unknown and must not be interpreted as traversable space.

- multi-frame median depth and plane-to-plane measurement;
- stronger loop closure, relocalisation and a live tracking-quality overlay;
- later: RGB-D + IMU + wheel-odometry cross-checking for the rescue rover;
- ROS 2 and Raspberry Pi 5 integration;
- quantitative validation against measured room dimensions.

### Isolated RGB-only LingBot-Map comparison (Stage 22)

Use the separate `.venv-lingbot` environment with the official checkout in
`external/lingbot-map` and checkpoint `external/lingbot-map/weights/lingbot-map.pt`:

```powershell
.venv-lingbot\Scripts\python.exe 22_compare_lingbot.py output/live_mappings/live_20261006_183452 --stride 10
```

This samples the whole recording, predicts geometry from RGB only, and saves a
unique folder under `report_evidence/lingbot_comparison`. It preserves model
version, checkpoint hash, frame IDs, depth/confidence predictions, poses, PLY,
runtime and post-inference comparisons with recorded depth. Model units are
unverified; recorded-depth ratios do not establish metric accuracy. Stage 17
does not automatically run this experiment. On Windows it uses PyTorch SDPA
instead of FlashInfer. The 8GB laptop defaults to width 392 and eight reference
keyframes, with CPU frame storage and prediction offload. Encoder weights use
BF16; prediction heads and LayerNorm retain FP32. These differ from the
original demo configuration and may affect quality. Preview the saved cloud with the original environment:

```powershell
.venv\Scripts\python.exe preview_lingbot.py report_evidence/lingbot_comparison/<run-folder>
```

The preview is not gravity calibrated. Unobserved regions remain unknown;
neural geometry must be checked before rescue navigation use.

### Audit a saved recording (Stage 23)

```powershell
.venv\Scripts\python.exe 23_audit_recording.py output/live_mappings/live_20261006_183452
```

This compares adjacent and every-tenth-frame RGB feature matching at sampled
endpoints and summarises saved lighting, depth validity, pairing and calibration
metadata. Results are saved under `report_evidence/recording_audit`. It does not
open a camera, change recordings, certify calibration or measure room coverage.

### Printed-board RGB calibration (Stage 24)

Close other camera applications. Mount the printed A4 target flat (a flat table
is suitable), verify its scale with a ruler, then run:

```powershell
.venv\Scripts\python.exe 24_calibrate_rgb_camera.py
```

The default is 17.1 mm for the current measured print, so IDE Run needs no
parameters. For a different print, supply its measured size with `--square-mm`.
The nominal PDF square size is 18 mm. The target has
9x6 inner corners (10x7 squares). The RGB stream must be 640x480, matching scans.
Move the camera around the fixed board, varying distance, tilt in both axes
and image position. Keep all corners visible. Space saves a detected view;
near-duplicate views are rejected. Aim for 25-30 views. C computes after at
least 20 views; Q exits. The console reports reprojection RMS and basic view
diversity checks. Add views and press C again if the result requires review.
Results and raw calibration images are saved in
`report_evidence/camera_calibration/<session>/`. These are RGB intrinsics and
distortion estimates only. They are not applied automatically to registered
depth, old recordings or Stage 17/18. RGB/depth spatial alignment still needs
separate verification before using them in the mapping pipeline.

### Known-size RGB/depth projection check (Stage 26)

Run `26_record_alignment_board.py` directly in the original environment; no
parameters are needed. It selects the latest saved RGB calibration for camera
index 0, including that session's measured square size. Close other camera apps.
Use the same flat board, move the camera between viewpoints, hold still for
about a second and press Space to save a detected board and registered-depth
pair. The window shows RGB and depth side by side; black depth means missing.
Start around 60-90 cm away and check at least 43 of 54 corner neighbourhoods
have valid depth. Frames failing that check are rejected before saving.
Aim for 10-15 varied views at distances where the whole board is clearly
visible, with different tilts and image positions. Q finishes; C is not needed.
Raw paired images, corners, software timing and calibration provenance are
saved under `report_evidence/alignment_board`. The program only records evidence;
projection checking and any mapping integration are separate steps.

Stage 26 runs UVC RGB acquisition in its own process, separately from OpenNI
depth, and prefers DirectShow. This avoids the reproduced native heap crash
when both camera APIs start in the same process. The parent supervises the
capture worker and can retry the alternate backend. `--probe` checks 30 paired
reads and checkerboard processing without opening the capture UI. Startup
attempts and exit codes are retained under `report_evidence/camera_startup`.
RGB pairing uses the UVC worker's software read-completion timestamp rather
than pipe-receipt time; this still does not provide hardware synchronisation.

Stage 27 (`27_validate_alignment_board.py`) checks the latest paired-board
session against its measured square size. The comparison is experimental:
depth bias, alignment and paper dimensions can affect its metric residuals.
An entirely missing board depth is reported as failure, not accepted calibration.

Stages 18 and 19 accept an explicit `--intrinsic` for calibrated offline replay.
Trajectory refinement saves `intrinsic_used.json`; fusion checks that its
projection matches the selected trajectory. New projection with old poses is
rejected. Final evidence retains the projection actually used and the original
recording intrinsic separately. Raw recording parameters and Stage 17 defaults
are not silently overwritten, and no distortion remap or hole fill is applied.

Run `28_review_calibrated_room.py` directly to open the latest completed
calibrated replay. It does not start a camera or repeat reconstruction.
Drag to orbit, scroll to zoom, C toggles coverage, L toggles interpreted layout
and Q exits. Global-correction failures remain labelled unverified in the viewer.

Stage 29 audits saved graph residuals and temporal revisit coverage. Stages
30 and 31 are diagnostic experiments for the 6 October recording: they search
long revisits, then attempt an independent correction with held-out revisit
targets. They do not change raw recordings or replace existing evidence.
Stage 31 exports a new model only if the certain-edge and held-out checks pass;
otherwise it retains a rejected proposal and validation report. The 8 October
experiment failed the 10 cm held-out threshold (approximately 10-19 cm), so no
replacement room model was produced. Large pose changes alone are not evidence
of correctness. Future trajectory retrieval now reserves candidates for distant
revisits and retains distributed historic anchors rather than only recent views.
