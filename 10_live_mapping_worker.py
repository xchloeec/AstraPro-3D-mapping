"""Build and display a coloured point-cloud map while Stage 10 records.

This worker deliberately runs in a different Python process from OpenNI and
OpenCV camera capture.  The camera writes complete RGB/depth pairs to disk;
this process consumes them with Open3D, estimates motion and updates the map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from processing.surface_coverage import SurfaceCoverage
from processing.validated_registration import make_frame, matches_between, register_pair


class LiveRGBDMapper:
    """Estimate camera motion and accumulate registered RGB-D point clouds."""

    def __init__(
        self,
        dataset_path: Path,
        frame_stride: int = 3,
        voxel_size_m: float = 0.04,
    ) -> None:
        self.dataset_path = dataset_path.resolve()
        self.colour_path = self.dataset_path / "color"
        self.depth_path = self.dataset_path / "depth"
        self.intrinsic_path = self.dataset_path / "intrinsic.json"
        self.finished_flag = self.dataset_path / "_capture_finished.flag"
        self.stop_flag = self.dataset_path / "_stop_requested.flag"
        self.output_path = self.dataset_path / "live_map.ply"
        self.trajectory_path = self.dataset_path / "live_trajectory.txt"
        self.frame_stride = frame_stride
        self.voxel_size_m = voxel_size_m
        self.depth_truncation_m = 5.0
        self.world_from_camera = np.eye(4)
        self.map_cloud = o3d.geometry.PointCloud()
        self.trajectory_rows: list[str] = []
        self.coverage = SurfaceCoverage()
        self.coverage_cloud = o3d.geometry.PointCloud()
        self.show_coverage = False
        self.geometry_added = False

    def run(self) -> int:
        intrinsic = self._wait_for_intrinsic()
        visualizer = self._create_visualizer()
        geometry_added = False
        previous_rgbd: o3d.geometry.RGBDImage | None = None
        previous_frame = None
        reference_pose = self.world_from_camera.copy()
        reference_processed_count = 0
        last_valid_frame = None
        last_valid_pose = None
        next_frame = 0
        processed_frames = 0
        lost_frames = 0
        anchors = []
        last_tracked_index = None
        processing_error = False

        print("3D controls: C coverage/colour; Space pause/resume capture. Red=rescan, amber=add view, green=more observed views.", flush=True)
        print("Coverage is based on processed frames; the mapper may lag capture.", flush=True)
        try:
            while True:
                if not visualizer.poll_events():
                    self.stop_flag.write_text("viewer closed", encoding="utf-8")
                    print("3D viewer closed; requesting capture stop.", flush=True)
                    break

                pair = self._find_next_pair(next_frame)
                if pair is None:
                    if self.finished_flag.exists():
                        break
                    time.sleep(0.02)
                    visualizer.update_renderer()
                    continue

                frame_index, colour_file, depth_file = pair
                next_frame = frame_index + self.frame_stride
                current_rgbd = self._load_rgbd(colour_file, depth_file)
                current_frame = make_frame(current_rgbd, intrinsic)
                relocalised = False
                recent_reference = False

                if previous_rgbd is None:
                    tracking_succeeded = True
                else:
                    tracking_succeeded, source_to_target, _, details = register_pair(previous_frame, current_frame)
                    if not tracking_succeeded and reference_processed_count != processed_frames:
                        advanced, measured, _, recent_details = register_pair(last_valid_frame,current_frame)
                        if advanced:
                            tracking_succeeded = recent_reference = True
                            source_to_target = measured
                            details = {**recent_details,'reason':'validated_recent_reference'}
                    if not tracking_succeeded:
                        ranked = sorted(((len(matches_between(frame,current_frame)),pose,frame)
                                         for pose,frame in anchors),key=lambda x:x[0],reverse=True)
                        for score, anchor_pose, anchor_frame in ranked[:3]:
                            if score < 25:
                                break
                            recovered, measured, _, recovery_details = register_pair(anchor_frame,current_frame,loop=True)
                            if recovered:
                                recovered_pose = anchor_pose @ np.linalg.inv(measured)
                                source_to_target = np.linalg.inv(recovered_pose) @ self.world_from_camera
                                tracking_succeeded = relocalised = True
                                details = {**recovery_details,"reason":"relocalised_validated_view"}
                                break
                    with (self.dataset_path / "tracking_quality.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps({"frame": frame_index, **details}) + "\n")
                    if tracking_succeeded and not relocalised:
                        translation_m = float(
                            np.linalg.norm(source_to_target[:3, 3])
                        )
                        cosine_angle = float(
                            np.clip(
                                (np.trace(source_to_target[:3, :3]) - 1.0) / 2.0,
                                -1.0,
                                1.0,
                            )
                        )
                        rotation_degrees = float(
                            np.degrees(np.arccos(cosine_angle))
                        )
                        # One processed step spans about 0.3 s. A handheld
                        # scan cannot move 0.50 m or turn 35 degrees in that
                        # interval while retaining useful RGB-D overlap.
                        plausible_motion = (
                            translation_m <= 0.30
                            and rotation_degrees <= 25.0
                        )
                        if not plausible_motion:
                            tracking_succeeded = False
                            print(
                                "Rejected implausible odometry at frame "
                                f"{frame_index}: {translation_m:.3f} m, "
                                f"{rotation_degrees:.1f} degrees.",
                                flush=True,
                            )
                    if tracking_succeeded:
                        # Open3D returns source-camera -> target-camera.  The
                        # inverse places the new camera into our world frame.
                        self.world_from_camera = (recovered_pose if relocalised else
                                                  (last_valid_pose if recent_reference else reference_pose)
                                                  @ np.linalg.inv(source_to_target))

                if not tracking_succeeded:
                    lost_frames += 1
                    self._write_tracking_status(frame_index, last_tracked_index, False)
                    print(
                        f"TRACKING LOST at frame {frame_index}; move back to "
                        "the last view and continue more slowly.",
                        flush=True,
                    )
                    # Retain BOTH the last validated image and its pose.
                    # Rebasing only the image silently attaches later frames
                    # to an unrelated old pose and corrupts the room geometry.
                    # Recovery now requires returning to the last tracked view.
                    continue

                frame_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                    current_frame.rgbd,
                    current_frame.intrinsic,
                    project_valid_depth_only=True,
                )
                frame_cloud.transform(self.world_from_camera)
                self.coverage.update(
                    frame_cloud.voxel_down_sample(0.10).points,
                    self.world_from_camera[:3, 3],
                )
                frame_cloud = frame_cloud.voxel_down_sample(self.voxel_size_m)
                self.map_cloud += frame_cloud
                processed_frames += 1
                if (previous_frame is None or relocalised or recent_reference
                        or np.linalg.norm(source_to_target[:3,3]) >= .08
                        or np.degrees(np.arccos(np.clip((np.trace(source_to_target[:3,:3])-1)/2,-1,1))) >= 5
                        or processed_frames-reference_processed_count >= 5):
                    previous_rgbd = current_rgbd
                    previous_frame = current_frame
                    reference_pose = self.world_from_camera.copy()
                    reference_processed_count = processed_frames
                last_valid_frame = current_frame
                last_valid_pose = self.world_from_camera.copy()
                last_tracked_index = frame_index
                if processed_frames == 1 or processed_frames % 10 == 0:
                    anchors.append((self.world_from_camera.copy(),current_frame))
                    # Bound live relocalisation memory for unlimited scans.
                    if len(anchors) > 100:
                        del anchors[1]
                self._write_tracking_status(frame_index,last_tracked_index,True)
                self._record_pose(frame_index)

                # Bound memory and rendering cost as the map grows.
                if processed_frames % 8 == 0:
                    compact = self.map_cloud.voxel_down_sample(self.voxel_size_m)
                    # Keep the same Python geometry object registered with the
                    # Visualizer; replace only its data arrays.
                    self.map_cloud.points = compact.points
                    self.map_cloud.colors = compact.colors

                if not geometry_added:
                    visualizer.add_geometry(self.map_cloud)
                    # Raw camera-world coordinates are X-right, Y-down,
                    # Z-forward. Set the view explicitly rather than using
                    # Open3D's default +Y-up orientation.
                    control = visualizer.get_view_control()
                    control.set_lookat(self.map_cloud.get_center())
                    control.set_front([0.0, 0.0, -1.0])
                    control.set_up([0.0, -1.0, 0.0])
                    control.set_zoom(0.55)
                    geometry_added = True
                    self.geometry_added = True
                else:
                    visualizer.update_geometry(self.map_cloud)
                if self.show_coverage:
                    self._refresh_coverage(visualizer)
                visualizer.update_renderer()
                print(
                    f"Live map: source frame {frame_index:03d}, "
                    f"{len(self.map_cloud.points):,} points, "
                    f"tracking losses {lost_frames}",
                    flush=True,
                )
        except Exception as error:
            processing_error = True
            print(f"Live processing failed: {error}; saving validated partial geometry.",flush=True)
            try:
                self.stop_flag.write_text("mapper processing failed",encoding="utf-8")
            except OSError:
                pass
        finally:
            visualizer.destroy_window()

        if not self.map_cloud.has_points():
            print("Live mapper ended without any 3D points.", flush=True)
            return 1

        self.map_cloud = self.map_cloud.voxel_down_sample(self.voxel_size_m)
        if not o3d.io.write_point_cloud(str(self.output_path), self.map_cloud):
            print(f"Could not save {self.output_path}", flush=True)
            return 1
        self.trajectory_path.write_text(
            "\n".join(self.trajectory_rows), encoding="utf-8"
        )
        o3d.io.write_point_cloud(str(self.dataset_path / "coverage_camera_world.ply"), self.coverage.cloud())
        (self.dataset_path / "coverage_summary.json").write_text(
            json.dumps(self.coverage.summary(), indent=2), encoding="utf-8"
        )
        (self.dataset_path / "live_mapping_summary.json").write_text(json.dumps({
            "processed_keyframes":processed_frames,"tracking_losses":lost_frames,
            "processing_error":processing_error,"saved_partial_on_error":processing_error,
        },indent=2),encoding="utf-8")
        print("\nLIVE 3D PARTIAL MAP SAVED" if processing_error else "\nLIVE 3D MAP COMPLETED", flush=True)
        print(f"Processed keyframes: {processed_frames}", flush=True)
        print(f"Tracking losses:     {lost_frames}", flush=True)
        print(f"Saved PLY: {self.output_path}", flush=True)
        return 1 if processing_error else 0

    def _wait_for_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        while not self.intrinsic_path.is_file():
            if self.finished_flag.exists():
                raise RuntimeError("Capture ended before intrinsic.json was created")
            time.sleep(0.05)
        return o3d.io.read_pinhole_camera_intrinsic(str(self.intrinsic_path))

    def _create_visualizer(self) -> o3d.visualization.Visualizer:
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        visualizer.create_window(
            window_name="Stage 10 - Live RGB-D room mapping",
            width=1100,
            height=760,
        )
        options = visualizer.get_render_option()
        options.background_color = np.asarray([0.04, 0.05, 0.06])
        options.point_size = 2.0
        visualizer.register_key_callback(ord("C"), self._toggle_coverage)
        visualizer.register_key_callback(ord(" "), self._toggle_pause)
        return visualizer

    def _refresh_coverage(self, visualizer):
        cloud = self.coverage.cloud()
        self.coverage_cloud.points = cloud.points
        self.coverage_cloud.colors = cloud.colors
        visualizer.update_geometry(self.coverage_cloud)

    def _toggle_coverage(self, visualizer):
        if not self.geometry_added:
            return False
        self.show_coverage = not self.show_coverage
        if self.show_coverage:
            cloud = self.coverage.cloud()
            self.coverage_cloud.points = cloud.points
            self.coverage_cloud.colors = cloud.colors
            visualizer.remove_geometry(self.map_cloud, reset_bounding_box=False)
            visualizer.add_geometry(self.coverage_cloud, reset_bounding_box=False)
        else:
            visualizer.remove_geometry(self.coverage_cloud, reset_bounding_box=False)
            visualizer.add_geometry(self.map_cloud, reset_bounding_box=False)
        print("Coverage: red=rescan; amber=add view; green=3+ viewpoint bins. Unseen areas remain unknown." if self.show_coverage else "Colour map", flush=True)
        return True

    def _toggle_pause(self, visualizer):
        flag = self.dataset_path / "_paused.flag"
        if flag.exists():
            flag.unlink()
            print("Capture resumed; return to the last tracked view first.", flush=True)
        else:
            flag.write_text("paused from 3D review", encoding="utf-8")
            print("Capture paused. Press C to review weak areas; Space resumes.", flush=True)
        return False

    def _find_next_pair(self, requested_index: int) -> tuple[int, Path, Path] | None:
        # Normally the exact stride frame is available.  If capture has ended,
        # accept the next available complete pair so a resumed session is safe.
        colour_file = self.colour_path / f"{requested_index:05d}.jpg"
        depth_file = self.depth_path / f"{requested_index:05d}.png"
        if colour_file.is_file() and depth_file.is_file():
            return requested_index, colour_file, depth_file
        return None

    def _load_rgbd(self, colour_file: Path, depth_file: Path) -> o3d.geometry.RGBDImage:
        colour = o3d.io.read_image(str(colour_file))
        depth = o3d.io.read_image(str(depth_file))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            colour,
            depth,
            depth_scale=1000.0,
            depth_trunc=self.depth_truncation_m,
            convert_rgb_to_intensity=False,
        )

    def _record_pose(self, frame_index: int) -> None:
        flat_pose = " ".join(f"{value:.9f}" for value in self.world_from_camera.ravel())
        row = f"{frame_index} {flat_pose}"
        # Checkpoint each validated pose, not just at the end of the viewer.
        with self.trajectory_path.open("a",encoding="utf-8") as file:
            file.write(row + "\n")
        self.trajectory_rows.append(row)

    def _write_tracking_status(self, processed_index, reference_index, tracked):
        temporary = self.dataset_path / "_tracking_status.tmp"
        # Windows readers may open the target without FILE_SHARE_DELETE.
        # This UI telemetry is optional: a transient sharing violation must
        # never discard a map or end camera capture.
        try:
            temporary.write_text(json.dumps({"processed_frame":processed_index,
                                 "last_tracked_frame":reference_index,"tracked":tracked}),encoding="utf-8")
            for attempt in range(3):
                try:
                    temporary.replace(self.dataset_path / "tracking_status.json")
                    return True
                except PermissionError:
                    if attempt < 2:
                        time.sleep(.005)
            return False
        except OSError:
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Open3D RGB-D mapper")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--voxel", type=float, default=0.04)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    raise SystemExit(
        LiveRGBDMapper(
            arguments.dataset,
            frame_stride=arguments.stride,
            voxel_size_m=arguments.voxel,
        ).run()
    )
