"""Build and display a coloured point-cloud map while Stage 10 records.

This worker deliberately runs in a different Python process from OpenNI and
OpenCV camera capture.  The camera writes complete RGB/depth pairs to disk;
this process consumes them with Open3D, estimates motion and updates the map.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import open3d as o3d


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

    def run(self) -> int:
        intrinsic = self._wait_for_intrinsic()
        visualizer = self._create_visualizer()
        geometry_added = False
        previous_rgbd: o3d.geometry.RGBDImage | None = None
        next_frame = 0
        processed_frames = 0
        lost_frames = 0

        print("Stage 10 live mapper is waiting for RGB-D keyframes...", flush=True)
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

                if previous_rgbd is None:
                    tracking_succeeded = True
                else:
                    tracking_succeeded, source_to_target, _ = (
                        o3d.pipelines.odometry.compute_rgbd_odometry(
                            previous_rgbd,
                            current_rgbd,
                            intrinsic,
                            np.eye(4),
                            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                            o3d.pipelines.odometry.OdometryOption(
                                depth_diff_max=0.07
                            ),
                        )
                    )
                    if tracking_succeeded:
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
                        self.world_from_camera = (
                            self.world_from_camera
                            @ np.linalg.inv(source_to_target)
                        )

                if not tracking_succeeded:
                    lost_frames += 1
                    print(
                        f"TRACKING LOST at frame {frame_index}; move back to "
                        "the last view and continue more slowly.",
                        flush=True,
                    )
                    # Freeze the global pose once, but rebase the image
                    # reference.  Otherwise every later frame is compared to
                    # an increasingly old view and one loss becomes many.
                    previous_rgbd = current_rgbd
                    continue

                frame_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                    current_rgbd,
                    intrinsic,
                    project_valid_depth_only=True,
                )
                frame_cloud.transform(self.world_from_camera)
                frame_cloud = frame_cloud.voxel_down_sample(self.voxel_size_m)
                self.map_cloud += frame_cloud
                processed_frames += 1
                previous_rgbd = current_rgbd
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
                    geometry_added = True
                else:
                    visualizer.update_geometry(self.map_cloud)
                visualizer.update_renderer()
                print(
                    f"Live map: source frame {frame_index:03d}, "
                    f"{len(self.map_cloud.points):,} points, "
                    f"tracking losses {lost_frames}",
                    flush=True,
                )
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
        print("\nLIVE 3D MAP COMPLETED", flush=True)
        print(f"Processed keyframes: {processed_frames}", flush=True)
        print(f"Tracking losses:     {lost_frames}", flush=True)
        print(f"Saved PLY: {self.output_path}", flush=True)
        return 0

    def _wait_for_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        while not self.intrinsic_path.is_file():
            if self.finished_flag.exists():
                raise RuntimeError("Capture ended before intrinsic.json was created")
            time.sleep(0.05)
        return o3d.io.read_pinhole_camera_intrinsic(str(self.intrinsic_path))

    def _create_visualizer(self) -> o3d.visualization.Visualizer:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(
            window_name="Stage 10 - Live RGB-D room mapping",
            width=1100,
            height=760,
        )
        options = visualizer.get_render_option()
        options.background_color = np.asarray([0.04, 0.05, 0.06])
        options.point_size = 2.0
        return visualizer

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
        self.trajectory_rows.append(f"{frame_index} {flat_pose}")


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
