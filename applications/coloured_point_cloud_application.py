"""OOP controller for the Stage 5 coloured point-cloud test."""

from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from camera import OpenNIError, RGBCameraError, RGBDCamera
from processing import ColouredPointCloudGenerator, DepthIntrinsics
from reporting import EvidenceManager


class ColouredPointCloudApplication:
    """Capture one registered RGB-D pair and produce an XYZRGB PLY file."""

    def __init__(
        self,
        output_path: Path,
        minimum_depth_mm: int = 400,
        maximum_depth_mm: int = 8000,
        pixel_stride: int = 2,
        experiment_name: str = "8m_stride2",
    ) -> None:
        self.camera = RGBDCamera(rgb_camera_index=0)
        self.output_path = output_path
        self.evidence = EvidenceManager()
        self.minimum_depth_mm = minimum_depth_mm
        self.maximum_depth_mm = maximum_depth_mm
        self.pixel_stride = pixel_stride
        self.experiment_name = experiment_name

    def run(self) -> int:
        try:
            print("Stage 5: starting registered RGB-D capture...", flush=True)
            acquisition_started = time.perf_counter()
            with self.camera:
                # Warm up both interfaces so exposure and depth values settle.
                for _ in range(30):
                    depth_mm, colour_bgr = self.camera.read()

                print(
                    f"Captured depth {depth_mm.shape} and RGB "
                    f"{colour_bgr.shape[:2]}.",
                    flush=True,
                )

                # The UVC frame should be 640x480, but verify before assigning
                # RGB by pixel coordinate. Resizing here would hide bad data.
                if colour_bgr.shape[:2] != depth_mm.shape:
                    raise ValueError(
                        f"Depth/RGB size mismatch: {depth_mm.shape} and "
                        f"{colour_bgr.shape[:2]}"
                    )

                horizontal_fov, vertical_fov = (
                    self.camera.depth_camera.get_field_of_view()
                )
                rgb_path = self.evidence.save_cv_image(
                    "coloured_point_cloud",
                    f"source_registered_rgb_{self.experiment_name}",
                    colour_bgr,
                )

                # Keep independent NumPy-owned copies. No Open3D object is
                # created while OpenNI and OpenCV still own native resources.
                depth_snapshot = depth_mm.copy()
                colour_snapshot = colour_bgr.copy()

            acquisition_seconds = time.perf_counter() - acquisition_started
            print("Camera interfaces closed safely.", flush=True)

            # Create Open3D data only after camera-driver cleanup. Keeping the
            # native libraries out of each other's cleanup phase avoids the
            # Windows 0xC0000374 heap-corruption crash seen in the first test.
            height, width = depth_snapshot.shape
            intrinsics = DepthIntrinsics.from_field_of_view(
                width, height, horizontal_fov, vertical_fov
            )
            generator = ColouredPointCloudGenerator(
                intrinsics,
                minimum_depth_mm=self.minimum_depth_mm,
                maximum_depth_mm=self.maximum_depth_mm,
                pixel_stride=self.pixel_stride,
            )
            statistics = generator.calculate_statistics(depth_snapshot)
            statistics_report = self._format_statistics(statistics)
            print(statistics_report, flush=True)
            statistics_path = self.evidence.save_text(
                "coloured_point_cloud",
                f"depth_validity_statistics_{self.experiment_name}",
                statistics_report,
            )
            conversion_started = time.perf_counter()
            point_cloud = generator.generate_coloured(
                depth_snapshot, colour_snapshot
            )
            conversion_seconds = time.perf_counter() - conversion_started
            print("XYZRGB conversion completed.", flush=True)

            # Import only after OpenNI/OpenCV camera resources are closed.
            import open3d as o3d

            point_count = len(point_cloud.points)
            if point_count == 0:
                print("Coloured point-cloud generation failed: no valid points.")
                return 1

            # Convert OpenNI/Open3D camera coordinates (X-right, Y-down,
            # Z-forward) to the same conventional Z-up frame used by the
            # room reconstruction stages.
            point_cloud.transform(
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            )

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            ply_save_started = time.perf_counter()
            if not o3d.io.write_point_cloud(str(self.output_path), point_cloud):
                print(f"Could not save point cloud to {self.output_path}")
                return 1
            ply_save_seconds = time.perf_counter() - ply_save_started
            ply_size_mb = self.output_path.stat().st_size / (1024.0 * 1024.0)

            performance_report = self._format_performance(
                point_count,
                acquisition_seconds,
                conversion_seconds,
                ply_save_seconds,
                ply_size_mb,
            )
            performance_path = self.evidence.save_text(
                "coloured_point_cloud",
                f"performance_{self.experiment_name}",
                performance_report,
            )

            print(f"Generated {point_count:,} XYZRGB points.", flush=True)
            print(performance_report, flush=True)
            print(
                f"Depth statistics evidence saved: {statistics_path}",
                flush=True,
            )
            print(f"Source RGB evidence saved: {rgb_path}", flush=True)
            print(
                f"Saved coloured cloud: {self.output_path.resolve()}", flush=True
            )
            print(
                f"Performance evidence saved: {performance_path}", flush=True
            )
            self._open_external_viewer()
            return 0

        except (OpenNIError, RGBCameraError, RuntimeError, ValueError) as error:
            print(f"Coloured point-cloud test failed: {error}")
            return 1
        finally:
            cv2.destroyAllWindows()

    def _format_statistics(self, statistics) -> str:
        """Create terminal/report text for the current depth filter."""
        return "\n".join(
            [
                "Stage 5 depth validity statistics",
                "=================================",
                "Configuration: "
                f"{self.minimum_depth_mm}-{self.maximum_depth_mm} mm, "
                f"pixel stride {self.pixel_stride}",
                f"Total depth pixels:       {statistics.total_pixels:,}",
                f"Zero/invalid depth:       {statistics.zero_depth_pixels:,}",
                f"Below {self.minimum_depth_mm} mm:"
                f"{statistics.below_minimum_pixels:14,}",
                f"Above {self.maximum_depth_mm} mm:"
                f"{statistics.above_maximum_pixels:14,}",
                f"Valid full-res pixels:    {statistics.valid_pixels:,}",
                f"Valid full-res percentage:{statistics.valid_percentage:8.2f}%",
            ]
        )

    def _format_performance(
        self,
        point_count: int,
        acquisition_seconds: float,
        conversion_seconds: float,
        ply_save_seconds: float,
        ply_size_mb: float,
    ) -> str:
        """Create repeatable timing and file-size evidence."""
        return "\n".join(
            [
                "Stage 5 point-cloud performance",
                "===============================",
                f"Experiment:             {self.experiment_name}",
                f"Generated points:       {point_count:,}",
                f"RGB-D acquisition:      {acquisition_seconds:.3f} s",
                f"XYZRGB conversion:      {conversion_seconds:.3f} s",
                f"PLY save:               {ply_save_seconds:.3f} s",
                f"PLY file size:          {ply_size_mb:.3f} MiB",
            ]
        )

    def _open_external_viewer(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        screenshot_path = self.evidence.new_path(
            "coloured_point_cloud",
            f"xyzrgb_view_{self.experiment_name}",
            "png",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "view_point_cloud.py"),
                str(self.output_path.resolve()),
                "--screenshot",
                str(screenshot_path),
                "--title",
                "Astra Pro - coloured XYZRGB point cloud "
                f"({self.experiment_name})",
            ],
            check=False,
        )
        if screenshot_path.is_file():
            print(f"Point-cloud evidence saved: {screenshot_path}")
        if result.returncode != 0:
            print(
                "Open3D viewer closed with native Windows code "
                f"{result.returncode}. The saved PLY is still valid."
            )
