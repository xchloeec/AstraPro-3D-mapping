"""Stage 7B: convert one saved RGB-D sequence into separate point clouds."""

from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import open3d as o3d

from processing import (
    ColouredPointCloudGenerator,
    DepthIntrinsics,
    PointCloudFilter,
)
from reporting import EvidenceManager


class MultiFramePointCloudApplication:
    """Convert and filter every RGB-D frame without combining coordinate systems."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.input_path = project_root / "output" / "multiframe_rgbd_capture.npz"
        self.output_folder = project_root / "output" / "multiframe_clouds"
        self.evidence = EvidenceManager(project_root)
        self.point_filter = PointCloudFilter(
            neighbour_count=20,
            standard_ratio=2.0,
        )

    def run(self) -> int:
        if not self.input_path.is_file():
            print(f"RGB-D sequence not found: {self.input_path}")
            print("Run 07_test_multiframe_capture.py first.")
            return 1

        try:
            sequence = np.load(self.input_path, allow_pickle=False)
            depth_frames = sequence["depth_mm"]
            colour_frames = sequence["colour_bgr"]
            horizontal_fov = float(sequence["horizontal_fov"])
            vertical_fov = float(sequence["vertical_fov"])
            minimum_depth_mm = int(sequence["minimum_depth_mm"])
            maximum_depth_mm = int(sequence["maximum_depth_mm"])

            self._validate_sequence(depth_frames, colour_frames)
            frame_count, height, width = depth_frames.shape
            intrinsics = DepthIntrinsics.from_field_of_view(
                width,
                height,
                horizontal_fov,
                vertical_fov,
            )
            generator = ColouredPointCloudGenerator(
                intrinsics,
                minimum_depth_mm=minimum_depth_mm,
                maximum_depth_mm=maximum_depth_mm,
                pixel_stride=1,
            )

            self.output_folder.mkdir(parents=True, exist_ok=True)
            frame_results: list[tuple[int, int, int, float]] = []
            print(f"Stage 7B: converting {frame_count} saved RGB-D frames...")

            for frame_index in range(frame_count):
                started = time.perf_counter()
                raw_cloud = generator.generate_coloured(
                    depth_frames[frame_index],
                    colour_frames[frame_index],
                )
                filtered_cloud, filter_statistics = self.point_filter.filter(
                    raw_cloud
                )
                elapsed = time.perf_counter() - started

                output_path = self.output_folder / (
                    f"frame_{frame_index + 1:03d}_filtered.ply"
                )
                if not o3d.io.write_point_cloud(str(output_path), filtered_cloud):
                    print(f"Could not save frame {frame_index + 1}: {output_path}")
                    return 1

                frame_results.append(
                    (
                        frame_index + 1,
                        filter_statistics.input_points,
                        filter_statistics.output_points,
                        elapsed,
                    )
                )
                print(
                    f"Frame {frame_index + 1}: "
                    f"{filter_statistics.input_points:,} raw -> "
                    f"{filter_statistics.output_points:,} filtered points "
                    f"({elapsed:.3f} s)",
                    flush=True,
                )

            report = self._format_report(frame_results)
            report_path = self.evidence.save_text(
                "multiframe_point_clouds",
                "conversion_and_filtering_report",
                report,
            )
            print(report)
            print(f"Point-cloud folder: {self.output_folder}")
            print(f"Stage 7B report saved: {report_path}")
            self._open_first_frame_viewer()
            return 0

        except (KeyError, OSError, RuntimeError, ValueError) as error:
            print(f"Stage 7B failed: {error}")
            return 1

    @staticmethod
    def _validate_sequence(
        depth_frames: np.ndarray,
        colour_frames: np.ndarray,
    ) -> None:
        """Reject incomplete archives before any Open3D processing begins."""
        if depth_frames.ndim != 3:
            raise ValueError(f"Expected depth shape (N,H,W), got {depth_frames.shape}")
        if colour_frames.ndim != 4 or colour_frames.shape[-1] != 3:
            raise ValueError(
                f"Expected colour shape (N,H,W,3), got {colour_frames.shape}"
            )
        if depth_frames.shape[0] != colour_frames.shape[0]:
            raise ValueError("Depth and colour frame counts do not match")
        if depth_frames.shape[1:3] != colour_frames.shape[1:3]:
            raise ValueError("Depth and colour image sizes do not match")
        if depth_frames.shape[0] == 0:
            raise ValueError("The RGB-D archive contains no frames")

    @staticmethod
    def _format_report(
        frame_results: list[tuple[int, int, int, float]],
    ) -> str:
        lines = [
            "Stage 7B multi-frame point-cloud conversion",
            "============================================",
            "Each frame remains in its own camera coordinate system.",
            "No point clouds have been merged or registered yet.",
            "",
        ]
        for frame_number, raw_points, filtered_points, elapsed in frame_results:
            removed = raw_points - filtered_points
            retained = 100.0 * filtered_points / raw_points
            lines.extend(
                [
                    f"Frame {frame_number}",
                    f"  Raw points:       {raw_points:,}",
                    f"  Filtered points:  {filtered_points:,}",
                    f"  Removed points:   {removed:,}",
                    f"  Retained:         {retained:.2f}%",
                    f"  Processing time:  {elapsed:.3f} s",
                ]
            )

        filtered_counts = [result[2] for result in frame_results]
        processing_times = [result[3] for result in frame_results]
        lines.extend(
            [
                "",
                f"Mean filtered points: {float(np.mean(filtered_counts)):,.0f}",
                f"Point-count variation: {float(np.std(filtered_counts)):.1f}",
                f"Total processing time: {float(np.sum(processing_times)):.3f} s",
            ]
        )
        return "\n".join(lines)

    def _open_first_frame_viewer(self) -> None:
        first_cloud = self.output_folder / "frame_001_filtered.ply"
        screenshot = self.evidence.new_path(
            "multiframe_point_clouds",
            "frame_001_filtered_view",
            "png",
        )
        subprocess.run(
            [
                sys.executable,
                str(self.project_root / "view_point_cloud.py"),
                str(first_cloud),
                "--screenshot",
                str(screenshot),
                "--title",
                "Astra Pro - Stage 7B filtered frame 1",
            ],
            check=False,
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(MultiFramePointCloudApplication(root).run())
