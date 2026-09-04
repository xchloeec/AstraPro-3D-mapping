"""Stage 6: filter the saved dense cloud without reopening the camera."""

from pathlib import Path
import subprocess
import sys
import time

import open3d as o3d

from processing import PointCloudFilter
from reporting import EvidenceManager


class PointCloudFilterApplication:
    """Load, filter, save and display one previously captured point cloud."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.input_path = project_root / "output" / "dense_coloured_point_cloud.ply"
        self.output_path = project_root / "output" / "filtered_coloured_point_cloud.ply"
        self.evidence = EvidenceManager(project_root)
        self.point_filter = PointCloudFilter(
            neighbour_count=20,
            standard_ratio=2.0,
        )

    def run(self) -> int:
        if not self.input_path.is_file():
            print(f"Input cloud was not found: {self.input_path}")
            print("Run 05a_test_dense_coloured_point_cloud.py first.")
            return 1

        cloud = o3d.io.read_point_cloud(str(self.input_path))
        if cloud.is_empty():
            print(f"Input cloud contains no points: {self.input_path}")
            return 1

        started = time.perf_counter()
        filtered_cloud, statistics = self.point_filter.filter(cloud)
        filtering_seconds = time.perf_counter() - started

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(self.output_path), filtered_cloud):
            print(f"Could not save: {self.output_path}")
            return 1

        report = self._format_report(statistics, filtering_seconds)
        report_path = self.evidence.save_text(
            "point_cloud_filter", "statistical_filter_report", report
        )
        print(report)
        print(f"Filtered cloud saved: {self.output_path}")
        print(f"Filter evidence saved: {report_path}")
        self._open_viewer()
        return 0

    def _format_report(self, statistics, filtering_seconds: float) -> str:
        return "\n".join(
            [
                "Stage 6 statistical point-cloud filtering",
                "=========================================",
                "Method: statistical outlier removal",
                f"Neighbour count:          {self.point_filter.neighbour_count}",
                f"Standard-deviation ratio: {self.point_filter.standard_ratio:.2f}",
                f"Input points:             {statistics.input_points:,}",
                f"Output points:            {statistics.output_points:,}",
                f"Removed points:           {statistics.removed_points:,}",
                f"Retained percentage:      {statistics.retained_percentage:.2f}%",
                f"Filtering time:           {filtering_seconds:.3f} s",
            ]
        )

    def _open_viewer(self) -> None:
        screenshot = self.evidence.new_path(
            "point_cloud_filter", "filtered_xyzrgb_view", "png"
        )
        subprocess.run(
            [
                sys.executable,
                str(self.project_root / "view_point_cloud.py"),
                str(self.output_path),
                "--screenshot",
                str(screenshot),
                "--title",
                "Astra Pro - statistically filtered coloured point cloud",
            ],
            check=False,
        )


if __name__ == "__main__":
    raise SystemExit(PointCloudFilterApplication(Path(__file__).resolve().parent).run())
