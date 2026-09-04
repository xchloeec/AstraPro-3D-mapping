"""Stage 3 entry point: capture and display one depth-only XYZ point cloud."""

from pathlib import Path

from applications import PointCloudTestApplication


if __name__ == "__main__":
    output_file = Path("output") / "depth_point_cloud.ply"
    application = PointCloudTestApplication(output_path=output_file)
    raise SystemExit(application.run())
