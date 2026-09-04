"""Native worker for the Stage 5A stride-1 density experiment."""

from pathlib import Path

from applications import ColouredPointCloudApplication


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    application = ColouredPointCloudApplication(
        output_path=project_root / "output" / "dense_coloured_point_cloud.ply",
        minimum_depth_mm=400,
        maximum_depth_mm=8000,
        pixel_stride=1,
        experiment_name="8m_stride1",
    )
    raise SystemExit(application.run())
