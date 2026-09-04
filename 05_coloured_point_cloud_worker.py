"""Native Stage 5 worker launched and monitored by the supervisor."""

from pathlib import Path

from applications import ColouredPointCloudApplication


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    application = ColouredPointCloudApplication(
        project_root / "output" / "coloured_point_cloud.ply"
    )
    raise SystemExit(application.run())
