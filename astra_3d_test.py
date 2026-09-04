"""Stage 2 entry point: run the separate OOP RGB-camera test."""

from applications import RGBTestApplication


if __name__ == "__main__":
    application = RGBTestApplication(camera_index=0)
    raise SystemExit(application.run())
