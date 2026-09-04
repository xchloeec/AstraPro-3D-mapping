"""Stage 4 entry point: determine the correct RGB-D alignment route."""

from applications import RGBDCapabilityApplication


if __name__ == "__main__":
    application = RGBDCapabilityApplication()
    raise SystemExit(application.run())
