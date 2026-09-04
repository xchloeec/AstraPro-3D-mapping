"""Native Stage 4 worker monitored by the alignment supervisor."""

from applications import RGBDAlignmentApplication


if __name__ == "__main__":
    application = RGBDAlignmentApplication()
    raise SystemExit(application.run())
