"""Stage 2 entry point: test the Astra Pro RGB camera independently.

Examples from the PyCharm terminal:
    python 02_test_rgb_camera.py --camera-index 0
    python 02_test_rgb_camera.py --camera-index 1

The correct index is the one that displays the Astra Pro view rather than the
laptop's built-in webcam. Depth/OpenNI2 is intentionally not used in Stage 2.
"""

import argparse

from applications import RGBTestApplication


class RGBTestConfiguration:
    """Read the camera index selected by the user."""

    @staticmethod
    def from_command_line() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Test one RGB camera index")
        parser.add_argument(
            "--camera-index",
            type=int,
            default=0,
            help="OpenCV camera index to test (default: 0)",
        )
        return parser.parse_args()


if __name__ == "__main__":
    settings = RGBTestConfiguration.from_command_line()
    application = RGBTestApplication(camera_index=settings.camera_index)
    raise SystemExit(application.run())
