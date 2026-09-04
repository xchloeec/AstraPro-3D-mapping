"""Manual integration test for the OOP depth-camera components."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from applications import DepthTestApplication


class DepthCameraIntegrationTest:
    """Check the camera, processor and application together on real hardware."""

    def __init__(self) -> None:
        self.application = DepthTestApplication(display_limit_mm=5000)

    def run(self) -> int:
        return self.application.run()


if __name__ == "__main__":
    test = DepthCameraIntegrationTest()
    raise SystemExit(test.run())
