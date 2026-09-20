"""Combined Astra Pro RGB-D interface.

The original Astra Pro exposes depth through OpenNI and colour through a
separate Windows UVC interface. Therefore this class deliberately combines
DepthCamera (OpenNI) and RGBCamera (OpenCV) instead of trying to read colour
frames from OpenNI.
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from .depth_camera import DepthCamera, OpenNIError
from .rgb_camera import RGBCamera


class RGBDCamera:
    """Coordinate the Astra Pro's separate depth and RGB interfaces."""

    def __init__(
        self,
        runtime_path: str | Path | None = None,
        rgb_camera_index: int = 0,
        rgb_backends: tuple[int, ...] | None = None,
    ) -> None:
        self.depth_camera = DepthCamera(
            runtime_path,
            mirrored=False,
            register_depth_to_colour=True,
        )
        self.rgb_camera = RGBCamera(
            camera_index=rgb_camera_index,
            requested_width=640,
            requested_height=480,
            # MSMF has been more stable than DirectShow while the OpenNI
            # interface of the same composite Astra device is already active.
            backends=rgb_backends or (cv2.CAP_MSMF, cv2.CAP_DSHOW),
        )
        self.registration_enabled = False

        # OpenNI synchronization only works when both streams pass through
        # OpenNI. Astra Pro RGB is UVC/OpenCV, so this honestly remains False.
        self.synchronization_enabled = False

    @property
    def runtime_path(self) -> Path:
        return self.depth_camera.runtime_path

    def start(self) -> "RGBDCamera":
        """Start OpenNI depth, enable D2C registration, then start UVC RGB."""
        try:
            print("RGB-D startup 1/4: starting registered OpenNI depth...", flush=True)
            self.depth_camera.start()
            self.registration_enabled = self.depth_camera.registration_enabled
            if not self.registration_enabled:
                raise OpenNIError("OpenNI did not enable depth-to-colour registration.")

            print("RGB-D startup 2/4: warming up depth stream...", flush=True)
            for _ in range(12):
                self.depth_camera.read()

            # Let the composite USB device finish allocating its depth
            # endpoints before Windows opens the independent UVC endpoint.
            print("RGB-D startup 3/4: waiting before UVC startup...", flush=True)
            time.sleep(0.5)

            print("RGB-D startup 4/4: starting Astra UVC RGB...", flush=True)
            self.rgb_camera.start()
            print(
                "RGB-D startup complete. RGB backend: "
                f"{self.rgb_camera.active_backend_name}",
                flush=True,
            )
            return self
        except Exception:
            self.close()
            raise

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return registered depth in mm and the nearest available BGR frame.

        These frames come from two USB interfaces, so they are sequentially
        acquired rather than hardware synchronized.
        """
        depth_mm = self.depth_camera.read()
        colour_bgr = self.rgb_camera.read()
        return depth_mm, colour_bgr

    def close(self) -> None:
        self.rgb_camera.close()
        self.depth_camera.close()

    def __enter__(self) -> "RGBDCamera":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
