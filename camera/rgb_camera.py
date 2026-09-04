"""OpenCV interface for the Astra Pro RGB camera."""

from __future__ import annotations

import cv2
import numpy as np


class RGBCameraError(RuntimeError):
    """Raised when the RGB camera cannot be opened or read."""


class RGBCamera:
    """Own the OpenCV VideoCapture object for one RGB camera."""

    def __init__(
        self,
        camera_index: int = 0,
        requested_width: int = 1280,
        requested_height: int = 720,
        backends: tuple[int, ...] = (cv2.CAP_DSHOW, cv2.CAP_MSMF), #directShow/mediaFoundation
    ) -> None:
        self.camera_index = camera_index
        self.requested_width = requested_width
        self.requested_height = requested_height
        self.backends = backends
        self.active_backend_name: str | None = None
        self.capture: cv2.VideoCapture | None = None

    def start(self) -> "RGBCamera":
        # Do not use CAP_ANY here: it selected OBSENSOR instead of the Astra's
        # normal RGB/UVC interface. Try both standard Windows webcam backends.
        attempted_backends: list[str] = []
        for backend in self.backends:
            backend_name = cv2.videoio_registry.getBackendName(backend)
            attempted_backends.append(backend_name)
            candidate = cv2.VideoCapture(self.camera_index, backend)
            if candidate.isOpened():
                self.capture = candidate
                self.active_backend_name = backend_name
                break
            candidate.release()

        if self.capture is None:
            attempted = ", ".join(attempted_backends)
            raise RGBCameraError(
                f"RGB camera index {self.camera_index} could not be opened "
                f"through the Windows backends: {attempted}."
            )

        # These are requests. The camera/driver may select the closest mode it
        # supports, so the application reads back and displays the actual mode.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        # Read one frame during startup. This verifies that an opened handle is
        # actually the RGB stream rather than a non-video sensor interface.
        frame_received, _ = self.capture.read()
        if not frame_received:
            self.close()
            raise RGBCameraError(
                f"Camera index {self.camera_index} opened but did not provide "
                f"an RGB frame through {self.active_backend_name}."
            )
        return self

    def get_stream_information(self) -> tuple[int, int, float]:
        """Return the actual width, height and FPS reported by OpenCV."""
        if self.capture is None:
            raise RGBCameraError("RGB camera is not running. Call start() first.")

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        return width, height, fps

    def read(self) -> np.ndarray:
        if self.capture is None:
            raise RGBCameraError("RGB camera is not running. Call start() first.")

        frame_received, frame = self.capture.read()
        if not frame_received:
            raise RGBCameraError("The RGB camera did not return a frame.")
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "RGBCamera":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
