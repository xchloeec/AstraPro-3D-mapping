"""Turn the Astra Pro's two camera interfaces into one standard RGB-D frame.

The original Astra Pro does not deliver colour and depth through one API:
OpenNI2 supplies depth while Windows UVC supplies colour.  This adapter is the
small bridge between that camera-specific behaviour and later SLAM code.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import threading
import time

import cv2
import numpy as np

from processing import DepthIntrinsics

from .rgbd_camera import RGBDCamera


@dataclass(frozen=True)
class StandardRGBDFrame:
    """One RGB-D observation with timing and calibration information."""

    sequence: int
    colour_bgr: np.ndarray
    depth_mm: np.ndarray
    depth_timestamp_ns: int
    colour_timestamp_ns: int
    intrinsics: DepthIntrinsics

    @property
    def timestamp_difference_ms(self) -> float:
        """Software time separation between the depth and colour reads."""
        return abs(self.colour_timestamp_ns - self.depth_timestamp_ns) / 1_000_000

    @property
    def centre_depth_mm(self) -> int:
        row = self.depth_mm.shape[0] // 2
        column = self.depth_mm.shape[1] // 2
        return int(self.depth_mm[row, column])


class RGBDFrameAdapter:
    """Read, validate and timestamp the Astra Pro RGB and depth pair."""

    def __init__(self, camera: RGBDCamera | None = None) -> None:
        self.camera = camera or RGBDCamera(rgb_camera_index=0)
        self._sequence = 0
        self._intrinsics: DepthIntrinsics | None = None
        # UVC colour is continuously drained in a background thread. A small
        # ring buffer lets the depth reader select the colour frame closest in
        # software time instead of starting a new blocking RGB read afterward.
        self._colour_buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=8)
        self._colour_condition = threading.Condition()
        self._stop_colour_reader = threading.Event()
        self._colour_thread: threading.Thread | None = None
        self._colour_error: Exception | None = None

    def start(self) -> "RGBDFrameAdapter":
        self.camera.start()
        horizontal_fov, vertical_fov = self.camera.depth_camera.get_field_of_view()
        self._intrinsics = DepthIntrinsics.from_field_of_view(
            width=640,
            height=480,
            horizontal_fov=horizontal_fov,
            vertical_fov=vertical_fov,
        )
        self._stop_colour_reader.clear()
        self._colour_error = None
        self._colour_thread = threading.Thread(
            target=self._colour_reader_loop,
            name="AstraUVCColourReader",
            daemon=True,
        )
        self._colour_thread.start()

        # Do not return until at least one valid UVC frame is available.
        deadline = time.perf_counter() + 2.0
        with self._colour_condition:
            while not self._colour_buffer and self._colour_error is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._colour_condition.wait(timeout=remaining)
        if self._colour_error is not None:
            raise RuntimeError("Background RGB reader failed.") from self._colour_error
        if not self._colour_buffer:
            raise RuntimeError("Background RGB reader did not produce a frame.")
        return self

    def read(self) -> StandardRGBDFrame:
        """Read depth first and colour second, recording both software times.

        The timestamps are taken immediately after each blocking read returns.
        They are not hardware timestamps, but their difference tells us how far
        apart the two independently delivered images reached Python.
        """
        if self._intrinsics is None:
            raise RuntimeError("RGBDFrameAdapter is not running. Call start() first.")

        depth_mm = self.camera.depth_camera.read()
        depth_timestamp_ns = time.perf_counter_ns()
        colour_timestamp_ns, colour_bgr = self._nearest_colour(depth_timestamp_ns)

        self._validate_depth(depth_mm)
        colour_bgr = self._prepare_colour(colour_bgr, depth_mm.shape)

        frame = StandardRGBDFrame(
            sequence=self._sequence,
            colour_bgr=colour_bgr,
            depth_mm=depth_mm,
            depth_timestamp_ns=depth_timestamp_ns,
            colour_timestamp_ns=colour_timestamp_ns,
            intrinsics=self._intrinsics,
        )
        self._sequence += 1
        return frame

    def _colour_reader_loop(self) -> None:
        """Continuously remove UVC frames so an old buffered frame is not used."""
        try:
            while not self._stop_colour_reader.is_set():
                colour_bgr = self.camera.rgb_camera.read()
                timestamp_ns = time.perf_counter_ns()
                with self._colour_condition:
                    self._colour_buffer.append((timestamp_ns, colour_bgr.copy()))
                    self._colour_condition.notify_all()
        except Exception as error:
            if not self._stop_colour_reader.is_set():
                with self._colour_condition:
                    self._colour_error = error
                    self._colour_condition.notify_all()

    def _nearest_colour(self, depth_timestamp_ns: int) -> tuple[int, np.ndarray]:
        """Choose the buffered RGB frame nearest to the current depth time.

        We briefly allow the next UVC frame to arrive. This means the selected
        colour can be just before or just after the depth read, cutting the
        expected difference to roughly half of one RGB frame period.
        """
        wait_deadline = time.perf_counter() + 0.050
        with self._colour_condition:
            while True:
                if self._colour_error is not None:
                    raise RuntimeError("Background RGB reader failed.") from self._colour_error
                if self._colour_buffer and self._colour_buffer[-1][0] >= depth_timestamp_ns:
                    break
                remaining = wait_deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._colour_condition.wait(timeout=remaining)

            if not self._colour_buffer:
                raise RuntimeError("No RGB frame is available for depth pairing.")
            timestamp_ns, colour_bgr = min(
                self._colour_buffer,
                key=lambda item: abs(item[0] - depth_timestamp_ns),
            )
            return timestamp_ns, colour_bgr.copy()

    @staticmethod
    def _validate_depth(depth_mm: np.ndarray) -> None:
        if depth_mm.ndim != 2:
            raise ValueError(f"Expected one-channel depth; got {depth_mm.shape}.")
        if depth_mm.dtype != np.uint16:
            raise ValueError(f"Expected uint16 depth in millimetres; got {depth_mm.dtype}.")

    @staticmethod
    def _prepare_colour(
        colour_bgr: np.ndarray,
        depth_shape: tuple[int, int],
    ) -> np.ndarray:
        if colour_bgr.ndim != 3 or colour_bgr.shape[2] != 3:
            raise ValueError(f"Expected three-channel BGR; got {colour_bgr.shape}.")
        depth_height, depth_width = depth_shape
        if colour_bgr.shape[:2] != depth_shape:
            colour_bgr = cv2.resize(
                colour_bgr,
                (depth_width, depth_height),
                interpolation=cv2.INTER_AREA,
            )
        return colour_bgr

    def close(self) -> None:
        self._stop_colour_reader.set()
        with self._colour_condition:
            self._colour_condition.notify_all()
        if self._colour_thread is not None:
            self._colour_thread.join(timeout=1.0)
            self._colour_thread = None
        self.camera.close()
        self._intrinsics = None
        with self._colour_condition:
            self._colour_buffer.clear()

    def __enter__(self) -> "RGBDFrameAdapter":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
