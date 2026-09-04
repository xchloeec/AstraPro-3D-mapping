"""OpenNI2 interface used by the Astra Pro mapping experiments.

The hardware-specific code is kept here so later point-cloud and mapping code
does not need to deal directly with OpenNI setup and shutdown.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

import numpy as np


class OpenNIError(RuntimeError):
    """Raised when OpenNI2 or the depth camera cannot be initialized."""


KNOWN_OPENNI_PATHS = (
    Path(
        r"C:\1 Swinburne\Sem 6\openni\OpenNI"
        r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
        r"\Win64-Release\sdk\libs"
    ),
    Path(r"C:\Program Files\OpenNI2\Redist"),
    Path(r"C:\Program Files\Orbbec\OpenNI2\Redist"),
)


def _load_openni_module() -> Any:
    # The installed package normally uses `openni`, although some OpenNI2
    # Python wrappers use the older `primesense` namespace. Supporting both
    # avoids coupling this project to one wrapper layout.
    try:
        from openni import openni2

        return openni2
    except ImportError:
        try:
            from primesense import openni2

            return openni2
        except ImportError as exc:
            raise OpenNIError(
                "OpenNI Python bindings are missing. In the PyCharm terminal run: "
                "python -m pip install -r requirements.txt"
            ) from exc


def find_openni_runtime(explicit_path: str | Path | None = None) -> Path:
    """Find the OpenNI runtime directory containing OpenNI2.dll."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    environment_path = os.environ.get("OPENNI2_REDIST")
    if environment_path:
        candidates.append(Path(environment_path))

    candidates.extend(KNOWN_OPENNI_PATHS)
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "OpenNI2.dll").is_file():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise OpenNIError(
        "Could not locate OpenNI2.dll. Set OPENNI2_REDIST to its folder.\n"
        f"Searched:\n{searched}"
    )


class DepthCamera:
    """Context-managed Astra depth stream."""

    def __init__(
        self,
        runtime_path: str | Path | None = None,
        mirrored: bool = False,
        register_depth_to_colour: bool = False,
    ) -> None:
        self.runtime_path = find_openni_runtime(runtime_path)
        self.openni2 = _load_openni_module()
        self.mirrored = mirrored
        self.register_depth_to_colour = register_depth_to_colour
        self.registration_enabled = False
        self.device: Any | None = None
        self.stream: Any | None = None

    def start(self) -> "DepthCamera":
        try:
            # OpenNI loads the Orbbec driver from the OpenNI2/Drivers folder
            # beside the DLL, then opens the first available depth device.
            self.openni2.initialize(str(self.runtime_path))
            self.device = self.openni2.Device.open_any()
            self.stream = self.device.create_depth_stream()

            # OpenNI may mirror depth by default for a selfie-style preview.
            # Mapping requires camera coordinates rather than a mirror image,
            # so keep it unmirrored to match the Astra Pro UVC RGB stream.
            self.stream.set_mirroring_enabled(self.mirrored)

            # Astra Pro's official D2C sample enables hardware registration
            # after creating the depth stream but before starting that stream.
            # Changing this property after frames are already flowing proved
            # unstable when OpenNI and Windows UVC were used together.
            if self.register_depth_to_colour:
                registration_mode = (
                    self.openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
                )
                if not self.device.is_image_registration_mode_supported(
                    registration_mode
                ):
                    raise OpenNIError(
                        "Depth-to-colour registration is not supported."
                    )
                self.device.set_image_registration_mode(registration_mode)
                self.registration_enabled = (
                    self.device.get_image_registration_mode()
                    == registration_mode
                )
                if not self.registration_enabled:
                    raise OpenNIError(
                        "OpenNI did not enable depth-to-colour registration."
                    )
            self.stream.start()
        except Exception as exc:
            self.close()
            raise OpenNIError(
                "Could not start the depth stream. Close NiViewer/OrbbecViewer and "
                "other camera programs, reconnect the Astra, then try again. "
                f"OpenNI reported: {exc}"
            ) from exc
        return self

    def read(self) -> np.ndarray:
        """Read one depth image as a uint16 array, normally in millimetres."""
        if self.stream is None:
            raise OpenNIError("Depth stream is not running. Call start() first.")

        frame = self.stream.read_frame()
        height = frame.height
        width = frame.width
        # Copying the reshaped buffer is intentional. OpenNI can reuse its frame
        # memory after the next read, while NumPy data may still be processed.
        depth = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16)
        return depth.reshape((height, width)).copy()

    def get_field_of_view(self) -> tuple[float, float]:
        """Return horizontal and vertical depth FOV in radians."""
        if self.stream is None:
            raise OpenNIError("Depth stream is not running. Call start() first.")

        horizontal_fov = float(self.stream.get_horizontal_fov())
        vertical_fov = float(self.stream.get_vertical_fov())
        if not (0.0 < horizontal_fov < math.pi):
            raise OpenNIError(f"Invalid horizontal FOV: {horizontal_fov}")
        if not (0.0 < vertical_fov < math.pi):
            raise OpenNIError(f"Invalid vertical FOV: {vertical_fov}")
        return horizontal_fov, vertical_fov

    def inspect_rgbd_capabilities(self) -> dict[str, bool]:
        """Query RGB-D features exposed by this device through OpenNI2.

        This does not start a colour stream or change registration settings.
        It only reports what the active OpenNI driver says it supports.
        """
        if self.device is None:
            raise OpenNIError("Depth device is not open. Call start() first.")

        has_depth = bool(self.device.has_sensor(self.openni2.SENSOR_DEPTH))
        has_colour = bool(self.device.has_sensor(self.openni2.SENSOR_COLOR))
        has_ir = bool(self.device.has_sensor(self.openni2.SENSOR_IR))
        registration_supported = bool(
            self.device.is_image_registration_mode_supported(
                self.openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
            )
        )

        try:
            depth_colour_sync_enabled = bool(
                self.device.get_depth_color_sync_enabled()
            )
        except Exception:
            depth_colour_sync_enabled = False

        return {
            "openni_depth_sensor": has_depth,
            "openni_colour_sensor": has_colour,
            "openni_ir_sensor": has_ir,
            "depth_to_colour_registration": registration_supported,
            "depth_colour_sync_currently_enabled": depth_colour_sync_enabled,
        }

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            self.stream = None
        self.device = None
        try:
            if self.openni2.is_initialized():
                self.openni2.unload()
        except Exception:
            pass

    def __enter__(self) -> "DepthCamera":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
