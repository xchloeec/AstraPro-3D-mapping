"""Camera interfaces for the Astra 3D mapping project."""

from .depth_camera import DepthCamera, OpenNIError
from .rgb_camera import RGBCamera, RGBCameraError
from .rgbd_camera import RGBDCamera
from .rgbd_adapter import RGBDFrameAdapter, StandardRGBDFrame

__all__ = [
    "DepthCamera",
    "OpenNIError",
    "RGBCamera",
    "RGBCameraError",
    "RGBDCamera",
    "RGBDFrameAdapter",
    "StandardRGBDFrame",
]
