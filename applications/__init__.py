"""Runnable application controllers for individual project stages."""

from .depth_test_application import DepthTestApplication
from .rgb_test_application import RGBTestApplication
from .point_cloud_test_application import PointCloudTestApplication
from .rgbd_capability_application import RGBDCapabilityApplication
from .rgbd_alignment_application import RGBDAlignmentApplication
from .coloured_point_cloud_application import ColouredPointCloudApplication

__all__ = [
    "DepthTestApplication",
    "RGBTestApplication",
    "PointCloudTestApplication",
    "RGBDCapabilityApplication",
    "RGBDAlignmentApplication",
    "ColouredPointCloudApplication",
]
