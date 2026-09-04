"""Convert one depth frame from image coordinates into XYZ coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthIntrinsics:
    """Pinhole projection parameters for the active depth-stream mode."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_field_of_view(
        cls,
        width: int,
        height: int,
        horizontal_fov: float,
        vertical_fov: float,
    ) -> "DepthIntrinsics":
        # FOV is supplied by the active OpenNI depth stream. The focal lengths
        # below are in pixel units for the current frame resolution.
        fx = width / (2.0 * math.tan(horizontal_fov / 2.0))
        fy = height / (2.0 * math.tan(vertical_fov / 2.0))
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy)


@dataclass(frozen=True)
class DepthValidityStatistics:
    """Explain how the configured depth filter accepts or rejects pixels."""

    total_pixels: int
    zero_depth_pixels: int
    below_minimum_pixels: int
    above_maximum_pixels: int
    valid_pixels: int

    @property
    def valid_percentage(self) -> float:
        if self.total_pixels == 0:
            return 0.0
        return 100.0 * self.valid_pixels / self.total_pixels


class PointCloudGenerator:
    """Generate a filtered Open3D point cloud from `depth_mm`."""

    def __init__(
        self,
        intrinsics: DepthIntrinsics,
        minimum_depth_mm: int = 400,
        maximum_depth_mm: int = 5000,
        pixel_stride: int = 2,
    ) -> None:
        self.intrinsics = intrinsics
        self.minimum_depth_mm = minimum_depth_mm
        self.maximum_depth_mm = maximum_depth_mm
        self.pixel_stride = pixel_stride

    def calculate_statistics(
        self, depth_mm: np.ndarray
    ) -> DepthValidityStatistics:
        """Count why full-resolution depth pixels will be excluded."""
        zero = depth_mm == 0
        below = (depth_mm > 0) & (depth_mm < self.minimum_depth_mm)
        above = depth_mm > self.maximum_depth_mm
        valid = (
            (depth_mm >= self.minimum_depth_mm)
            & (depth_mm <= self.maximum_depth_mm)
        )
        return DepthValidityStatistics(
            total_pixels=int(depth_mm.size),
            zero_depth_pixels=int(np.count_nonzero(zero)),
            below_minimum_pixels=int(np.count_nonzero(below)),
            above_maximum_pixels=int(np.count_nonzero(above)),
            valid_pixels=int(np.count_nonzero(valid)),
        )

    def generate(self, depth_mm: np.ndarray) -> Any:
        """Convert valid depth pixels to XYZ points measured in metres."""
        # Open3D is loaded only when conversion is actually requested. This
        # avoids initializing its native DLLs during camera-driver startup.
        import open3d as o3d

        # Using every second pixel reduces the first preview from about 307,000
        # possible pixels to about 76,800 before invalid-depth filtering.
        sampled_depth = depth_mm[::self.pixel_stride, ::self.pixel_stride]
        sampled_v, sampled_u = np.indices(sampled_depth.shape)
        u = sampled_u * self.pixel_stride
        v = sampled_v * self.pixel_stride

        valid = (
            (sampled_depth >= self.minimum_depth_mm)
            & (sampled_depth <= self.maximum_depth_mm)
        )

        z_mm = sampled_depth[valid].astype(np.float32)
        u_valid = u[valid].astype(np.float32)
        v_valid = v[valid].astype(np.float32)

        # Pinhole back-projection. X points right, Y points up and Z points
        # forward from the depth-camera optical centre.
        x_mm = (u_valid - self.intrinsics.cx) * z_mm / self.intrinsics.fx
        y_mm = -(v_valid - self.intrinsics.cy) * z_mm / self.intrinsics.fy

        points_metres = np.column_stack((x_mm, y_mm, z_mm)) / 1000.0

        # Stage 3 has no RGB alignment, so colour each point only by normalized
        # depth to make the geometry easier to interpret in the 3D viewer.
        depth_scale = (
            (z_mm - self.minimum_depth_mm)
            / (self.maximum_depth_mm - self.minimum_depth_mm)
        )
        colours = np.column_stack(
            (1.0 - depth_scale, 0.35 + 0.45 * depth_scale, depth_scale)
        )
        colours = np.clip(colours, 0.0, 1.0)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_metres)
        point_cloud.colors = o3d.utility.Vector3dVector(colours)
        return point_cloud


class ColouredPointCloudGenerator(PointCloudGenerator):
    """Add registered RGB values to the Stage 3 XYZ conversion.

    This is inheritance because a coloured point-cloud generator still performs
    the same depth-to-XYZ operation, with RGB sampling added to the result.
    """

    def generate_coloured(
        self,
        depth_mm: np.ndarray,
        colour_bgr: np.ndarray,
    ) -> Any:
        """Convert aligned depth/RGB pixels into XYZRGB Open3D points."""
        import open3d as o3d

        if colour_bgr.shape[:2] != depth_mm.shape:
            raise ValueError(
                "Registered depth and RGB must have the same image size. "
                f"Depth={depth_mm.shape}, RGB={colour_bgr.shape[:2]}"
            )

        sampled_depth = depth_mm[::self.pixel_stride, ::self.pixel_stride]
        sampled_colour = colour_bgr[::self.pixel_stride, ::self.pixel_stride]
        sampled_v, sampled_u = np.indices(sampled_depth.shape)
        u = sampled_u * self.pixel_stride
        v = sampled_v * self.pixel_stride

        valid = (
            (sampled_depth >= self.minimum_depth_mm)
            & (sampled_depth <= self.maximum_depth_mm)
        )
        z_mm = sampled_depth[valid].astype(np.float32)
        u_valid = u[valid].astype(np.float32)
        v_valid = v[valid].astype(np.float32)

        x_mm = (u_valid - self.intrinsics.cx) * z_mm / self.intrinsics.fx
        y_mm = -(v_valid - self.intrinsics.cy) * z_mm / self.intrinsics.fy
        points_metres = np.column_stack((x_mm, y_mm, z_mm)) / 1000.0

        # OpenCV stores BGR. Open3D expects RGB normalized to the range 0..1.
        colours_rgb = sampled_colour[valid][:, ::-1].astype(np.float64) / 255.0

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_metres)
        point_cloud.colors = o3d.utility.Vector3dVector(colours_rgb)
        return point_cloud
