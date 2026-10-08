"""Observed-surface coverage heuristic, independent of frame repetition."""
from __future__ import annotations

import numpy as np
import open3d as o3d


class SurfaceCoverage:
    def __init__(self, voxel_m=0.10, viewpoint_spacing_m=0.15):
        self.voxel_m = voxel_m
        self.viewpoint_spacing_m = viewpoint_spacing_m
        self.views = {}

    def update(self, world_points, camera_position):
        points = np.asarray(world_points)
        points = points[np.isfinite(points).all(axis=1)]
        station = tuple(np.floor(np.asarray(camera_position) / self.viewpoint_spacing_m).astype(int))
        cells = np.unique(np.floor(points / self.voxel_m).astype(int), axis=0)
        for cell in cells:
            self.views.setdefault(tuple(cell), set()).add(station)

    def cloud(self):
        cloud = o3d.geometry.PointCloud()
        if not self.views:
            return cloud
        keys = list(self.views)
        cloud.points = o3d.utility.Vector3dVector((np.asarray(keys) + .5) * self.voxel_m)
        counts = np.array([len(self.views[k]) for k in keys])
        colors = np.tile([.15, .8, .3], (len(keys), 1))
        colors[counts == 1] = [.95, .15, .12]
        colors[counts == 2] = [1., .65, .1]
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def summary(self):
        counts = np.array([len(v) for v in self.views.values()])
        return {
            "observed_surface_cells": len(counts),
            "red_single_viewpoint_cells": int(np.count_nonzero(counts == 1)),
            "amber_two_viewpoint_cells": int(np.count_nonzero(counts == 2)),
            "green_three_or_more_viewpoint_cells": int(np.count_nonzero(counts >= 3)),
            "surface_cell_m": self.voxel_m,
            "viewpoint_bin_m": self.viewpoint_spacing_m,
            "interpretation": "Red: revisit from an overlapping shifted view. Amber: add another view. Green: observed from at least three position bins; not a guarantee of accuracy.",
            "limitations": "Only observed surfaces are scored. Unseen walls are unknown. Pose drift and viewpoint-bin boundaries affect scores. No overall room-completion percentage is claimed.",
        }
