"""Remove isolated measurement noise from a saved or generated point cloud."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PointCloudFilterStatistics:
    """Record how many points were accepted and rejected by filtering."""

    input_points: int
    output_points: int

    @property
    def removed_points(self) -> int:
        return self.input_points - self.output_points

    @property
    def retained_percentage(self) -> float:
        if self.input_points == 0:
            return 0.0
        return 100.0 * self.output_points / self.input_points


class PointCloudFilter:
    """Apply statistical outlier removal to an Open3D point cloud.

    For every point, Open3D measures the average distance to nearby points.
    Points whose neighbour distance is unusually large are treated as isolated
    depth noise. The original cloud is not modified.
    """

    def __init__(self, neighbour_count: int = 20, standard_ratio: float = 2.0) -> None:
        if neighbour_count < 2:
            raise ValueError("neighbour_count must be at least 2")
        if standard_ratio <= 0:
            raise ValueError("standard_ratio must be positive")
        self.neighbour_count = neighbour_count
        self.standard_ratio = standard_ratio

    def filter(self, point_cloud: Any) -> tuple[Any, PointCloudFilterStatistics]:
        """Return a filtered copy and statistics describing the operation."""
        input_points = len(point_cloud.points)
        if input_points == 0:
            raise ValueError("Cannot filter an empty point cloud")

        filtered_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=self.neighbour_count,
            std_ratio=self.standard_ratio,
        )
        statistics = PointCloudFilterStatistics(
            input_points=input_points,
            output_points=len(filtered_cloud.points),
        )
        return filtered_cloud, statistics
