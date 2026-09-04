"""Open one saved PLY in an isolated Open3D viewer process."""

import argparse
from pathlib import Path
import time

import numpy as np
import open3d as o3d


class PointCloudViewer:
    """Load and display a saved point cloud without opening camera drivers."""

    def __init__(
        self,
        point_cloud_path: Path,
        screenshot_path: Path | None = None,
        window_title: str = "Astra Pro - depth-only XYZ point cloud",
    ) -> None:
        self.point_cloud_path = point_cloud_path
        self.screenshot_path = screenshot_path
        self.window_title = window_title

    def run(self) -> int:
        if not self.point_cloud_path.is_file():
            print(f"PLY file does not exist: {self.point_cloud_path}")
            return 1

        point_cloud = o3d.io.read_point_cloud(str(self.point_cloud_path))
        if point_cloud.is_empty():
            print(f"PLY contains no points: {self.point_cloud_path}")
            return 1

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.25
        )
        print(f"Displaying {len(point_cloud.points):,} points.")
        print("Close the Open3D window when inspection is finished.")
        viewer = o3d.visualization.Visualizer()
        window_created = viewer.create_window(
            window_name=self.window_title,
            width=1100,
            height=750,
        )
        if not window_created:
            print("Open3D could not create its viewer window.")
            return 1

        # On Windows, create_window() may return before GLFW has delivered the
        # first real window-size event. Adding geometry or setting ViewControl
        # too early causes: "SetViewPoint() failed because window height and
        # width are not set." Process a few events before touching the view.
        self._warm_up_window(viewer)

        viewer.add_geometry(point_cloud)
        viewer.add_geometry(coordinate_frame)
        self._set_camera_front_view(viewer, point_cloud)
        self._configure_rendering(viewer)
        self._render_frames(viewer, frame_count=5)

        if self.screenshot_path is not None:
            self.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            viewer.capture_screen_image(str(self.screenshot_path), do_render=True)
            print(f"Screenshot saved: {self.screenshot_path}")

        viewer.run()
        viewer.destroy_window()
        return 0

    @staticmethod
    def _warm_up_window(viewer: o3d.visualization.Visualizer) -> None:
        """Allow Windows/GLFW to establish the framebuffer dimensions."""
        for _ in range(10):
            viewer.poll_events()
            viewer.update_renderer()
            time.sleep(0.02)

    @staticmethod
    def _set_camera_front_view(
        viewer: o3d.visualization.Visualizer,
        point_cloud: o3d.geometry.PointCloud,
    ) -> None:
        """Look from the RGB-D camera origin towards positive Z.

        The generated coordinate convention is X right, Y up and Z forward.
        A front vector of +Z therefore reproduces the camera-facing view rather
        than showing the single-frame cloud as thin sheets from the side.
        """
        centre = point_cloud.get_axis_aligned_bounding_box().get_center()
        control = viewer.get_view_control()
        control.set_lookat(np.asarray(centre, dtype=float))
        control.set_front([0.0, 0.0, 1.0])
        control.set_up([0.0, 1.0, 0.0])
        control.set_zoom(0.55)

    @staticmethod
    def _configure_rendering(viewer: o3d.visualization.Visualizer) -> None:
        """Make sparse depth samples easier to inspect."""
        options = viewer.get_render_option()
        options.point_size = 2.5
        options.background_color = np.asarray([0.06, 0.07, 0.08])

    @staticmethod
    def _render_frames(
        viewer: o3d.visualization.Visualizer,
        frame_count: int,
    ) -> None:
        """Apply the new camera and render settings before a screenshot."""
        for _ in range(frame_count):
            viewer.poll_events()
            viewer.update_renderer()
            time.sleep(0.02)


class ViewerConfiguration:
    """Read the PLY filename supplied on the command line."""

    @staticmethod
    def from_command_line() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="View one PLY point cloud")
        parser.add_argument("point_cloud", type=Path)
        parser.add_argument(
            "--screenshot",
            type=Path,
            default=None,
            help="Optional output path for an automatic PNG capture",
        )
        parser.add_argument(
            "--title",
            default="Astra Pro - depth-only XYZ point cloud",
            help="Viewer window title",
        )
        return parser.parse_args()


if __name__ == "__main__":
    settings = ViewerConfiguration.from_command_line()
    viewer = PointCloudViewer(
        settings.point_cloud,
        settings.screenshot,
        settings.title,
    )
    raise SystemExit(viewer.run())
