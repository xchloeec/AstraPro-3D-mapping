"""OOP controller for generating one depth-only point cloud."""

from pathlib import Path
import subprocess
import sys

from camera import DepthCamera, OpenNIError
from processing import DepthIntrinsics, PointCloudGenerator
from reporting import EvidenceManager


class PointCloudTestApplication:
    """Capture one depth frame, generate XYZ, save it and open a 3D viewer."""

    def __init__(self, output_path: Path) -> None:
        self.camera = DepthCamera()
        self.output_path = output_path
        self.evidence = EvidenceManager()

    def run(self) -> int:
        try:
            with self.camera:
                # Discard a few initial frames so the stream has stabilized.
                for _ in range(15):
                    depth_mm = self.camera.read()

                horizontal_fov, vertical_fov = self.camera.get_field_of_view()
                height, width = depth_mm.shape
                intrinsics = DepthIntrinsics.from_field_of_view(
                    width, height, horizontal_fov, vertical_fov
                )

                generator = PointCloudGenerator(intrinsics)
                point_cloud = generator.generate(depth_mm)

            point_count = len(point_cloud.points)
            if point_count == 0:
                print("Point-cloud generation failed: no valid depth points.")
                return 1

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            import open3d as o3d

            saved = o3d.io.write_point_cloud(str(self.output_path), point_cloud)
            if not saved:
                print(f"Could not save point cloud to {self.output_path}")
                return 1

            print(f"Generated {point_count:,} XYZ points.")
            print(f"Saved: {self.output_path.resolve()}")
            print("Opening the saved cloud in a separate Open3D process.")
            self._open_external_viewer()
            return 0

        except (OpenNIError, RuntimeError) as error:
            print(f"Point-cloud test failed: {error}")
            return 1

    def _open_external_viewer(self) -> None:
        """Isolate Open3D's Windows GUI from the camera acquisition process.

        Some Windows/Open3D combinations display correctly but crash inside
        native GLFW cleanup when the window closes. A child process prevents
        that GUI cleanup issue from corrupting the successful capture process.
        """
        project_root = Path(__file__).resolve().parents[1]
        viewer_script = project_root / "view_point_cloud.py"
        screenshot_path = self.evidence.new_path(
            "point_cloud", "depth_xyz_view", "png"
        )
        viewer_result = subprocess.run(
            [
                sys.executable,
                str(viewer_script),
                str(self.output_path.resolve()),
                "--screenshot",
                str(screenshot_path),
            ],
            check=False,
        )

        if screenshot_path.is_file():
            print(f"Point-cloud evidence saved: {screenshot_path}")

        if viewer_result.returncode != 0:
            print(
                "Open3D viewer closed with native Windows code "
                f"{viewer_result.returncode}. The saved PLY is still valid."
            )
