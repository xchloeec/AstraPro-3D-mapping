"""Stage 8: export the saved Astra RGB-D sequence for Open3D reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import open3d as o3d

from processing import DepthIntrinsics
from reporting import EvidenceManager


class Open3DDatasetExporter:
    """Create the directory and calibration format used by Open3D examples."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.input_path = project_root / "output" / "multiframe_rgbd_capture.npz"
        self.dataset_path = project_root / "output" / "open3d_room_dataset"
        self.colour_path = self.dataset_path / "color"
        self.depth_path = self.dataset_path / "depth"
        self.intrinsic_path = self.dataset_path / "intrinsic.json"
        self.config_path = self.dataset_path / "open3d_config.json"
        self.evidence = EvidenceManager(project_root)

    def run(self) -> int:
        if not self.input_path.is_file():
            print(f"Input RGB-D sequence not found: {self.input_path}")
            print("Run 07_test_multiframe_capture.py first.")
            return 1

        try:
            with np.load(self.input_path, allow_pickle=False) as sequence:
                depth_frames = sequence["depth_mm"]
                colour_frames = sequence["colour_bgr"]
                horizontal_fov = float(sequence["horizontal_fov"])
                vertical_fov = float(sequence["vertical_fov"])
                valid_percentages = sequence["valid_percentage"].copy()

            self._validate_arrays(depth_frames, colour_frames)
            frame_count, height, width = depth_frames.shape
            intrinsics = DepthIntrinsics.from_field_of_view(
                width,
                height,
                horizontal_fov,
                vertical_fov,
            )

            # Only these generated subfolders are replaced. The whole output
            # directory and all earlier experiments remain untouched.
            self._prepare_output_folder(self.colour_path)
            self._prepare_output_folder(self.depth_path)

            for index in range(frame_count):
                colour_file = self.colour_path / f"{index:05d}.jpg"
                depth_file = self.depth_path / f"{index:05d}.png"
                if not cv2.imwrite(
                    str(colour_file),
                    colour_frames[index],
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(f"Could not save RGB frame: {colour_file}")
                if not cv2.imwrite(str(depth_file), depth_frames[index]):
                    raise RuntimeError(f"Could not save depth frame: {depth_file}")

            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
            )
            if not o3d.io.write_pinhole_camera_intrinsic(
                str(self.intrinsic_path), camera_intrinsic
            ):
                raise RuntimeError(
                    f"Could not save camera intrinsics: {self.intrinsic_path}"
                )

            config = self._create_config()
            self.config_path.write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )

            validation = self._validate_open3d_round_trip(
                width,
                height,
                frame_count,
            )
            report = self._format_report(
                frame_count,
                width,
                height,
                intrinsics,
                valid_percentages,
                validation,
            )
            report_path = self.evidence.save_text(
                "open3d_dataset", "dataset_export_validation", report
            )
            print(report)
            print(f"Open3D dataset saved: {self.dataset_path}")
            print(f"Open3D config saved: {self.config_path}")
            print(f"Validation report saved: {report_path}")
            return 0

        except (KeyError, OSError, RuntimeError, ValueError) as error:
            print(f"Stage 8 export failed: {error}")
            return 1

    @staticmethod
    def _prepare_output_folder(folder: Path) -> None:
        """Recreate one generated frame folder so stale images cannot remain."""
        if folder.is_dir():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_arrays(
        depth_frames: np.ndarray,
        colour_frames: np.ndarray,
    ) -> None:
        if depth_frames.ndim != 3 or depth_frames.dtype != np.uint16:
            raise ValueError(
                "Depth data must have shape (N,H,W) and uint16 millimetres; "
                f"received {depth_frames.shape}, {depth_frames.dtype}"
            )
        if colour_frames.ndim != 4 or colour_frames.shape[-1] != 3:
            raise ValueError(
                f"Colour data must have shape (N,H,W,3); got {colour_frames.shape}"
            )
        if depth_frames.shape[0] != colour_frames.shape[0]:
            raise ValueError("Depth and colour frame counts do not match")
        if depth_frames.shape[1:] != colour_frames.shape[1:3]:
            raise ValueError("Depth and colour frame sizes do not match")

    def _create_config(self) -> dict[str, object]:
        """Create a conservative starting configuration for an indoor room."""
        return {
            "name": "Astra Pro interior-room RGB-D reconstruction",
            "path_dataset": str(self.dataset_path.resolve()),
            "path_intrinsic": str(self.intrinsic_path.resolve()),
            "depth_map_type": "redwood",
            "depth_scale": 1000.0,
            "depth_min": 0.4,
            "depth_max": 5.0,
            "voxel_size": 0.025,
            "depth_diff_max": 0.07,
            "n_frames_per_fragment": 100,
            "n_keyframes_per_n_frame": 20,
            "preference_loop_closure_odometry": 0.1,
            "preference_loop_closure_registration": 5.0,
            "tsdf_cubic_size": 6.0,
            "icp_method": "color",
            "global_registration": "ransac",
            "python_multi_threading": False,
        }

    def _validate_open3d_round_trip(
        self,
        width: int,
        height: int,
        frame_count: int,
    ) -> dict[str, object]:
        """Read exported files through Open3D exactly as reconstruction will."""
        colour_files = sorted(self.colour_path.glob("*.jpg"))
        depth_files = sorted(self.depth_path.glob("*.png"))
        if len(colour_files) != frame_count or len(depth_files) != frame_count:
            raise RuntimeError("The number of exported RGB/depth files is incorrect")

        loaded_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            str(self.intrinsic_path)
        )
        if loaded_intrinsic.width != width or loaded_intrinsic.height != height:
            raise RuntimeError("Open3D read back the wrong intrinsic image size")

        colour = o3d.io.read_image(str(colour_files[0]))
        depth = o3d.io.read_image(str(depth_files[0]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colour,
            depth,
            depth_scale=1000.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            loaded_intrinsic,
        )
        if cloud.is_empty():
            raise RuntimeError("Open3D produced an empty cloud from exported frame 0")
        return {
            "colour_files": len(colour_files),
            "depth_files": len(depth_files),
            "round_trip_points": len(cloud.points),
        }

    @staticmethod
    def _format_report(
        frame_count: int,
        width: int,
        height: int,
        intrinsics: DepthIntrinsics,
        valid_percentages: np.ndarray,
        validation: dict[str, object],
    ) -> str:
        return "\n".join(
            [
                "Stage 8 Open3D dataset export",
                "=============================",
                f"RGB-D frame pairs:       {frame_count}",
                f"Image resolution:        {width} x {height}",
                "Depth storage:           uint16 PNG in millimetres",
                "Colour storage:          JPEG quality 95",
                "Depth scale:             1000 units per metre",
                f"fx:                      {intrinsics.fx:.6f} px",
                f"fy:                      {intrinsics.fy:.6f} px",
                f"cx:                      {intrinsics.cx:.6f} px",
                f"cy:                      {intrinsics.cy:.6f} px",
                f"Mean capture validity:   {float(np.mean(valid_percentages)):.2f}%",
                f"RGB files read back:     {validation['colour_files']}",
                f"Depth files read back:   {validation['depth_files']}",
                "Frame 0 round-trip cloud: "
                f"{int(validation['round_trip_points']):,} points",
                "Status: Open3D-compatible dataset validation passed",
            ]
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(Open3DDatasetExporter(root).run())
