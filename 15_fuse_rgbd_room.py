"""Stage 15 - fuse a Stage 13 RGB-D dataset using its Stage 14 trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


class RGBDRoomFusion:
    """Back-project registered RGB-D frames into one common world frame."""

    def __init__(
        self,
        dataset: Path,
        voxel_size_m: float = 0.02,
        image_decimation: int = 2,
        show_result: bool = True,
        use_stage16: bool = False,
    ) -> None:
        self.dataset = dataset.resolve()
        self.voxel_size_m = voxel_size_m
        self.image_decimation = image_decimation
        self.show_result = show_result
        self.use_stage16 = use_stage16
        self.result_folder = self.dataset / "stage_15_fusion"
        self.result_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        pairs = self._read_associations()
        poses = self._read_poses()
        intrinsic = self._load_intrinsic()
        usable = [(index, pairs[index], pose) for index, pose in poses.items() if index < len(pairs)]
        if len(usable) < 2:
            raise RuntimeError("Stage 15 needs at least two Stage 14 camera poses.")

        print("Stage 15: trajectory-guided RGB-D room fusion", flush=True)
        print(f"Dataset: {self.dataset}", flush=True)
        print(f"Usable posed frames: {len(usable)}/{len(pairs)}", flush=True)
        print(f"Voxel size: {self.voxel_size_m * 1000:.0f} mm", flush=True)
        started = time.perf_counter()
        room = o3d.geometry.PointCloud()
        raw_points = 0

        for number, (index, pair, pose) in enumerate(usable, start=1):
            rgbd = self._load_rgbd(pair)
            cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            raw_points += len(cloud.points)
            cloud.transform(pose)
            room += cloud

            # Compact regularly so memory does not grow with every raw pixel.
            if number % 10 == 0:
                room = room.voxel_down_sample(self.voxel_size_m)
                print(
                    f"Fused {number:03d}/{len(usable)} frames: "
                    f"{len(room.points):,} voxels",
                    flush=True,
                )

        room = room.voxel_down_sample(self.voxel_size_m)
        if len(room.points) == 0:
            raise RuntimeError("Fusion produced an empty point cloud.")

        # Remove only clearly isolated speckles; retain walls and thin objects.
        room, _ = room.remove_statistical_outlier(nb_neighbors=12, std_ratio=2.5)

        # Open3D camera coordinates are X-right, Y-down, Z-forward.  Export in
        # a conventional robotics/construction frame: X-right, Y-forward,
        # Z-up.  Without this basis change the room looks tipped/upside-down
        # even though the relative reconstruction is correct.
        camera_world_to_z_up = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        room.transform(camera_world_to_z_up)
        ply_path = self.result_folder / "integrated_room.ply"
        if not o3d.io.write_point_cloud(str(ply_path), room, write_ascii=False):
            raise RuntimeError(f"Open3D could not save {ply_path}")

        preview_path = self.result_folder / "integrated_room_preview.png"
        self._save_preview(room, preview_path)
        elapsed = time.perf_counter() - started
        points = np.asarray(room.points)
        extent = points.max(axis=0) - points.min(axis=0)
        summary = {
            "dataset": str(self.dataset),
            "input_rgbd_pairs": len(pairs),
            "fused_frames": len(usable),
            "frames_without_pose": len(pairs) - len(usable),
            "raw_back_projected_points": raw_points,
            "final_points": len(room.points),
            "voxel_size_m": self.voxel_size_m,
            "image_decimation": self.image_decimation,
            "processing_seconds": elapsed,
            "axis_aligned_extent_m": {
                "x": float(extent[0]),
                "y": float(extent[1]),
                "z": float(extent[2]),
            },
            "export_coordinate_frame": "X right, Y forward, Z up",
            "orientation_correction": "X'=X, Y'=Z, Z'=-Y",
            "point_cloud": str(ply_path),
            "note": "Sequential odometry fusion; loop closure is not applied yet.",
        }
        (self.result_folder / "fusion_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print("\nSTAGE 15 COMPLETED", flush=True)
        print(f"Raw projected points: {raw_points:,}", flush=True)
        print(f"Final voxel points:   {len(room.points):,}", flush=True)
        print(
            "Map extent (X/Y/Z):   "
            f"{extent[0]:.2f} / {extent[1]:.2f} / {extent[2]:.2f} m",
            flush=True,
        )
        print(f"Processing time:      {elapsed:.2f} s", flush=True)
        print(f"PLY: {ply_path}", flush=True)
        print(f"Preview: {preview_path}", flush=True)

        if self.show_result:
            print("Close the Open3D window when inspection is finished.", flush=True)
            map_centre = room.get_axis_aligned_bounding_box().get_center()
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=map_centre
            )
            o3d.visualization.draw_geometries(
                [room, coordinate_frame],
                window_name="Stage 15 - trajectory-guided room reconstruction",
                width=1280,
                height=800,
                zoom=0.62,
                # Level front view: camera direction is horizontal while Z
                # remains exactly vertical.  This avoids a visually rolled or
                # tilted room caused only by the initial viewer perspective.
                front=[0.0, -1.0, 0.0],
                lookat=map_centre.tolist(),
                up=[0.0, 0.0, 1.0],
            )
        return 0

    def _read_associations(self) -> list[tuple[Path, Path]]:
        pairs = []
        for line in (self.dataset / "associations.txt").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            _, depth_path, _, rgb_path = line.split()
            pairs.append((self.dataset / depth_path, self.dataset / rgb_path))
        return pairs

    def _read_poses(self) -> dict[int, np.ndarray]:
        optimized = self.dataset / "stage_16_pose_graph" / "trajectory.csv"
        trajectory_path = (
            optimized
            if self.use_stage16 and optimized.is_file()
            else self.dataset / "stage_14_odometry" / "trajectory.csv"
        )
        if not trajectory_path.is_file():
            raise FileNotFoundError("Run Stage 14 successfully before Stage 15.")
        poses = {}
        with trajectory_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rotation = self._quaternion_to_rotation(
                    float(row["qx"]), float(row["qy"]),
                    float(row["qz"]), float(row["qw"]),
                )
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = rotation
                pose[:3, 3] = [float(row["x"]), float(row["y"]), float(row["z"])]
                poses[int(row["index"])] = pose
        return poses

    def _load_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        info = json.loads((self.dataset / "camera_info.json").read_text(encoding="utf-8"))
        scale = float(self.image_decimation)
        return o3d.camera.PinholeCameraIntrinsic(
            int(info["width"] / scale), int(info["height"] / scale),
            float(info["fx"]) / scale, float(info["fy"]) / scale,
            float(info["cx"]) / scale, float(info["cy"]) / scale,
        )

    def _load_rgbd(self, pair: tuple[Path, Path]) -> o3d.geometry.RGBDImage:
        depth_path, rgb_path = pair
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        colour = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if depth is None or colour is None:
            raise RuntimeError(f"Could not read {depth_path} or {rgb_path}")
        width = colour.shape[1] // self.image_decimation
        height = colour.shape[0] // self.image_decimation
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        colour = cv2.resize(
            cv2.cvtColor(colour, cv2.COLOR_BGR2RGB),
            (width, height), interpolation=cv2.INTER_AREA,
        )
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(colour)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1000.0,
            depth_trunc=8.0,
            convert_rgb_to_intensity=False,
        )

    @staticmethod
    def _quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        x, y, z, w = quaternion
        return np.asarray([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])

    @staticmethod
    def _save_preview(cloud: o3d.geometry.PointCloud, path: Path) -> None:
        points = np.asarray(cloud.points)
        colours = np.asarray(cloud.colors)
        stride = max(1, len(points) // 100_000)
        points = points[::stride]
        colours = colours[::stride]
        figure = plt.figure(figsize=(10, 8), facecolor="#101418")
        axis = figure.add_subplot(111, projection="3d", facecolor="#101418")
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=colours, s=0.25)
        axis.set_title("Stage 15 fused RGB-D room", color="white")
        axis.set_xlabel("X (m)", color="white")
        axis.set_ylabel("Y (m)", color="white")
        axis.set_zlabel("Z (m)", color="white")
        axis.tick_params(colors="white")
        figure.tight_layout()
        figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
        plt.close(figure)


def find_latest_dataset(project: Path) -> Path:
    candidates = sorted(
        (project / "output" / "rgbd_datasets").glob("dataset_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "stage_14_odometry" / "trajectory.csv").is_file():
            return candidate
    raise FileNotFoundError("No Stage 13 dataset with Stage 14 trajectory was found.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument(
        "--use-stage16",
        action="store_true",
        help="Use the experimental Stage 16 pose graph instead of Stage 14.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parent
    selected_dataset = arguments.dataset or find_latest_dataset(project_root)
    raise SystemExit(
        RGBDRoomFusion(
            selected_dataset,
            voxel_size_m=arguments.voxel_size,
            show_result=not arguments.no_view,
            use_stage16=arguments.use_stage16,
        ).run()
    )
