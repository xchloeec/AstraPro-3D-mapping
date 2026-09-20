"""Stage 14 - estimate an offline camera trajectory from a Stage 13 dataset.

This stage deliberately does not fuse a room point cloud.  It answers the more
fundamental question first: can consecutive validated RGB-D frames produce a
plausible six-degree-of-freedom camera trajectory?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


class OfflineRGBDOdometryTest:
    """Run fast, decimated RGB-D odometry and report every tracking decision."""

    def __init__(self, dataset_folder: Path, image_decimation: int = 2) -> None:
        self.dataset_folder = dataset_folder.resolve()
        self.image_decimation = image_decimation
        self.results_folder = self.dataset_folder / "stage_14_odometry"
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.maximum_translation_m = 0.30
        self.maximum_rotation_degrees = 25.0
        self.depth_truncation_m = 8.0

    def run(self) -> int:
        pairs = self._read_associations()
        intrinsic = self._load_intrinsic()
        print("Stage 14: offline RGB-D visual odometry", flush=True)
        print(f"Dataset: {self.dataset_folder}", flush=True)
        print(f"Frame pairs: {len(pairs)}", flush=True)
        print(
            f"Processing resolution: {intrinsic.width}x{intrinsic.height} "
            f"(decimation {self.image_decimation})",
            flush=True,
        )

        started = time.perf_counter()
        world_from_camera = np.eye(4, dtype=np.float64)
        previous_rgbd = None
        previous_pair = None
        previous_index = None
        trajectory: list[dict[str, object]] = []
        steps: list[dict[str, object]] = []
        path_length_m = 0.0
        tracking_losses = 0
        feature_recovery_attempts = 0
        feature_recovery_successes = 0

        for index, pair in enumerate(pairs):
            current_rgbd = self._load_rgbd(pair)
            if previous_rgbd is None:
                success = True
                translation_m = 0.0
                rotation_degrees = 0.0
                information_trace = 0.0
                reason = "first_frame"
                method = "initial"
            else:
                success, source_to_target, information = (
                    o3d.pipelines.odometry.compute_rgbd_odometry(
                        previous_rgbd,
                        current_rgbd,
                        intrinsic,
                        np.eye(4),
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                        o3d.pipelines.odometry.OdometryOption(depth_diff_max=0.07),
                    )
                )
                translation_m = float(np.linalg.norm(source_to_target[:3, 3]))
                rotation_degrees = self._rotation_angle_degrees(
                    source_to_target[:3, :3]
                )
                information_trace = float(np.trace(information))
                reason = "tracked" if success else "odometry_failed"
                method = "open3d_rgbd"

                if success and translation_m > self.maximum_translation_m:
                    success = False
                    reason = "implausible_translation"
                if success and rotation_degrees > self.maximum_rotation_degrees:
                    success = False
                    reason = "implausible_rotation"

                if not success:
                    feature_recovery_attempts += 1
                    (
                        recovered,
                        recovered_transform,
                        recovery_details,
                    ) = self._feature_recovery(previous_pair, pair, intrinsic)
                    if recovered:
                        source_to_target = recovered_transform
                        translation_m = float(
                            np.linalg.norm(source_to_target[:3, 3])
                        )
                        rotation_degrees = self._rotation_angle_degrees(
                            source_to_target[:3, :3]
                        )
                        success = (
                            translation_m <= self.maximum_translation_m
                            and rotation_degrees <= self.maximum_rotation_degrees
                        )
                        if success:
                            feature_recovery_successes += 1
                            reason = "feature_recovery"
                            method = "orb_depth_pnp"
                            information_trace = float(
                                recovery_details["inliers"]
                            )
                        else:
                            reason = "feature_recovery_implausible"

                if success:
                    world_from_camera = (
                        world_from_camera @ np.linalg.inv(source_to_target)
                    )
                    path_length_m += translation_m
                else:
                    tracking_losses += 1

                steps.append(
                    {
                        "source_index": previous_index,
                        "target_index": index,
                        "success": success,
                        "reason": reason,
                        "method": method,
                        "translation_m": translation_m,
                        "rotation_degrees": rotation_degrees,
                        "information_trace": information_trace,
                    }
                )

            # Always rebase to the newest image.  If one pair is untrackable,
            # its pose is held once instead of comparing every future frame
            # against an increasingly old reference image.
            previous_rgbd = current_rgbd
            previous_pair = pair
            previous_index = index
            if success:
                trajectory.append(
                    self._trajectory_row(index, pair[0], world_from_camera)
                )

            if index == 0 or (index + 1) % 10 == 0 or not success:
                print(
                    f"Frame {index + 1:03d}/{len(pairs)}: {reason}, "
                    f"step {translation_m:.3f} m, {rotation_degrees:.2f} deg, "
                    f"losses {tracking_losses}",
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        net_displacement_m = float(
            np.linalg.norm(world_from_camera[:3, 3])
        )
        self._write_steps(steps)
        self._write_trajectory(trajectory)
        self._plot_trajectory(trajectory)

        successful_steps = sum(bool(row["success"]) for row in steps)
        summary = {
            "dataset": str(self.dataset_folder),
            "input_frames": len(pairs),
            "tracked_poses": len(trajectory),
            "successful_steps": successful_steps,
            "tracking_losses": tracking_losses,
            "feature_recovery_attempts": feature_recovery_attempts,
            "feature_recovery_successes": feature_recovery_successes,
            "tracking_success_percentage": (
                100.0 * successful_steps / len(steps) if steps else 100.0
            ),
            "path_length_m": path_length_m,
            "net_displacement_m": net_displacement_m,
            "processing_seconds": elapsed,
            "image_decimation": self.image_decimation,
            "processing_resolution": [intrinsic.width, intrinsic.height],
            "maximum_allowed_translation_m": self.maximum_translation_m,
            "maximum_allowed_rotation_degrees": self.maximum_rotation_degrees,
            "note": (
                "This is sequential visual odometry only. No loop closure or "
                "global pose-graph optimisation has been applied yet."
            ),
        }
        (self.results_folder / "odometry_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print("\nSTAGE 14 COMPLETED", flush=True)
        print(f"Successful steps: {successful_steps}/{len(steps)}", flush=True)
        print(f"Tracking losses:  {tracking_losses}", flush=True)
        print(
            f"Feature recovery: {feature_recovery_successes}/"
            f"{feature_recovery_attempts}",
            flush=True,
        )
        print(f"Path length:       {path_length_m:.3f} m", flush=True)
        print(f"Net displacement:  {net_displacement_m:.3f} m", flush=True)
        print(f"Processing time:   {elapsed:.2f} s", flush=True)
        print(f"Results: {self.results_folder}", flush=True)
        return 0 if successful_steps >= max(1, int(0.8 * len(steps))) else 1

    def _feature_recovery(
        self,
        source_pair: tuple[float, Path, float, Path],
        target_pair: tuple[float, Path, float, Path],
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
    ) -> tuple[bool, np.ndarray, dict[str, int]]:
        """Estimate source-camera to target-camera motion using 3D-2D PnP."""
        _, source_depth_path, _, source_rgb_path = source_pair
        _, _, _, target_rgb_path = target_pair
        source_depth = cv2.imread(str(source_depth_path), cv2.IMREAD_UNCHANGED)
        source_bgr = cv2.imread(str(source_rgb_path), cv2.IMREAD_COLOR)
        target_bgr = cv2.imread(str(target_rgb_path), cv2.IMREAD_COLOR)
        if source_depth is None or source_bgr is None or target_bgr is None:
            return False, np.eye(4), {"matches": 0, "inliers": 0}

        width, height = intrinsic.width, intrinsic.height
        source_depth = cv2.resize(
            source_depth, (width, height), interpolation=cv2.INTER_NEAREST
        )
        source_gray = cv2.resize(
            cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        target_gray = cv2.resize(
            cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

        detector = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
        source_keypoints, source_descriptors = detector.detectAndCompute(
            source_gray, None
        )
        target_keypoints, target_descriptors = detector.detectAndCompute(
            target_gray, None
        )
        if source_descriptors is None or target_descriptors is None:
            return False, np.eye(4), {"matches": 0, "inliers": 0}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        candidate_matches = matcher.knnMatch(
            source_descriptors, target_descriptors, k=2
        )
        matches = [
            first
            for first, second in candidate_matches
            if first.distance < 0.75 * second.distance
        ]

        object_points = []
        image_points = []
        fx, fy = intrinsic.intrinsic_matrix[0, 0], intrinsic.intrinsic_matrix[1, 1]
        cx, cy = intrinsic.intrinsic_matrix[0, 2], intrinsic.intrinsic_matrix[1, 2]
        for match in matches:
            u, v = source_keypoints[match.queryIdx].pt
            pixel_x = int(round(u))
            pixel_y = int(round(v))
            if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                continue
            z = float(source_depth[pixel_y, pixel_x]) / 1000.0
            if not (0.4 <= z <= self.depth_truncation_m):
                continue
            object_points.append(
                ((u - cx) * z / fx, (v - cy) * z / fy, z)
            )
            image_points.append(target_keypoints[match.trainIdx].pt)

        details = {"matches": len(object_points), "inliers": 0}
        if len(object_points) < 20:
            return False, np.eye(4), details

        camera_matrix = np.asarray(intrinsic.intrinsic_matrix)
        solved, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            camera_matrix,
            None,
            iterationsCount=500,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inlier_count = 0 if inliers is None else len(inliers)
        details["inliers"] = inlier_count
        if not solved or inlier_count < 15 or inlier_count / len(object_points) < 0.25:
            return False, np.eye(4), details

        rotation, _ = cv2.Rodrigues(rotation_vector)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation.reshape(3)
        return True, transform, details

    def _read_associations(self) -> list[tuple[float, Path, float, Path]]:
        path = self.dataset_folder / "associations.txt"
        pairs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            depth_time, depth_path, rgb_time, rgb_path = line.split()
            pairs.append(
                (
                    float(depth_time),
                    self.dataset_folder / depth_path,
                    float(rgb_time),
                    self.dataset_folder / rgb_path,
                )
            )
        if len(pairs) < 2:
            raise RuntimeError("Stage 14 requires at least two associated frames.")
        return pairs

    def _load_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        document = json.loads(
            (self.dataset_folder / "camera_info.json").read_text(encoding="utf-8")
        )
        scale = float(self.image_decimation)
        return o3d.camera.PinholeCameraIntrinsic(
            int(document["width"] / self.image_decimation),
            int(document["height"] / self.image_decimation),
            float(document["fx"]) / scale,
            float(document["fy"]) / scale,
            float(document["cx"]) / scale,
            float(document["cy"]) / scale,
        )

    def _load_rgbd(
        self, pair: tuple[float, Path, float, Path]
    ) -> o3d.geometry.RGBDImage:
        _, depth_path, _, rgb_path = pair
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        colour_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if depth is None or colour_bgr is None:
            raise RuntimeError(f"Could not read pair: {rgb_path}, {depth_path}")

        width = colour_bgr.shape[1] // self.image_decimation
        height = colour_bgr.shape[0] // self.image_decimation
        colour_rgb = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2RGB)
        colour_rgb = cv2.resize(
            colour_rgb, (width, height), interpolation=cv2.INTER_AREA
        )
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(colour_rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1000.0,
            depth_trunc=self.depth_truncation_m,
            convert_rgb_to_intensity=False,
        )

    @staticmethod
    def _rotation_angle_degrees(rotation: np.ndarray) -> float:
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _trajectory_row(
        index: int, timestamp: float, pose: np.ndarray
    ) -> dict[str, object]:
        qx, qy, qz, qw = OfflineRGBDOdometryTest._rotation_to_quaternion(
            pose[:3, :3]
        )
        return {
            "index": index,
            "timestamp": timestamp,
            "x": float(pose[0, 3]),
            "y": float(pose[1, 3]),
            "z": float(pose[2, 3]),
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
        }

    @staticmethod
    def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
        """Convert a 3x3 rotation matrix to x,y,z,w quaternion form."""
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif axis == 1:
                scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
        return float(qx), float(qy), float(qz), float(qw)

    def _write_steps(self, rows: list[dict[str, object]]) -> None:
        with (self.results_folder / "odometry_steps.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_trajectory(self, rows: list[dict[str, object]]) -> None:
        with (self.results_folder / "trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        tum_lines = [
            "# timestamp tx ty tz qx qy qz qw",
            *[
                "{timestamp:.9f} {x:.9f} {y:.9f} {z:.9f} "
                "{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}".format(**row)
                for row in rows
            ],
        ]
        (self.results_folder / "trajectory_tum.txt").write_text(
            "\n".join(tum_lines) + "\n", encoding="utf-8"
        )

    def _plot_trajectory(self, rows: list[dict[str, object]]) -> None:
        x = [float(row["x"]) for row in rows]
        y = [float(row["y"]) for row in rows]
        z = [float(row["z"]) for row in rows]
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(x, y, z, color="#1473E6", marker=".", linewidth=1.5)
        axis.scatter([x[0]], [y[0]], [z[0]], color="green", label="start")
        axis.scatter([x[-1]], [y[-1]], [z[-1]], color="red", label="end")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.set_title("Stage 14 sequential RGB-D odometry trajectory")
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.results_folder / "trajectory_preview.png", dpi=160)
        plt.close(figure)


def find_latest_complete_dataset(project_root: Path) -> Path:
    datasets_root = project_root / "output" / "rgbd_datasets"
    candidates = sorted(
        (path for path in datasets_root.glob("dataset_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        summary_path = candidate / "dataset_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("accepted_frames", 0)) >= 2:
            return candidate
    raise FileNotFoundError("No completed Stage 13 RGB-D dataset was found.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    project = Path(__file__).resolve().parent
    dataset = arguments.dataset or find_latest_complete_dataset(project)
    raise SystemExit(OfflineRGBDOdometryTest(dataset).run())
