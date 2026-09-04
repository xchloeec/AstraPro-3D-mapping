"""Record registered Astra RGB-D frames in Open3D reconstruction format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from camera import OpenNIError, RGBCameraError, RGBDCamera
from processing import DepthIntrinsics


class RoomRecordingApplication:
    """Save quality-approved frames directly to a resumable dataset folder."""

    def __init__(
        self,
        output_path: Path,
        target_frames: int = 600,
        capture_interval_seconds: float = 0.1,
        minimum_valid_percentage: float = 30.0,
    ) -> None:
        self.output_path = output_path
        self.colour_path = output_path / "color"
        self.depth_path = output_path / "depth"
        self.quality_log_path = output_path / "frame_quality.csv"
        self.intrinsic_path = output_path / "intrinsic.json"
        self.config_path = output_path / "open3d_config.json"
        self.summary_path = output_path / "recording_summary.txt"
        self.target_frames = target_frames
        self.capture_interval_seconds = capture_interval_seconds
        self.minimum_valid_percentage = minimum_valid_percentage
        self.depth_warning_percentage = 70.0
        self.minimum_orb_matches = 80
        self.minimum_depth_mm = 400
        self.maximum_depth_mm = 5000
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def run(self) -> int:
        self.colour_path.mkdir(parents=True, exist_ok=True)
        self.depth_path.mkdir(parents=True, exist_ok=True)
        next_index = self._next_frame_index()
        if next_index >= self.target_frames:
            print("The recording already contains all requested frames.")
            return 0

        print(
            "Move the camera slowly. Keep 60-80% overlap between views.\n"
            "Press Q or Escape to stop early.",
            flush=True,
        )
        if next_index:
            print(f"Resuming at frame {next_index:05d}.", flush=True)

        existing_rows = self._read_existing_quality()
        accepted_quality = [row[0] for row in existing_rows]
        accepted_overlap = [row[1] for row in existing_rows if row[1] >= 0]
        candidate_count = 0
        last_saved_time = 0.0
        previous_gray = self._load_previous_saved_gray(next_index)

        try:
            with RGBDCamera(rgb_camera_index=0) as camera:
                for _ in range(30):
                    camera.read()

                horizontal_fov, vertical_fov = (
                    camera.depth_camera.get_field_of_view()
                )
                self._write_intrinsic_and_config(
                    horizontal_fov,
                    vertical_fov,
                    width=640,
                    height=480,
                )

                while next_index < self.target_frames:
                    depth_mm, colour_bgr = camera.read()
                    now = time.perf_counter()
                    valid_percentage = self._valid_percentage(depth_mm)
                    candidate_count += 1

                    current_gray = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2GRAY)
                    good_matches, overlap_score = self._measure_rgb_overlap(
                        previous_gray,
                        current_gray,
                    )
                    overlap_ready = (
                        previous_gray is None
                        or good_matches >= self.minimum_orb_matches
                    )

                    preview = self._create_preview(
                        colour_bgr,
                        next_index,
                        valid_percentage,
                        good_matches,
                        overlap_ready,
                    )
                    cv2.imshow("Stage 9 - Open3D room recording", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("User requested an early stop.", flush=True)
                        return 2

                    interval_ready = (
                        now - last_saved_time >= self.capture_interval_seconds
                    )
                    quality_ready = (
                        valid_percentage >= self.minimum_valid_percentage
                    )
                    if not interval_ready or not quality_ready or not overlap_ready:
                        continue

                    self._save_pair(next_index, depth_mm, colour_bgr)
                    self._append_quality(
                        next_index,
                        valid_percentage,
                        good_matches,
                        overlap_score,
                    )
                    accepted_quality.append(valid_percentage)
                    if previous_gray is not None:
                        accepted_overlap.append(good_matches)
                    previous_gray = current_gray
                    next_index += 1
                    last_saved_time = now
                    print(
                        f"Saved frame {next_index:03d}/{self.target_frames}: "
                        f"{valid_percentage:.2f}% valid depth",
                        flush=True,
                    )

            summary = self._write_summary(
                accepted_quality,
                accepted_overlap,
                candidate_count,
            )
            print(summary, flush=True)
            print(f"Room dataset completed: {self.output_path}", flush=True)
            return 0

        except (OpenNIError, RGBCameraError, RuntimeError, ValueError) as error:
            print(f"Stage 9 camera worker failed: {error}", flush=True)
            return 1
        finally:
            cv2.destroyAllWindows()

    def _next_frame_index(self) -> int:
        colour_indices = {path.stem for path in self.colour_path.glob("*.jpg")}
        depth_indices = {path.stem for path in self.depth_path.glob("*.png")}
        complete = sorted(colour_indices & depth_indices)
        if not complete:
            return 0
        expected = [f"{index:05d}" for index in range(len(complete))]
        if complete != expected:
            raise RuntimeError(
                "Existing recording does not contain one continuous frame sequence"
            )
        return len(complete)

    def _save_pair(
        self,
        index: int,
        depth_mm: np.ndarray,
        colour_bgr: np.ndarray,
    ) -> None:
        colour_file = self.colour_path / f"{index:05d}.jpg"
        depth_file = self.depth_path / f"{index:05d}.png"
        if not cv2.imwrite(
            str(colour_file),
            colour_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"Could not write {colour_file}")
        if not cv2.imwrite(str(depth_file), depth_mm):
            colour_file.unlink(missing_ok=True)
            raise RuntimeError(f"Could not write {depth_file}")

    def _valid_percentage(self, depth_mm: np.ndarray) -> float:
        valid = (
            (depth_mm >= self.minimum_depth_mm)
            & (depth_mm <= self.maximum_depth_mm)
        )
        return 100.0 * float(np.count_nonzero(valid)) / float(depth_mm.size)

    def _write_intrinsic_and_config(
        self,
        horizontal_fov: float,
        vertical_fov: float,
        width: int,
        height: int,
    ) -> None:
        intrinsics = DepthIntrinsics.from_field_of_view(
            width,
            height,
            horizontal_fov,
            vertical_fov,
        )
        # Open3D stores this 3x3 matrix in column-major order. Writing the
        # small JSON structure directly keeps the Open3D native DLL completely
        # outside the camera process.
        intrinsic_document = {
            "width": width,
            "height": height,
            "intrinsic_matrix": [
                intrinsics.fx,
                0.0,
                0.0,
                0.0,
                intrinsics.fy,
                0.0,
                intrinsics.cx,
                intrinsics.cy,
                1.0,
            ],
        }
        self.intrinsic_path.write_text(
            json.dumps(intrinsic_document, indent=2),
            encoding="utf-8",
        )

        config = {
            "name": "Astra Pro interior-room RGB-D reconstruction",
            "path_dataset": str(self.output_path.resolve()),
            "path_intrinsic": str(self.intrinsic_path.resolve()),
            "depth_map_type": "redwood",
            "depth_scale": 1000.0,
            "depth_min": 0.4,
            "depth_max": 5.0,
            "voxel_size": 0.025,
            "depth_diff_max": 0.07,
            "n_frames_per_fragment": 100,
            # Compare every 20th frame for local loop constraints. The
            # upstream default of 5 creates about 190 extra pair comparisons
            # per 100-frame fragment and is excessive for this CPU-only test.
            "n_keyframes_per_n_frame": 20,
            "preference_loop_closure_odometry": 0.1,
            "preference_loop_closure_registration": 5.0,
            "tsdf_cubic_size": 6.0,
            "icp_method": "color",
            "global_registration": "ransac",
            # The current Windows Open3D wheel does not expose
            # o3d.utility.set_max_threads(), which the upstream example uses
            # to initialise multiprocessing workers. Sequential fragments are
            # slower but compatible and easier to reproduce on this laptop.
            "python_multi_threading": False,
        }
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _append_quality(
        self,
        index: int,
        valid_percentage: float,
        good_matches: int,
        overlap_score: float,
    ) -> None:
        new_file = not self.quality_log_path.exists()
        with self.quality_log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(
                    [
                        "frame",
                        "capture_time_unix_s",
                        "valid_depth_percentage",
                        "orb_good_matches_to_previous",
                        "orb_overlap_score_percentage",
                    ]
                )
            writer.writerow(
                [
                    index,
                    f"{time.time():.6f}",
                    f"{valid_percentage:.4f}",
                    good_matches if index else -1,
                    f"{overlap_score:.4f}" if index else "-1.0000",
                ]
            )

    def _read_existing_quality(self) -> list[tuple[float, int]]:
        if not self.quality_log_path.is_file():
            return []
        with self.quality_log_path.open("r", newline="", encoding="utf-8") as file:
            return [
                (
                    float(row["valid_depth_percentage"]),
                    int(row.get("orb_good_matches_to_previous", -1)),
                )
                for row in csv.DictReader(file)
            ]

    def _load_previous_saved_gray(self, next_index: int) -> np.ndarray | None:
        """Restore the visual reference when a crashed recording resumes."""
        if next_index == 0:
            return None
        previous_path = self.colour_path / f"{next_index - 1:05d}.jpg"
        previous = cv2.imread(str(previous_path), cv2.IMREAD_GRAYSCALE)
        if previous is None:
            raise RuntimeError(f"Could not resume from {previous_path}")
        return previous

    def _measure_rgb_overlap(
        self,
        previous_gray: np.ndarray | None,
        current_gray: np.ndarray,
    ) -> tuple[int, float]:
        """Count repeatable ORB features shared with the last saved image."""
        if previous_gray is None:
            return self.minimum_orb_matches, 100.0

        previous_keypoints, previous_descriptors = self.orb.detectAndCompute(
            previous_gray, None
        )
        current_keypoints, current_descriptors = self.orb.detectAndCompute(
            current_gray, None
        )
        if previous_descriptors is None or current_descriptors is None:
            return 0, 0.0

        good_matches = []
        for pair in self.matcher.knnMatch(
            previous_descriptors,
            current_descriptors,
            k=2,
        ):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good_matches.append(pair[0])

        denominator = max(
            1,
            min(len(previous_keypoints), len(current_keypoints)),
        )
        score = 100.0 * len(good_matches) / denominator
        return len(good_matches), score

    def _write_summary(
        self,
        quality: list[float],
        overlap_matches: list[int],
        candidates_this_attempt: int,
    ) -> str:
        summary = "\n".join(
            [
                "Stage 9 Open3D room recording",
                "=============================",
                f"Saved RGB-D pairs:          {len(quality)}",
                f"Target RGB-D pairs:         {self.target_frames}",
                f"Capture interval:           {self.capture_interval_seconds:.3f} s",
                f"Approximate accepted rate:  {1/self.capture_interval_seconds:.1f} fps",
                f"Quality threshold:          {self.minimum_valid_percentage:.2f}%",
                f"Depth warning threshold:    {self.depth_warning_percentage:.2f}%",
                f"Minimum ORB matches:        {self.minimum_orb_matches}",
                f"Mean valid depth:           {float(np.mean(quality)):.2f}%",
                f"Minimum valid depth:        {float(np.min(quality)):.2f}%",
                f"Maximum valid depth:        {float(np.max(quality)):.2f}%",
                "Mean ORB matches:            "
                f"{float(np.mean(overlap_matches)):.1f}",
                f"Minimum ORB matches:         {int(np.min(overlap_matches))}",
                f"Candidates in final attempt:{candidates_this_attempt:8d}",
            ]
        )
        self.summary_path.write_text(summary, encoding="utf-8")
        return summary

    def _create_preview(
        self,
        colour_bgr: np.ndarray,
        next_index: int,
        valid_percentage: float,
        good_matches: int,
        overlap_ready: bool,
    ) -> np.ndarray:
        preview = colour_bgr.copy()
        ready = (
            valid_percentage >= self.minimum_valid_percentage
            and overlap_ready
        )
        colour = (60, 220, 60) if ready else (40, 40, 230)
        cv2.putText(
            preview,
            f"saved {next_index}/{self.target_frames}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"valid depth {valid_percentage:.1f}%  |  Q/Esc stop",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"ORB overlap {good_matches} matches (need {self.minimum_orb_matches})",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
        if not overlap_ready:
            cv2.putText(
                preview,
                "RETURN TO LAST VIEW / MOVE SLOWER",
                (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 240),
                2,
                cv2.LINE_AA,
            )
        elif valid_percentage < self.depth_warning_percentage:
            cv2.putText(
                preview,
                "LOW DEPTH QUALITY - CONTINUE SLOWLY",
                (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
        return preview


class WorkerArguments:
    @staticmethod
    def parse() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Record one RGB-D room scan")
        parser.add_argument("--output", required=True, type=Path)
        return parser.parse_args()


if __name__ == "__main__":
    arguments = WorkerArguments.parse()
    raise SystemExit(RoomRecordingApplication(arguments.output).run())
