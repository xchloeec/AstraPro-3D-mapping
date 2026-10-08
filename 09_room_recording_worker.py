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
from camera.rgbd_adapter import RGBDFrameAdapter
from processing import DepthIntrinsics
from processing.photometric import light_metrics


class RoomRecordingApplication:
    """Save quality-approved frames directly to a resumable dataset folder."""

    def __init__(
        self,
        output_path: Path,
        target_frames: int = 600,
        capture_interval_seconds: float = 0.1,
        minimum_valid_percentage: float = 30.0,
        rgb_backend: str = "MSMF",
        manual_control: bool = False,
    ) -> None:
        self.output_path = output_path
        self.colour_path = output_path / "color"
        self.depth_path = output_path / "depth"
        self.quality_log_path = output_path / "frame_quality.csv"
        self.intrinsic_path = output_path / "intrinsic.json"
        self.config_path = output_path / "open3d_config.json"
        self.summary_path = output_path / "recording_summary.txt"
        self.target_frames = target_frames
        self.manual_control = manual_control
        self.pause_flag = output_path / "_paused.flag"
        self.stop_flag = output_path / "_stop_requested.flag"
        self.capture_interval_seconds = capture_interval_seconds
        self.minimum_valid_percentage = minimum_valid_percentage
        self.rgb_backend = rgb_backend
        self.depth_warning_percentage = 70.0
        self.minimum_depth_mm = 400
        self.maximum_depth_mm = 5000

    def run(self) -> int:
        self.colour_path.mkdir(parents=True, exist_ok=True)
        self.depth_path.mkdir(parents=True, exist_ok=True)
        next_index = self._next_frame_index()
        if self.target_frames > 0 and next_index >= self.target_frames:
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
        candidate_count = 0
        last_saved_time = 0.0

        try:
            backend = {
                "MSMF": cv2.CAP_MSMF,
                "DSHOW": cv2.CAP_DSHOW,
            }[self.rgb_backend]
            print(f"Requested RGB backend: {self.rgb_backend}", flush=True)
            with RGBDFrameAdapter(RGBDCamera(
                rgb_camera_index=0,
                rgb_backends=(backend,),
            )) as adapter:
                for _ in range(30):
                    adapter.read()

                horizontal_fov, vertical_fov = (
                    adapter.camera.depth_camera.get_field_of_view()
                )
                self._write_intrinsic_and_config(
                    horizontal_fov,
                    vertical_fov,
                    width=640,
                    height=480,
                )
                (self.output_path / "capture_metadata.json").write_text(json.dumps({
                    "pairing": "nearest software timestamp", "maximum_pair_offset_ms": 25.,
                    "hardware_synchronised": False,
                    "depth_to_colour_registration_enabled": adapter.camera.registration_enabled,
                    "intrinsic_source": "OpenNI depth field-of-view estimate",
                    "spatial_calibration_measured": False,
                },indent=2),encoding="utf-8")

                while self.target_frames == 0 or next_index < self.target_frames:
                    if self.stop_flag.exists():
                        break
                    frame = adapter.read()
                    depth_mm, colour_bgr = frame.depth_mm, frame.colour_bgr
                    self.pair_offset_ms = frame.timestamp_difference_ms
                    now = time.perf_counter()
                    valid_percentage = self._valid_percentage(depth_mm)
                    candidate_count += 1
                    lighting = light_metrics(colour_bgr)

                    preview = self._create_preview(
                        colour_bgr,
                        next_index,
                        valid_percentage,
                    )
                    if lighting["median_luma"] < 45 or lighting["dark_fraction"] > .5:
                        cv2.putText(preview,"LOW LIGHT: add light / hold camera steady",
                                    (15,215),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,180,255),2,cv2.LINE_AA)
                    try:
                        status = json.loads((self.output_path / "tracking_status.json").read_text())
                    except (OSError, ValueError):
                        status = None
                    if status is not None:
                        tracked = status["tracked"]
                        text = ("TRACKING OK" if tracked else "TRACKING LOST - return to reference")
                        cv2.putText(preview, f"{text} | processed {status['processed_frame']}",
                                    (15,155),cv2.FONT_HERSHEY_SIMPLEX,.5,
                                    (60,220,60) if tracked else (40,40,230),2,cv2.LINE_AA)
                        if not tracked and status.get("last_tracked_frame") is not None:
                            reference = cv2.imread(str(self.colour_path / f"{status['last_tracked_frame']:05d}.jpg"))
                            if reference is not None:
                                cv2.imshow("Return to this tracked view",cv2.resize(reference,(320,240)))
                        elif tracked:
                            try:
                                cv2.destroyWindow("Return to this tracked view")
                            except cv2.error:
                                pass
                    cv2.imshow("Stage 9 - Open3D room recording", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("User requested an early stop.", flush=True)
                        self.stop_flag.write_text("user finished", encoding="utf-8")
                        break
                    if self.manual_control and key == ord(" "):
                        if self.pause_flag.exists():
                            self.pause_flag.unlink()
                            print("Scanning resumed; keep overlap with the last view.", flush=True)
                        else:
                            self.pause_flag.write_text("paused", encoding="utf-8")
                            print("Paused; return to this view before resuming.", flush=True)
                    if self.manual_control and self.pause_flag.exists():
                        continue

                    interval_ready = (
                        now - last_saved_time >= self.capture_interval_seconds
                    )
                    quality_ready = (
                        valid_percentage >= self.minimum_valid_percentage
                        and frame.timestamp_difference_ms <= 25.0
                    )
                    if not interval_ready or not quality_ready:
                        continue

                    self._save_pair(next_index, depth_mm, colour_bgr)
                    self._append_quality(
                        next_index,
                        valid_percentage,
                        -1,
                        -1.0,
                    )
                    with (self.output_path / "pair_timing.csv").open("a", encoding="utf-8") as timing:
                        if next_index == 0:
                            timing.write("frame,depth_timestamp_ns,colour_timestamp_ns,offset_ms\n")
                        timing.write(f"{next_index},{frame.depth_timestamp_ns},{frame.colour_timestamp_ns},{frame.timestamp_difference_ms:.3f}\n")
                    accepted_quality.append(valid_percentage)
                    light_log = self.output_path / "lighting_quality.csv"
                    new_light_log = not light_log.exists()
                    with light_log.open("a",newline="",encoding="utf-8") as file:
                        writer = csv.DictWriter(file,fieldnames=["frame",*lighting.keys()])
                        if new_light_log:
                            writer.writeheader()
                        writer.writerow({"frame":next_index,**lighting})
                    next_index += 1
                    last_saved_time = now
                    print(
                        f"Saved frame {next_index:03d}/{self.target_frames or 'unlimited'}: "
                        f"{valid_percentage:.2f}% valid depth",
                        flush=True,
                    )

            summary = self._write_summary(
                accepted_quality,
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
        complete = sorted(colour_indices & depth_indices, key=int)
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
        colour_ok, colour_data = cv2.imencode(".jpg",colour_bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
        depth_ok, depth_data = cv2.imencode(".png",depth_mm)
        if not colour_ok:
            raise RuntimeError(f"Could not write {colour_file}")
        if not depth_ok:
            raise RuntimeError(f"Could not write {depth_file}")
        # Publish only complete encoded images. A live reader must not see a
        # half-written PNG merely because its final filename already exists.
        colour_pending = colour_file.with_suffix(".jpg.part")
        depth_pending = depth_file.with_suffix(".png.part")
        try:
            colour_pending.write_bytes(colour_data.tobytes())
            depth_pending.write_bytes(depth_data.tobytes())
            depth_pending.replace(depth_file)
            colour_pending.replace(colour_file)
        finally:
            colour_pending.unlink(missing_ok=True)
            depth_pending.unlink(missing_ok=True)

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

    def _write_summary(
        self,
        quality: list[float],
        candidates_this_attempt: int,
    ) -> str:
        summary = "\n".join(
            [
                "Stage 9 Open3D room recording",
                "=============================",
                f"Saved RGB-D pairs:          {len(quality)}",
                f"Target RGB-D pairs:         {self.target_frames or 'unlimited'}",
                f"Capture interval:           {self.capture_interval_seconds:.3f} s",
                f"Approximate accepted rate:  {1/self.capture_interval_seconds:.1f} fps",
                f"Quality threshold:          {self.minimum_valid_percentage:.2f}%",
                f"Depth warning threshold:    {self.depth_warning_percentage:.2f}%",
                "Online ORB overlap check:    disabled for camera stability",
                "RGB-D pairing:              nearest software timestamp; accepted offset <=25 ms",
                f"Mean valid depth:           {float(np.mean(quality)) if quality else 0.0:.2f}%",
                f"Minimum valid depth:        {float(np.min(quality)) if quality else 0.0:.2f}%",
                f"Maximum valid depth:        {float(np.max(quality)) if quality else 0.0:.2f}%",
                "ORB overlap:                 validate offline after recording",
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
    ) -> np.ndarray:
        preview = colour_bgr.copy()
        ready = (
            valid_percentage >= self.minimum_valid_percentage
            and getattr(self,"pair_offset_ms",0.) <= 25.
        )
        colour = (60, 220, 60) if ready else (40, 40, 230)
        cv2.putText(
            preview,
            f"saved {next_index}/{self.target_frames or 'unlimited'}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"depth {valid_percentage:.1f}% | pair {getattr(self, 'pair_offset_ms', 0):.1f} ms | Q/Esc finish",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            ("PAUSED - Space: play; return to last view" if self.pause_flag.exists()
             else "SCANNING - Space: pause" if self.manual_control
             else "Move slowly; keep 60-80% visual overlap"),
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
        if valid_percentage < self.depth_warning_percentage:
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
        if getattr(self,"pair_offset_ms",0.) > 25.:
            cv2.putText(preview,"RGB-D TIMING TOO FAR APART - FRAME SKIPPED",
                        (15,185),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,40,230),2,cv2.LINE_AA)
        return preview


class WorkerArguments:
    @staticmethod
    def parse() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Record one RGB-D room scan")
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument(
            "--rgb-backend",
            choices=("MSMF", "DSHOW"),
            default="MSMF",
            help="Windows UVC backend used for the Astra RGB interface",
        )
        parser.add_argument("--frames", type=int, default=600, help="0 means unlimited")
        parser.add_argument("--manual-control", action="store_true")
        args = parser.parse_args()
        if args.frames < 0:
            parser.error("--frames must be nonnegative")
        return args


if __name__ == "__main__":
    arguments = WorkerArguments.parse()
    application = RoomRecordingApplication(
        arguments.output,
        rgb_backend=arguments.rgb_backend,
        target_frames=arguments.frames,
        manual_control=arguments.manual_control,
    )
    raise SystemExit(application.run())
