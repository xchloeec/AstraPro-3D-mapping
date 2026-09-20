"""Record a quality-controlled, timestamped RGB-D dataset for later SLAM."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import cv2
import numpy as np

from camera import RGBDFrameAdapter


class SLAMDatasetRecorder:
    """Save accepted RGB-D pairs in a simple TUM-style folder structure."""

    def __init__(
        self,
        output_folder: Path,
        target_frames: int = 100,
        save_rate_hz: float = 10.0,
    ) -> None:
        self.output_folder = output_folder
        self.rgb_folder = output_folder / "rgb"
        self.depth_folder = output_folder / "depth"
        self.target_frames = target_frames
        self.minimum_save_interval = 1.0 / save_rate_hz
        self.minimum_valid_depth_percentage = 30.0
        self.maximum_pair_difference_ms = 35.0

    def run(self) -> int:
        self.rgb_folder.mkdir(parents=True, exist_ok=True)
        self.depth_folder.mkdir(parents=True, exist_ok=True)

        accepted_rows: list[dict[str, object]] = []
        quality_rows: list[dict[str, object]] = []
        last_saved_time = 0.0
        last_accepted_time = time.perf_counter()
        recording_started_time = last_accepted_time
        candidate_count = 0

        print("Stage 13: recording a standard RGB-D SLAM dataset...", flush=True)
        print(
            f"Target: {self.target_frames} accepted frames at up to "
            f"{1.0 / self.minimum_save_interval:.1f} Hz.",
            flush=True,
        )
        print(
            "Closed-loop scan: start on a textured object, scan slowly, then "
            "return to the same object and hold for 3-5 seconds.",
            flush=True,
        )
        print("Move slowly with 60-80% overlap. Press Q or Escape to stop.", flush=True)

        startup_finished = threading.Event()
        startup_watchdog = threading.Thread(
            target=self._watch_camera_startup,
            args=(startup_finished, 15.0),
            name="Stage13StartupWatchdog",
            daemon=True,
        )
        startup_watchdog.start()

        with RGBDFrameAdapter() as adapter:
            for _ in range(20):
                adapter.read()
            startup_finished.set()

            while len(accepted_rows) < self.target_frames:
                frame = adapter.read()
                candidate_count += 1
                valid_percentage = self._valid_depth_percentage(frame.depth_mm)
                pair_difference_ms = frame.timestamp_difference_ms
                interval_ready = time.perf_counter() - last_saved_time >= self.minimum_save_interval
                quality_ready = (
                    valid_percentage >= self.minimum_valid_depth_percentage
                    and pair_difference_ms <= self.maximum_pair_difference_ms
                )

                status = "accepted" if interval_ready and quality_ready else "rejected"
                reason = ""
                if not interval_ready:
                    reason = "save_rate_limit"
                elif valid_percentage < self.minimum_valid_depth_percentage:
                    reason = "insufficient_depth"
                elif pair_difference_ms > self.maximum_pair_difference_ms:
                    reason = "rgb_depth_time_difference"

                quality_rows.append(
                    {
                        "candidate": candidate_count,
                        "adapter_sequence": frame.sequence,
                        "depth_timestamp_ns": frame.depth_timestamp_ns,
                        "colour_timestamp_ns": frame.colour_timestamp_ns,
                        "difference_ms": f"{pair_difference_ms:.4f}",
                        "valid_depth_percentage": f"{valid_percentage:.4f}",
                        "status": status,
                        "reason": reason,
                    }
                )

                # Rejected frames were previously silent, which made a bad
                # depth startup look like an infinite hang. Report health once
                # per 30 candidates and persist the diagnostic rows immediately.
                if candidate_count % 30 == 0:
                    print(
                        f"Waiting: candidate {candidate_count}, "
                        f"depth {valid_percentage:.2f}%, "
                        f"dt {pair_difference_ms:.2f} ms, "
                        f"reason {reason or 'ready'}",
                        flush=True,
                    )
                    self._write_quality_log(quality_rows)

                no_accepted_timeout = (
                    not accepted_rows
                    and time.perf_counter() - recording_started_time > 15.0
                )
                stalled_timeout = (
                    bool(accepted_rows)
                    and time.perf_counter() - last_accepted_time > 15.0
                )
                if no_accepted_timeout or stalled_timeout:
                    self._write_quality_log(quality_rows)
                    raise RuntimeError(
                        "No RGB-D pair passed the quality gates for 15 seconds. "
                        f"Latest depth={valid_percentage:.2f}%, "
                        f"dt={pair_difference_ms:.2f} ms, reason={reason}."
                    )

                preview = self._make_preview(
                    frame.colour_bgr,
                    frame.depth_mm,
                    len(accepted_rows),
                    valid_percentage,
                    pair_difference_ms,
                )
                cv2.imshow("Stage 13 - RGB-D SLAM dataset recording", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Recording stopped early by the user.", flush=True)
                    break

                if status != "accepted":
                    continue

                accepted_index = len(accepted_rows)
                row = self._save_frame(accepted_index, frame)
                accepted_rows.append(row)
                last_saved_time = time.perf_counter()
                last_accepted_time = last_saved_time
                print(
                    f"Saved {len(accepted_rows):03d}/{self.target_frames}: "
                    f"depth {valid_percentage:.2f}%, dt {pair_difference_ms:.2f} ms",
                    flush=True,
                )

            if accepted_rows:
                self._write_camera_info(accepted_rows[0]["intrinsics"])

        cv2.destroyAllWindows()
        self._write_dataset_indexes(accepted_rows)
        self._write_quality_log(quality_rows)
        self._write_summary(accepted_rows, quality_rows)

        if not accepted_rows:
            print("No frame passed the Stage 13 quality gates.", flush=True)
            return 1

        print(f"Stage 13 dataset saved: {self.output_folder}", flush=True)
        print(f"Accepted {len(accepted_rows)} of {candidate_count} candidates.", flush=True)
        return 0

    @staticmethod
    def _watch_camera_startup(startup_finished: threading.Event, timeout: float) -> None:
        """Escape a native OpenNI read that blocks forever during startup.

        A normal Python exception cannot interrupt OpenNI's blocking C call.
        The camera work already runs in a disposable child process, so a hard
        worker exit is the safest recovery and lets the supervisor retry.
        """
        if not startup_finished.wait(timeout=timeout):
            print(
                f"Camera startup produced no complete RGB-D frame within {timeout:.0f} "
                "seconds. Ending this worker so the supervisor can retry.",
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(2)

    def _save_frame(self, index: int, frame) -> dict[str, object]:
        # TUM-style timestamp text uses seconds. Nanoseconds remain in the CSV.
        depth_seconds = frame.depth_timestamp_ns / 1_000_000_000
        colour_seconds = frame.colour_timestamp_ns / 1_000_000_000
        stem = f"{index:06d}"
        rgb_relative = Path("rgb") / f"{stem}.png"
        depth_relative = Path("depth") / f"{stem}.png"

        if not cv2.imwrite(str(self.output_folder / rgb_relative), frame.colour_bgr):
            raise RuntimeError(f"Could not save {rgb_relative}.")
        if not cv2.imwrite(str(self.output_folder / depth_relative), frame.depth_mm):
            (self.output_folder / rgb_relative).unlink(missing_ok=True)
            raise RuntimeError(f"Could not save {depth_relative}.")

        return {
            "depth_seconds": depth_seconds,
            "colour_seconds": colour_seconds,
            "rgb_relative": rgb_relative.as_posix(),
            "depth_relative": depth_relative.as_posix(),
            "intrinsics": frame.intrinsics,
        }

    def _write_camera_info(self, intrinsics) -> None:
        document = {
            "width": 640,
            "height": 480,
            "model": "pinhole",
            "depth_scale": 1000.0,
            "depth_unit": "millimetres",
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "K": [
                intrinsics.fx,
                0.0,
                intrinsics.cx,
                0.0,
                intrinsics.fy,
                intrinsics.cy,
                0.0,
                0.0,
                1.0,
            ],
            "note": (
                "Intrinsics are derived from the active OpenNI depth FOV. "
                "RGB is software-paired from the Astra Pro UVC interface."
            ),
        }
        (self.output_folder / "camera_info.json").write_text(
            json.dumps(document, indent=2),
            encoding="utf-8",
        )

    def _write_dataset_indexes(self, rows: list[dict[str, object]]) -> None:
        rgb_lines = ["# timestamp rgb_path"]
        depth_lines = ["# timestamp depth_path"]
        association_lines = ["# depth_timestamp depth_path rgb_timestamp rgb_path"]
        for row in rows:
            depth_time = float(row["depth_seconds"])
            colour_time = float(row["colour_seconds"])
            rgb_path = str(row["rgb_relative"])
            depth_path = str(row["depth_relative"])
            rgb_lines.append(f"{colour_time:.9f} {rgb_path}")
            depth_lines.append(f"{depth_time:.9f} {depth_path}")
            association_lines.append(
                f"{depth_time:.9f} {depth_path} {colour_time:.9f} {rgb_path}"
            )
        (self.output_folder / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
        (self.output_folder / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
        (self.output_folder / "associations.txt").write_text(
            "\n".join(association_lines) + "\n",
            encoding="utf-8",
        )

    def _write_quality_log(self, rows: list[dict[str, object]]) -> None:
        path = self.output_folder / "frame_quality.csv"
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary(
        self,
        accepted_rows: list[dict[str, object]],
        quality_rows: list[dict[str, object]],
    ) -> None:
        differences = [
            float(row["difference_ms"])
            for row in quality_rows
            if row["status"] == "accepted"
        ]
        summary = {
            "accepted_frames": len(accepted_rows),
            "candidate_frames": len(quality_rows),
            "rejected_frames": len(quality_rows) - len(accepted_rows),
            "minimum_valid_depth_percentage": self.minimum_valid_depth_percentage,
            "maximum_pair_difference_ms": self.maximum_pair_difference_ms,
            "median_accepted_difference_ms": (
                float(np.median(differences)) if differences else None
            ),
            "maximum_accepted_difference_ms": max(differences) if differences else None,
        }
        (self.output_folder / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _valid_depth_percentage(depth_mm: np.ndarray) -> float:
        valid = (depth_mm >= 400) & (depth_mm <= 8000)
        return 100.0 * float(np.count_nonzero(valid)) / float(depth_mm.size)

    @staticmethod
    def _make_preview(
        colour_bgr: np.ndarray,
        depth_mm: np.ndarray,
        accepted: int,
        valid_percentage: float,
        difference_ms: float,
    ) -> np.ndarray:
        valid = (depth_mm >= 400) & (depth_mm <= 8000)
        clipped = np.clip(depth_mm, 400, 8000).astype(np.float32)
        depth_8bit = np.clip((clipped - 400.0) * (255.0 / 7600.0), 0, 255).astype(np.uint8)
        depth_8bit[~valid] = 0
        depth_colour = cv2.applyColorMap(255 - depth_8bit, cv2.COLORMAP_TURBO)
        preview = np.hstack((colour_bgr, depth_colour))
        cv2.putText(
            preview,
            f"saved {accepted}/{100} | depth {valid_percentage:.1f}% | dt {difference_ms:.1f} ms",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--target-frames", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        code = SLAMDatasetRecorder(
            arguments.output_folder,
            target_frames=arguments.target_frames,
        ).run()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
