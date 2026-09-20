"""Camera worker for Stage 12.

Run 12_test_rgbd_adapter.py instead of launching this worker directly. The
supervisor can retry if the native OpenNI/UVC combination exits unexpectedly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

import cv2
import numpy as np

from camera import RGBDFrameAdapter


class RGBDAdapterTestApplication:
    """Preview standard frames and save one reproducible evidence package."""

    def __init__(self, evidence_folder: Path, sample_count: int = 100) -> None:
        self.evidence_folder = evidence_folder
        self.sample_count = sample_count

    def run(self) -> int:
        self.evidence_folder.mkdir(parents=True, exist_ok=True)
        timing_ms: list[float] = []
        saved_frame = None
        best_valid_percentage = -1.0
        zero_depth_frame_count = 0

        print("Stage 12B: testing nearest-timestamp RGB-D pairing...", flush=True)
        print("Keep the camera still. Press Q or Escape to stop early.", flush=True)

        try:
            with RGBDFrameAdapter() as adapter:
                # Discard early auto-exposure/depth-startup frames.
                for _ in range(20):
                    adapter.read()

                for _ in range(self.sample_count):
                    frame = adapter.read()
                    timing_ms.append(frame.timestamp_difference_ms)
                    valid_percentage = self._valid_depth_percentage(frame.depth_mm)
                    if valid_percentage == 0.0:
                        zero_depth_frame_count += 1
                    if valid_percentage > best_valid_percentage:
                        saved_frame = frame
                        best_valid_percentage = valid_percentage

                    preview = self._make_preview(frame)
                    cv2.imshow("Stage 12 - standard RGB-D adapter", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
        finally:
            cv2.destroyAllWindows()

        if saved_frame is None:
            raise RuntimeError("No complete RGB-D frame was captured.")
        if best_valid_percentage < 30.0:
            raise RuntimeError(
                "RGB was captured, but no frame contained at least 30% valid "
                f"depth (best was {best_valid_percentage:.2f}%)."
            )

        self._save_evidence(
            saved_frame,
            timing_ms,
            zero_depth_frame_count,
        )
        print(f"Captured {len(timing_ms)} validated RGB-D pairs.", flush=True)
        print(
            "Software RGB-depth time difference: "
            f"median {statistics.median(timing_ms):.2f} ms, "
            f"maximum {max(timing_ms):.2f} ms",
            flush=True,
        )
        print(f"Evidence saved: {self.evidence_folder}", flush=True)
        return 0

    @staticmethod
    def _valid_depth_percentage(depth_mm: np.ndarray) -> float:
        valid = (depth_mm >= 400) & (depth_mm <= 8000)
        return 100.0 * float(np.count_nonzero(valid)) / float(depth_mm.size)

    @staticmethod
    def _make_preview(frame) -> np.ndarray:
        valid = frame.depth_mm > 0
        # Convert the complete 2D image with NumPy. cv2.convertScaleAbs() turns
        # a boolean-selected 1D array into an (N, 1) matrix, which cannot be
        # assigned back through the original 1D boolean mask.
        clipped = np.clip(frame.depth_mm, 400, 5000).astype(np.float32)
        depth_8bit = np.clip(
            (clipped - 400.0) * (255.0 / 4600.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        depth_8bit[~valid] = 0
        depth_colour = cv2.applyColorMap(255 - depth_8bit, cv2.COLORMAP_TURBO)
        preview = np.hstack((frame.colour_bgr, depth_colour))
        label = (
            f"pair {frame.sequence} | dt {frame.timestamp_difference_ms:.1f} ms | "
            f"centre {frame.centre_depth_mm} mm"
        )
        cv2.putText(
            preview,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def _save_evidence(
        self,
        frame,
        timing_ms: list[float],
        zero_depth_frame_count: int,
    ) -> None:
        rgb_path = self.evidence_folder / "rgb.png"
        depth_path = self.evidence_folder / "depth_mm_uint16.png"
        preview_path = self.evidence_folder / "rgbd_preview.png"
        metadata_path = self.evidence_folder / "rgbd_metadata.json"

        cv2.imwrite(str(rgb_path), frame.colour_bgr)
        cv2.imwrite(str(depth_path), frame.depth_mm)
        cv2.imwrite(str(preview_path), self._make_preview(frame))

        # Use the same reliable range as the pass/fail quality gate so report
        # values cannot include invalid zero pixels or >8 m out-of-range data.
        reliable_mask = (frame.depth_mm >= 400) & (frame.depth_mm <= 8000)
        valid_depth = frame.depth_mm[reliable_mask]
        metadata = {
            "purpose": "Stage 12B nearest-timestamp RGB-D pairing validation",
            "colour": {
                "file": rgb_path.name,
                "shape": list(frame.colour_bgr.shape),
                "encoding": "BGR8",
            },
            "depth": {
                "file": depth_path.name,
                "shape": list(frame.depth_mm.shape),
                "encoding": "16UC1",
                "unit": "millimetres",
                "valid_percentage": float(100 * valid_depth.size / frame.depth_mm.size),
                "minimum_valid_mm": int(valid_depth.min()) if valid_depth.size else None,
                "maximum_valid_mm": int(valid_depth.max()) if valid_depth.size else None,
            },
            "intrinsics": {
                "fx": frame.intrinsics.fx,
                "fy": frame.intrinsics.fy,
                "cx": frame.intrinsics.cx,
                "cy": frame.intrinsics.cy,
            },
            "timing": {
                "method": (
                    "depth software timestamp paired to the nearest frame from "
                    "a continuously drained UVC RGB ring buffer"
                ),
                "samples": len(timing_ms),
                "median_difference_ms": statistics.median(timing_ms),
                "mean_difference_ms": statistics.mean(timing_ms),
                "maximum_difference_ms": max(timing_ms),
                "zero_depth_frames": zero_depth_frame_count,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-folder", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        exit_code = RGBDAdapterTestApplication(arguments.evidence_folder).run()
    except Exception:
        # Print the real Python failure, then still bypass unstable native
        # global destructors below. The supervisor will receive a normal 1.
        traceback.print_exc()
        exit_code = 1
    # OpenNI2 and Windows Media Foundation occasionally corrupt the native
    # heap during Python interpreter teardown, after all camera resources and
    # evidence have already been closed/saved. Flush logs and leave directly
    # so native global destructors cannot perform a second cleanup pass.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
