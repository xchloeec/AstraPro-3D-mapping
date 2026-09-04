"""Native Stage 7 worker that records RGB-D frames without loading Open3D."""

from pathlib import Path
import time

import cv2
import numpy as np

from camera import OpenNIError, RGBCameraError, RGBDCamera
from reporting import EvidenceManager


class MultiFrameCaptureApplication:
    """Capture several quality-checked RGB-D frames into one NumPy archive."""

    def __init__(
        self,
        project_root: Path,
        requested_frames: int = 5,
        interval_seconds: float = 0.5,
        minimum_valid_percentage: float = 70.0,
    ) -> None:
        self.project_root = project_root
        self.requested_frames = requested_frames
        self.interval_seconds = interval_seconds
        self.minimum_valid_percentage = minimum_valid_percentage
        self.minimum_depth_mm = 400
        self.maximum_depth_mm = 8000
        self.output_path = project_root / "output" / "multiframe_rgbd_capture.npz"
        self.evidence = EvidenceManager(project_root)

    def run(self) -> int:
        accepted_depth: list[np.ndarray] = []
        accepted_colour: list[np.ndarray] = []
        validity: list[float] = []
        attempted_frames = 0
        maximum_attempts = self.requested_frames * 4

        try:
            print("Stage 7: starting multi-frame RGB-D capture...", flush=True)
            with RGBDCamera(rgb_camera_index=0) as camera:
                # Allow depth, exposure and white balance to settle first.
                for _ in range(30):
                    camera.read()

                horizontal_fov, vertical_fov = (
                    camera.depth_camera.get_field_of_view()
                )

                while (
                    len(accepted_depth) < self.requested_frames
                    and attempted_frames < maximum_attempts
                ):
                    depth_mm, colour_bgr = camera.read()
                    attempted_frames += 1
                    valid_percentage = self._valid_percentage(depth_mm)
                    print(
                        f"Candidate {attempted_frames}: "
                        f"{valid_percentage:.2f}% valid depth",
                        flush=True,
                    )

                    if valid_percentage >= self.minimum_valid_percentage:
                        accepted_depth.append(depth_mm.copy())
                        accepted_colour.append(colour_bgr.copy())
                        validity.append(valid_percentage)
                        print(
                            f"Accepted frame {len(accepted_depth)}/"
                            f"{self.requested_frames}.",
                            flush=True,
                        )
                    else:
                        print("Rejected: depth quality is below threshold.", flush=True)

                    if len(accepted_depth) < self.requested_frames:
                        time.sleep(self.interval_seconds)

            print("Camera interfaces closed safely.", flush=True)

            if len(accepted_depth) != self.requested_frames:
                print(
                    f"Only {len(accepted_depth)} acceptable frames were captured; "
                    f"{self.requested_frames} were required. Nothing was overwritten.",
                    flush=True,
                )
                return 1

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.output_path,
                depth_mm=np.stack(accepted_depth),
                colour_bgr=np.stack(accepted_colour),
                valid_percentage=np.asarray(validity, dtype=np.float32),
                horizontal_fov=np.float64(horizontal_fov),
                vertical_fov=np.float64(vertical_fov),
                minimum_depth_mm=np.int32(self.minimum_depth_mm),
                maximum_depth_mm=np.int32(self.maximum_depth_mm),
                interval_seconds=np.float64(self.interval_seconds),
            )

            # A contact sheet gives report evidence without starting Open3D.
            contact_sheet = np.hstack(accepted_colour)
            image_path = self.evidence.save_cv_image(
                "multiframe_capture", "accepted_rgb_sequence", contact_sheet
            )
            report = self._format_report(validity, attempted_frames)
            report_path = self.evidence.save_text(
                "multiframe_capture", "capture_quality_report", report
            )
            print(report, flush=True)
            print(f"RGB-D sequence saved: {self.output_path}", flush=True)
            print(f"RGB sequence evidence saved: {image_path}", flush=True)
            print(f"Capture report saved: {report_path}", flush=True)
            return 0

        except (OpenNIError, RGBCameraError, RuntimeError, ValueError) as error:
            print(f"Stage 7 capture failed: {error}", flush=True)
            return 1
        finally:
            cv2.destroyAllWindows()

    def _valid_percentage(self, depth_mm: np.ndarray) -> float:
        valid = (
            (depth_mm >= self.minimum_depth_mm)
            & (depth_mm <= self.maximum_depth_mm)
        )
        return 100.0 * float(np.count_nonzero(valid)) / float(depth_mm.size)

    def _format_report(
        self, validity: list[float], attempted_frames: int
    ) -> str:
        lines = [
            "Stage 7 multi-frame RGB-D capture",
            "=================================",
            f"Accepted frames:          {len(validity)}",
            f"Candidate frames tested:  {attempted_frames}",
            f"Capture interval:         {self.interval_seconds:.2f} s",
            f"Quality threshold:        {self.minimum_valid_percentage:.2f}%",
        ]
        lines.extend(
            f"Frame {index} valid depth:   {percentage:.2f}%"
            for index, percentage in enumerate(validity, start=1)
        )
        lines.append(
            f"Mean valid depth:         {float(np.mean(validity)):.2f}%"
        )
        return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(MultiFrameCaptureApplication(root).run())
