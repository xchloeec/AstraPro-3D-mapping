"""Stage 13 entry point: record a timestamped RGB-D dataset for SLAM."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


class SLAMDatasetRecordingSupervisor:
    """Create one session and isolate unstable native camera libraries."""

    def __init__(self, target_frames: int = 200) -> None:
        self.project_root = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = (
            self.project_root / "output" / "rgbd_datasets" / f"dataset_{timestamp}"
        )
        self.worker = self.project_root / "13_rgbd_dataset_worker.py"
        self.target_frames = target_frames

    def run(self) -> int:
        self.output_folder.mkdir(parents=True, exist_ok=False)
        print(f"Stage 13 dataset folder: {self.output_folder}", flush=True)

        for attempt in range(1, 3):
            print(f"Stage 13 camera attempt {attempt}/2...", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.worker),
                    "--output-folder",
                    str(self.output_folder),
                    "--target-frames",
                    str(self.target_frames),
                ],
                cwd=self.project_root,
                check=False,
            )
            if result.returncode == 0 and self._dataset_is_complete():
                print("Stage 13 dataset recording passed.", flush=True)
                return 0

            print(f"Worker ended with exit code {result.returncode}.", flush=True)
            if attempt == 1:
                print("Waiting 5 seconds before one clean retry...", flush=True)
                time.sleep(5.0)

        print(f"Stage 13 failed. Partial data was kept at {self.output_folder}", flush=True)
        return 1

    def _dataset_is_complete(self) -> bool:
        try:
            summary = json.loads(
                (self.output_folder / "dataset_summary.json").read_text(encoding="utf-8")
            )
            return int(summary["accepted_frames"]) >= self.target_frames
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False


if __name__ == "__main__":
    raise SystemExit(SLAMDatasetRecordingSupervisor().run())
