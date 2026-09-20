"""Stage 12 entry point: validate the Python RGB-D bridge before SLAM."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


class RGBDAdapterTestSupervisor:
    """Run native camera work separately and retry one clean process."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evidence_folder = (
            self.project_root
            / "report_evidence"
            / "stage_12_rgbd_adapter"
            / timestamp
        )
        self.worker = self.project_root / "12_rgbd_adapter_worker.py"

    def run(self) -> int:
        for attempt in range(1, 3):
            print(f"Stage 12 camera attempt {attempt}/2...", flush=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(self.worker),
                    "--evidence-folder",
                    str(self.evidence_folder),
                ],
                cwd=self.project_root,
                check=False,
            )
            if completed.returncode == 0:
                print("Stage 12 adapter test passed.", flush=True)
                return 0

            print(
                f"Stage 12 worker ended with exit code {completed.returncode}.",
                flush=True,
            )
            if attempt == 1:
                print("Waiting for Windows to release the camera, then retrying...", flush=True)
                time.sleep(5.0)

        print("Stage 12 failed. Existing evidence files were kept.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(RGBDAdapterTestSupervisor().run())
