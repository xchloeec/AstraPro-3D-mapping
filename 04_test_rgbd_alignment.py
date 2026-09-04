"""Stage 4 supervisor for the native RGB-D alignment preview."""

from pathlib import Path
import subprocess
import sys
import time


class RGBDAlignmentSupervisor:
    """Monitor and restart the alignment worker after a native crash."""

    def __init__(self, maximum_attempts: int = 2) -> None:
        self.maximum_attempts = maximum_attempts
        self.worker = Path(__file__).resolve().with_name(
            "04_rgbd_alignment_worker.py"
        )

    def run(self) -> int:
        for attempt in range(1, self.maximum_attempts + 1):
            print(
                f"Stage 4 worker attempt {attempt}/{self.maximum_attempts}...",
                flush=True,
            )
            result = subprocess.run(
                [sys.executable, str(self.worker)],
                check=False,
            )
            if result.returncode == 0:
                return 0

            print(
                "Stage 4 worker ended unexpectedly with exit code "
                f"{result.returncode}.",
                flush=True,
            )
            if attempt < self.maximum_attempts:
                print(
                    "Restarting RGB-D alignment in a clean Python process...",
                    flush=True,
                )
                time.sleep(1.5)

        print(
            "Stage 4 failed after the automatic retry. Close Camera, NiViewer "
            "and other camera programs, reconnect the Astra Pro, then run "
            "this file again.",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    supervisor = RGBDAlignmentSupervisor()
    raise SystemExit(supervisor.run())
