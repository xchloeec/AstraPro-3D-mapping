"""Stage 7 supervisor: safely capture a short RGB-D frame sequence."""

from pathlib import Path
import subprocess
import sys
import time


class MultiFrameCaptureSupervisor:
    """Retry the native camera worker in a fresh process after a DLL crash."""

    def __init__(self, maximum_attempts: int = 2) -> None:
        self.maximum_attempts = maximum_attempts
        self.worker = Path(__file__).resolve().with_name(
            "07_multiframe_capture_worker.py"
        )

    def run(self) -> int:
        for attempt in range(1, self.maximum_attempts + 1):
            print(
                f"Stage 7 worker attempt {attempt}/{self.maximum_attempts}...",
                flush=True,
            )
            result = subprocess.run([sys.executable, str(self.worker)], check=False)
            if result.returncode == 0:
                return 0

            print(
                f"Stage 7 worker ended with exit code {result.returncode}.",
                flush=True,
            )
            if attempt < self.maximum_attempts:
                print("Retrying capture in a clean Python process...", flush=True)
                time.sleep(1.5)

        print("Stage 7 capture failed after the automatic retry.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(MultiFrameCaptureSupervisor().run())
