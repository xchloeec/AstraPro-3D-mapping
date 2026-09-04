"""Stage 5A supervisor for the dense stride-1 XYZRGB experiment."""

from pathlib import Path
import subprocess
import sys
import time


class DensePointCloudSupervisor:
    """Retry the dense native worker once after an uncatchable DLL crash."""

    def __init__(self, maximum_attempts: int = 2) -> None:
        self.maximum_attempts = maximum_attempts
        self.worker = Path(__file__).resolve().with_name(
            "05a_dense_coloured_point_cloud_worker.py"
        )

    def run(self) -> int:
        for attempt in range(1, self.maximum_attempts + 1):
            print(
                f"Stage 5A worker attempt {attempt}/{self.maximum_attempts}...",
                flush=True,
            )
            result = subprocess.run(
                [sys.executable, str(self.worker)],
                check=False,
            )
            if result.returncode == 0:
                return 0
            print(
                "Stage 5A worker ended unexpectedly with exit code "
                f"{result.returncode}.",
                flush=True,
            )
            if attempt < self.maximum_attempts:
                print("Retrying in a clean Python process...", flush=True)
                time.sleep(1.5)
        return 1


if __name__ == "__main__":
    raise SystemExit(DensePointCloudSupervisor().run())
