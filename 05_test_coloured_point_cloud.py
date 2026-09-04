"""Stage 5 supervisor: run the native camera/Open3D work safely."""

from pathlib import Path
import subprocess
import sys
import time


class ColouredPointCloudSupervisor:
    """Restart Stage 5 in a clean process after an uncatchable native crash.

    Windows error 0xC0000374 terminates the Python process before normal
    exception handling can run. A parent process is therefore required to
    observe the worker's exit code and safely retry it.
    """

    def __init__(self, maximum_attempts: int = 2) -> None:
        self.maximum_attempts = maximum_attempts
        self.worker = Path(__file__).resolve().with_name(
            "05_coloured_point_cloud_worker.py"
        )

    def run(self) -> int:
        for attempt in range(1, self.maximum_attempts + 1):
            print(
                f"Stage 5 worker attempt {attempt}/{self.maximum_attempts}...",
                flush=True,
            )
            result = subprocess.run(
                [sys.executable, str(self.worker)],
                check=False,
            )
            if result.returncode == 0:
                return 0

            print(
                "Stage 5 worker ended unexpectedly with exit code "
                f"{result.returncode}.",
                flush=True,
            )
            if attempt < self.maximum_attempts:
                print(
                    "Restarting the camera work in a clean Python process...",
                    flush=True,
                )
                time.sleep(1.5)

        print(
            "Stage 5 failed after the automatic retry. Close all camera "
            "programs, reconnect the Astra Pro and run this file again.",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    supervisor = ColouredPointCloudSupervisor()
    raise SystemExit(supervisor.run())
