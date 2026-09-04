"""Stage 9 supervisor for a resumable Open3D room recording."""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


class RoomRecordingSupervisor:
    """Keep one recording folder across native-camera worker restarts."""

    def __init__(self, maximum_attempts: int = 3) -> None:
        self.maximum_attempts = maximum_attempts
        self.project_root = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = (
            self.project_root / "output" / "room_recordings" / f"room_{timestamp}"
        )
        self.worker = self.project_root / "09_room_recording_worker.py"

    def run(self) -> int:
        print(f"Room recording folder: {self.recording_path}", flush=True)
        for attempt in range(1, self.maximum_attempts + 1):
            print(
                f"Stage 9 camera attempt {attempt}/{self.maximum_attempts}...",
                flush=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.worker),
                    "--output",
                    str(self.recording_path),
                ],
                check=False,
            )
            if result.returncode == 0:
                return 0
            if result.returncode == 2:
                print("Recording was stopped by the user.", flush=True)
                return 2

            print(
                f"Camera worker ended with exit code {result.returncode}.",
                flush=True,
            )
            if attempt < self.maximum_attempts:
                print(
                    "Restarting the camera and resuming the same recording...",
                    flush=True,
                )
                time.sleep(2.0)

        print(
            "Recording did not reach its target. The frames already written "
            f"remain available in {self.recording_path}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(RoomRecordingSupervisor().run())
