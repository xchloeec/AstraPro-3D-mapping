"""Stage 10 supervisor: capture RGB-D and display a growing 3D map live."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


class LiveMappingSupervisor:
    """Keep camera capture and Open3D mapping in isolated processes."""

    def __init__(self, maximum_camera_attempts: int = 5) -> None:
        self.project_root = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_path = (
            self.project_root
            / "output"
            / "live_mappings"
            / f"live_{timestamp}"
        )
        self.camera_worker = self.project_root / "09_room_recording_worker.py"
        self.mapping_worker = self.project_root / "10_live_mapping_worker.py"
        self.maximum_camera_attempts = maximum_camera_attempts

    def run(self) -> int:
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        finished_flag = self.dataset_path / "_capture_finished.flag"
        stop_flag = self.dataset_path / "_stop_requested.flag"
        print(f"Stage 10 live dataset: {self.dataset_path}", flush=True)

        mapper = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(self.mapping_worker),
                "--dataset",
                str(self.dataset_path),
                "--stride",
                "3",
                "--voxel",
                "0.04",
            ]
        )

        camera_result = 1
        try:
            for attempt in range(1, self.maximum_camera_attempts + 1):
                if stop_flag.exists() or mapper.poll() is not None:
                    print("Live viewer requested the recording to stop.", flush=True)
                    camera_result = 2
                    break

                backend = "MSMF" if attempt <= 2 else "DSHOW"
                print(
                    f"Camera attempt {attempt}/{self.maximum_camera_attempts} "
                    f"using {backend}...",
                    flush=True,
                )
                camera = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(self.camera_worker),
                        "--output",
                        str(self.dataset_path),
                        "--rgb-backend",
                        backend,
                    ]
                )

                while camera.poll() is None:
                    if stop_flag.exists() or mapper.poll() is not None:
                        camera.terminate()
                        camera.wait(timeout=5)
                        camera_result = 2
                        break
                    time.sleep(0.10)
                else:
                    camera_result = int(camera.returncode or 0)

                if camera_result in (0, 2):
                    break
                if attempt < self.maximum_camera_attempts:
                    print(
                        "Camera worker crashed; waiting 8 seconds before resuming.",
                        flush=True,
                    )
                    time.sleep(8.0)
        finally:
            finished_flag.write_text(
                f"camera return code {camera_result}", encoding="utf-8"
            )

        try:
            # Live odometry can accumulate a backlog while capture remains at
            # 10 FPS. Give it enough time to finish and save live_map.ply.
            mapper_result = mapper.wait(timeout=900.0)
        except subprocess.TimeoutExpired:
            mapper.terminate()
            mapper_result = mapper.wait(timeout=10.0)

        output = self.dataset_path / "live_map.ply"
        if output.is_file():
            print(f"\nStage 10 live PLY: {output}", flush=True)
        if camera_result == 0 and mapper_result == 0:
            print("Stage 10 completed successfully.", flush=True)
            return 0
        print(
            f"Stage 10 ended with camera={camera_result}, mapper={mapper_result}.",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(LiveMappingSupervisor().run())
