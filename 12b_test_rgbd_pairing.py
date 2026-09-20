"""Stage 12B: measure improved nearest-timestamp RGB-D pairing."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


class RGBDPairingTestSupervisor:
    """Protect PyCharm from native camera crashes and preserve test evidence."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evidence_folder = (
            self.project_root
            / "report_evidence"
            / "stage_12b_rgbd_pairing"
            / timestamp
        )
        self.worker = self.project_root / "12_rgbd_adapter_worker.py"

    def run(self) -> int:
        for attempt in range(1, 3):
            print(f"Stage 12B camera attempt {attempt}/2...", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.worker),
                    "--evidence-folder",
                    str(self.evidence_folder),
                ],
                cwd=self.project_root,
                check=False,
            )
            if result.returncode == 0 or self._complete_evidence_exists():
                if result.returncode != 0:
                    print(
                        "The native process crashed during shutdown, but the "
                        "complete validated evidence package was already saved.",
                        flush=True,
                    )
                print("Stage 12B pairing test passed.", flush=True)
                return 0

            print(f"Worker ended with exit code {result.returncode}.", flush=True)
            if attempt == 1:
                print("Waiting 5 seconds before one clean retry...", flush=True)
                time.sleep(5.0)

        print("Stage 12B failed. Existing evidence was preserved.", flush=True)
        return 1

    def _complete_evidence_exists(self) -> bool:
        """A saved metadata file proves capture and validation completed."""
        metadata_path = self.evidence_folder / "rgbd_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return int(metadata["timing"]["samples"]) > 0
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False


if __name__ == "__main__":
    raise SystemExit(RGBDPairingTestSupervisor().run())
