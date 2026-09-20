"""Safely turn a completed Stage 9 RGB-D recording into an Open3D PLY.

The camera worker is deliberately finished before this module is used.  That
keeps Open3D's native libraries out of the process that owns OpenNI/OpenCV
camera handles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


class RoomReconstructionPipeline:
    """Run Open3D's four reconstruction phases with safety checks."""

    def __init__(self, dataset_path: Path, strict_safety: bool = False) -> None:
        self.dataset_path = dataset_path.resolve()
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.dataset_path / "open3d_config.json"
        self.geometric_config_path = (
            self.dataset_path / "open3d_config_point_to_plane.json"
        )
        self.fragment_path = self.dataset_path / "fragments"
        self.scene_path = self.dataset_path / "scene"
        self.report_path = self.dataset_path / "reconstruction_safety_report.txt"
        self.strict_safety = strict_safety
        self.open3d_runner = (
            self.project_root
            / "external"
            / "Open3D"
            / "examples"
            / "python"
            / "reconstruction_system"
            / "run_system.py"
        )

    def run(self) -> int:
        pair_count = self._validate_dataset()
        expected_fragments = math.ceil(pair_count / 100)
        print("\nCamera recording has closed safely.", flush=True)
        print(
            f"Automatic reconstruction starting for {pair_count} RGB-D pairs.",
            flush=True,
        )

        if not self._fragments_are_present(expected_fragments):
            if not self._run_open3d_phase("make", self.config_path):
                return self._stop("Open3D could not create the local fragments.")
        else:
            print("Existing local fragments found; reusing them.", flush=True)

        safe, audit = self._audit_local_trajectory(expected_fragments)
        self.report_path.write_text(audit, encoding="utf-8")
        print(audit, flush=True)
        if not safe:
            warning = (
                "Unsafe camera-pose jumps were found. Reconstruction will "
                "continue automatically, but the PLY may be stretched, "
                "duplicated or broken."
            )
            if self.strict_safety:
                return self._stop(warning)
            print(f"\nRECONSTRUCTION QUALITY WARNING: {warning}", flush=True)

        if not self._run_open3d_phase("register", self.config_path):
            return self._stop("Fragment registration failed.")

        # Coloured ICP gives useful texture alignment, but some Astra room
        # fragments have too little colour correspondence at the finest scale.
        # Retry with geometric point-to-plane ICP instead of losing the dataset.
        refine_config = self.config_path
        if not self._run_open3d_phase("refine", refine_config):
            print(
                "Coloured ICP refinement failed; retrying with point-to-plane ICP.",
                flush=True,
            )
            self._write_geometric_config()
            refine_config = self.geometric_config_path
            if not self._run_open3d_phase("refine", refine_config):
                return self._stop("Both coloured and geometric refinement failed.")

        if not self._run_open3d_phase("integrate", refine_config):
            return self._stop("TSDF integration failed.")

        output = self.scene_path / "integrated.ply"
        if not output.is_file() or output.stat().st_size == 0:
            return self._stop("Integration ended without a usable integrated.ply.")

        print("\nLATEST 3D ROOM MODEL COMPLETED", flush=True)
        print(f"PLY file: {output}", flush=True)
        print(
            "Opening the reconstructed room automatically. Close the Open3D "
            "window to finish Stage 9.",
            flush=True,
        )
        viewer = self.project_root / "view_point_cloud.py"
        view_result = subprocess.run(
            [
                sys.executable,
                str(viewer),
                str(output),
                "--title",
                f"Latest reconstructed room - {self.dataset_path.name}",
            ],
            check=False,
        )
        if view_result.returncode != 0:
            print(
                "The PLY was generated successfully, but the viewer returned "
                f"{view_result.returncode}. You can still open: {output}",
                flush=True,
            )
        return 0

    def _validate_dataset(self) -> int:
        if not self.config_path.is_file():
            raise RuntimeError(f"Missing Open3D config: {self.config_path}")
        if not self.open3d_runner.is_file():
            raise RuntimeError(f"Missing Open3D reconstruction script: {self.open3d_runner}")

        colour = {path.stem for path in (self.dataset_path / "color").glob("*.jpg")}
        depth = {path.stem for path in (self.dataset_path / "depth").glob("*.png")}
        complete = colour & depth
        if not complete or colour != depth:
            raise RuntimeError(
                f"RGB/depth pairs are incomplete: {len(colour)} RGB and "
                f"{len(depth)} depth files."
            )
        return len(complete)

    def _fragments_are_present(self, expected: int) -> bool:
        files = list(self.fragment_path.glob("fragment_optimized_*.json"))
        clouds = list(self.fragment_path.glob("fragment_*.ply"))
        return len(files) == expected and len(clouds) == expected

    def _run_open3d_phase(self, phase: str, config: Path) -> bool:
        print(f"\nOpen3D phase: {phase}", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-B",
                str(self.open3d_runner),
                f"--{phase}",
                "--config",
                str(config),
            ],
            check=False,
        )
        if result.returncode != 0:
            print(
                f"Open3D phase '{phase}' returned {result.returncode}.",
                flush=True,
            )
        return result.returncode == 0

    def _audit_local_trajectory(self, expected: int) -> tuple[bool, str]:
        lines = [
            "Stage 9 automatic reconstruction safety audit",
            "================================================",
            "Translation is computed from Open3D column-major pose matrices.",
            "A step above 0.50 m at approximately 10 FPS is treated as unsafe.",
            "",
        ]
        safe = True
        files = sorted(self.fragment_path.glob("fragment_optimized_*.json"))
        if len(files) != expected:
            return False, "Expected %d fragments, found %d." % (expected, len(files))

        for fragment_index, path in enumerate(files):
            document = json.loads(path.read_text(encoding="utf-8"))
            poses = [
                np.asarray(node["pose"], dtype=float).reshape((4, 4), order="F")
                for node in document["nodes"]
            ]
            centres = np.asarray(
                [np.linalg.inv(pose)[:3, 3] for pose in poses], dtype=float
            )
            steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
            bad = np.flatnonzero(steps > 0.50)
            if bad.size:
                safe = False
            lines.append(
                f"Fragment {fragment_index}: median step {np.median(steps):.4f} m, "
                f"maximum step {np.max(steps):.4f} m, unsafe steps {bad.size}"
            )
            for local_index in bad[:10]:
                global_index = fragment_index * 100 + int(local_index)
                lines.append(
                    f"  frame {global_index}->{global_index + 1}: "
                    f"{steps[local_index]:.4f} m"
                )
            if bad.size > 10:
                lines.append(f"  ... plus {bad.size - 10} additional unsafe steps")

        lines.append("")
        lines.append("AUDIT RESULT: PASS" if safe else "AUDIT RESULT: WARNING")
        return safe, "\n".join(lines)

    def _write_geometric_config(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["name"] = f"{config.get('name', 'Room reconstruction')} - geometric ICP"
        config["icp_method"] = "point_to_plane"
        self.geometric_config_path.write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    def _stop(self, reason: str) -> int:
        print(f"\nAUTOMATIC RECONSTRUCTION STOPPED: {reason}", flush=True)
        print(f"Safety report: {self.report_path}", flush=True)
        return 1


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely reconstruct one room dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--strict-safety",
        action="store_true",
        help="Stop instead of generating a PLY when large pose jumps are found",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    raise SystemExit(
        RoomReconstructionPipeline(
            arguments.dataset,
            strict_safety=arguments.strict_safety,
        ).run()
    )
