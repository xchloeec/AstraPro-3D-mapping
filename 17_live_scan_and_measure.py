"""Stage 17: one-run live RGB-D mapping, PLY export and point measurement."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import open3d as o3d


def archive_scan(dataset_path: Path, evidence_path: Path, cloud, result: int) -> None:
    """Keep a standalone final map and provenance without duplicating raw frames."""
    evidence_path.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (
        "live_map.ply", "live_trajectory.txt", "recording_summary.txt",
        "frame_quality.csv", "intrinsic.json", "open3d_config.json",
        "point_measurements.csv",
        "coverage_summary.json",
        "pair_timing.csv", "tracking_quality.jsonl",
        "capture_metadata.json", "measurement_source.txt",
        "live_mapping_summary.json",
        "lighting_quality.csv",
    ):
        source = dataset_path / name
        if source.is_file():
            shutil.copy2(source, evidence_path / name)
            copied.append(name)
    manifest = {
        "scan_id": dataset_path.name,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_path.resolve()),
        "supervisor_exit_code": result,
        "point_count": len(cloud.points),
        "coordinate_frame": "X-right, Y-forward, Z-up; metres",
        "files": copied,
        "limitations": "Experimental reconstruction; cumulative pose drift remains. Not validated for accurate room dimensions.",
    }
    (evidence_path / "scan_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (evidence_path / "README.md").write_text(
        f"# Scan {dataset_path.name}\n\n"
        f"Original RGB-D recording: {dataset_path.resolve()}\n\n"
        "Open live_map.ply in Open3D, CloudCompare or MeshLab to revisit this scan.\n"
        "The PLY contains coloured points, not a watertight surface mesh.\n"
        "See scan_manifest.json for provenance and recording_summary.txt for capture statistics.\n"
        "Raw RGB/depth images remain in the original recording folder.\n"
        "Measurements are included when points were selected. Geometry accuracy remains experimental.\n",
        encoding="utf-8",
    )
    print(f"Report evidence saved: {evidence_path}", flush=True)


def load_live_supervisor(project_root: Path):
    path = project_root / "10_live_rgbd_mapping.py"
    spec = importlib.util.spec_from_file_location("stage10_live", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.LiveMappingSupervisor


def orient_z_up(cloud: o3d.geometry.PointCloud) -> None:
    cloud.transform(
        np.asarray(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64
        )
    )


def measure_points(cloud: o3d.geometry.PointCloud, result_folder: Path) -> None:
    print("\nPOINT-TO-POINT MEASUREMENT", flush=True)
    print("1. Hold Shift and left-click the first point.", flush=True)
    print("2. Hold Shift and left-click the second point.", flush=True)
    print("3. Press Q when selection is finished.", flush=True)
    editor = o3d.visualization.VisualizerWithEditing()
    editor.create_window("Stage 17 - Shift+click two points, then press Q", 1280, 800)
    # Let GLFW establish the window dimensions before setting the view.
    for _ in range(5):
        editor.poll_events()
        editor.update_renderer()
    editor.add_geometry(cloud)
    control = editor.get_view_control()
    control.set_lookat(cloud.get_center())
    control.set_front([0.0, -1.0, 0.0])
    control.set_up([0.0, 0.0, 1.0])
    control.set_zoom(0.55)
    options = editor.get_render_option()
    options.point_size = 3.0
    editor.run()
    editor.destroy_window()
    picked = editor.get_picked_points()
    if len(picked) < 2:
        print("Fewer than two points selected; no distance calculated.", flush=True)
        return
    points = np.asarray(cloud.points)
    lines = ["pair,point_1,point_2,distance_m,x1_m,y1_m,z1_m,x2_m,y2_m,z2_m"]
    for pair_number in range(0, len(picked) - 1, 2):
        first_index, second_index = picked[pair_number : pair_number + 2]
        first, second = points[first_index], points[second_index]
        distance = float(np.linalg.norm(second - first))
        number = pair_number // 2 + 1
        print(f"Measurement {number}: {distance:.4f} m ({distance*1000:.1f} mm)", flush=True)
        print(f"  Point 1 XYZ: {first}", flush=True)
        print(f"  Point 2 XYZ: {second}", flush=True)
        coordinates = ",".join(f"{value:.9f}" for value in np.concatenate([first,second]))
        lines.append(f"{number},{first_index},{second_index},{distance:.9f},{coordinates}")
    output = result_folder / "point_measurements.csv"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Measurements saved: {output}", flush=True)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    supervisor_type = load_live_supervisor(project_root)
    supervisor = supervisor_type(manual_control=True)
    print("Stage 17: live scan -> save PLY -> select points", flush=True)
    print("Move slowly with 60-80% overlap.", flush=True)
    print("Camera preview: Space starts/pauses/resumes; Q/Esc finishes and saves.", flush=True)
    print("Return to the paused camera view before resuming.", flush=True)
    print("In the 3D view: C shows rescan coverage; Space pauses/resumes capture.", flush=True)
    result = supervisor.run()
    ply_path = supervisor.dataset_path / "live_map.ply"
    if not ply_path.is_file():
        print("Stage 17 cannot measure because live_map.ply was not created.", flush=True)
        return result or 1
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if not cloud.has_points():
        print("The saved live map is empty.", flush=True)
        return 1
    orient_z_up(cloud)
    if not o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=False):
        print("Could not save the Z-up map; evidence archive was not created.", flush=True)
        return 1
    print(f"Z-up live PLY saved: {ply_path}", flush=True)
    evidence_path = (
        project_root / "report_evidence" / "stage_17_room_scans"
        / f"{supervisor.dataset_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    archive_scan(supervisor.dataset_path, evidence_path, cloud, result)
    measurement_source = ply_path
    final_path = None
    # Build a final surface automatically after capture, then review coverage.
    spec = importlib.util.spec_from_file_location("stage18_final", project_root / "18_finalize_room_scan.py")
    finalizer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(finalizer)
    try:
        final_path = finalizer.finalize(supervisor.dataset_path)
        (evidence_path / "final_surface_archive.txt").write_text(str(final_path.resolve()) + "\n", encoding="utf-8")
        final_cloud = o3d.io.read_point_cloud(str(final_path / "room_cleaned.ply"))
        if final_cloud.has_points():
            cloud = final_cloud
            measurement_source = final_path / "room_cleaned.ply"
        finalizer.review(final_path)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Final surface review failed; original scan evidence is preserved: {error}", flush=True)
    try:
        (supervisor.dataset_path / "measurement_source.txt").write_text(str(measurement_source.resolve()) + "\n",encoding="utf-8")
        measure_points(cloud, supervisor.dataset_path)
    finally:
        # The Stage 17 archive still describes the original live PLY.
        original_cloud = o3d.io.read_point_cloud(str(ply_path))
        archive_scan(supervisor.dataset_path, evidence_path, original_cloud, result)
        if final_path is not None:
            for name in ("point_measurements.csv","measurement_source.txt"):
                if (supervisor.dataset_path / name).is_file():
                    shutil.copy2(supervisor.dataset_path / name,final_path / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
