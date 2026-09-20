"""Stage 17: one-run live RGB-D mapping, PLY export and point measurement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


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
            dtype=np.float64,
        )
    )


def measure_points(cloud: o3d.geometry.PointCloud, result_folder: Path) -> None:
    print("\nPOINT-TO-POINT MEASUREMENT", flush=True)
    print("1. Hold Shift and left-click the first point.", flush=True)
    print("2. Hold Shift and left-click the second point.", flush=True)
    print("3. Press Q when selection is finished.", flush=True)
    editor = o3d.visualization.VisualizerWithEditing()
    editor.create_window("Stage 17 - Shift+click two points, then press Q", 1280, 800)
    editor.add_geometry(cloud)
    options = editor.get_render_option()
    options.point_size = 3.0
    editor.run()
    editor.destroy_window()
    picked = editor.get_picked_points()
    if len(picked) < 2:
        print("Fewer than two points selected; no distance calculated.", flush=True)
        return
    points = np.asarray(cloud.points)
    lines = ["pair,point_1,point_2,distance_m"]
    for pair_number in range(0, len(picked) - 1, 2):
        first_index, second_index = picked[pair_number : pair_number + 2]
        first, second = points[first_index], points[second_index]
        distance = float(np.linalg.norm(second - first))
        number = pair_number // 2 + 1
        print(f"Measurement {number}: {distance:.4f} m ({distance*1000:.1f} mm)", flush=True)
        print(f"  Point 1 XYZ: {first}", flush=True)
        print(f"  Point 2 XYZ: {second}", flush=True)
        lines.append(f"{number},{first_index},{second_index},{distance:.9f}")
    output = result_folder / "point_measurements.csv"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Measurements saved: {output}", flush=True)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    supervisor_type = load_live_supervisor(project_root)
    supervisor = supervisor_type()
    print("Stage 17: live scan -> save PLY -> select points", flush=True)
    print("Move slowly with 60-80% overlap.", flush=True)
    print("Close the LIVE mapping window when scanning is finished.", flush=True)
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
    o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=False)
    print(f"Z-up live PLY saved: {ply_path}", flush=True)
    measure_points(cloud, supervisor.dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
