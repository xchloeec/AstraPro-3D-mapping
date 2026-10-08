"""Archive a cleaned cloud and TSDF surface from a completed Stage 17 scan.

By default, recomputes validated poses and attempts checked revisit correction.
It does not invent unobserved surfaces or guarantee reconstruction accuracy.
Raw live poses remain in camera-world coordinates even though live_map.ply
has already been converted to Z-up by Stage 17.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
from processing.surface_coverage import SurfaceCoverage
from processing.room_trajectory import optimise_trajectory
from processing.room_layout import extract_layout
from processing.depth_quality import screen_depth
from processing.projection_intrinsic import load_projection


Z_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                 [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def load_poses(path: Path):
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 17:
            raise ValueError("Expected frame index plus 16 pose values")
        pose = np.array(values[1:], dtype=float).reshape(4, 4)
        rotation = pose[:3, :3]
        if (not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0, 0, 0, 1]) or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError("Invalid rigid camera pose")
        poses.append((int(values[0]), pose))
    if not poses:
        raise ValueError("No tracked poses found")
    return poses


def finalize(dataset: Path, voxel: float = 0.02, max_frames: int = 0,
             refine: bool = True, trajectory_path: Path | None = None,
             intrinsic_path: Path | None = None) -> Path:
    dataset = dataset.resolve()
    refinement = None
    if trajectory_path is None and refine:
        trajectory_path, refinement = optimise_trajectory(dataset,stride=1,intrinsic_path=intrinsic_path)
    elif trajectory_path is not None and (trajectory_path.parent / "refinement_report.json").is_file():
        refinement = json.loads((trajectory_path.parent / "refinement_report.json").read_text())
        if Path(refinement["source_dataset"]).resolve() != dataset:
            raise ValueError("The selected trajectory belongs to another recording")
    trajectory_path = trajectory_path or dataset / "live_trajectory.txt"
    poses = load_poses(trajectory_path)
    correction_failed = bool(refinement and refinement.get('optimisation_reason') ==
                             'graph_proposal_rejected_by_consistency_checks')
    if correction_failed:
        print('WARNING: global correction failed. Export is diagnostic evidence; '
              'do not treat this as a verified room layout.',flush=True)
    if refinement and refinement["validated_frame_fraction"] < .7:
        print("WARNING: less than 70% of tested frames aligned safely. This final model is partial; inspect tracking diagnostics before claiming room completeness.", flush=True)
    if max_frames and len(poses) > max_frames:
        poses = [poses[i] for i in np.linspace(0, len(poses)-1, max_frames, dtype=int)]
    intrinsic,projection_source = load_projection(dataset,intrinsic_path,trajectory_path)
    coverage = SurfaceCoverage()
    depth_quality = []
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for number, (index, pose) in enumerate(poses, 1):
        colour = dataset / "color" / f"{index:05d}.jpg"
        depth = dataset / "depth" / f"{index:05d}.png"
        if not colour.is_file() or not depth.is_file():
            raise FileNotFoundError(f"Missing RGB-D pair {index}")
        filtered_depth, statistics = screen_depth(
            np.asarray(o3d.io.read_image(str(depth)),dtype=np.float32)/1000.)
        depth_quality.append({'frame':index,**statistics})
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(colour)), o3d.geometry.Image(filtered_depth),
            depth_scale=1, depth_trunc=5, convert_rgb_to_intensity=False,
        )
        # TSDF expects world -> camera, unlike the saved camera -> world pose.
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        frame_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        frame_cloud = frame_cloud.voxel_down_sample(0.10)
        frame_cloud.transform(pose)
        coverage.update(frame_cloud.points, pose[:3, 3])
        if number % 20 == 0 or number == len(poses):
            print(f"TSDF integration {number}/{len(poses)}", flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError("No surface reconstructed; inspect depth and poses")
    mesh.transform(Z_UP)
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    cloud.transform(Z_UP)
    cloud = cloud.voxel_down_sample(voxel)
    before = len(cloud.points)
    if before >= 30:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if not cloud.has_points():
        raise RuntimeError("Filtering left no points")
    root = Path(__file__).resolve().parent
    out = root / "report_evidence" / "stage_18_final_room" / (
        f"{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    out.mkdir(parents=True)
    (out/'depth_quality.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in depth_quality)+'\n',encoding='utf-8')
    if not o3d.io.write_point_cloud(str(out / "room_cleaned.ply"), cloud):
        raise RuntimeError("Could not save cleaned cloud")
    if not o3d.io.write_triangle_mesh(str(out / "room_mesh.ply"), mesh):
        raise RuntimeError("Could not save mesh")
    try:
        layout_report = extract_layout(cloud,out,source_model=out/'room_cleaned.ply')
    except (RuntimeError,ValueError,OSError) as error:
        layout_report = {"error":str(error),"closed_room_boundary_verified":False}
        (out/'layout_report.json').write_text(json.dumps(layout_report,indent=2),encoding='utf-8')
    coverage_cloud = coverage.cloud()
    coverage_cloud.transform(Z_UP)
    if not o3d.io.write_point_cloud(str(out / "coverage_review.ply"), coverage_cloud):
        raise RuntimeError("Could not save coverage review")
    (out / "coverage_summary.json").write_text(json.dumps(coverage.summary(), indent=2), encoding="utf-8")
    for name in ("live_trajectory.txt", "recording_summary.txt", "frame_quality.csv", "intrinsic.json",
                 "pair_timing.csv", "capture_metadata.json", "live_mapping_summary.json", "lighting_quality.csv"):
        if (dataset / name).is_file():
            shutil.copy2(dataset / name, out / name)
    shutil.copy2(trajectory_path, out / "trajectory_used.txt")
    shutil.copy2(dataset/'intrinsic.json',out/'recording_intrinsic_original.json')
    if not o3d.io.write_pinhole_camera_intrinsic(str(out/'intrinsic.json'),intrinsic):
        raise RuntimeError('Could not archive fusion projection')
    for name in ("refinement_report.json", "loop_constraints.json", "tracking_quality.jsonl", "trajectory_before_graph.txt",
                 "pose_graph_before.json", "pose_graph_proposal.json", "trajectory_proposal_NOT_VERIFIED.txt"):
        if (trajectory_path.parent / name).is_file():
            shutil.copy2(trajectory_path.parent / name, out / name)
    report = {
        "source_dataset": str(dataset), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrated_frames": len(poses), "voxel_m": voxel,
        "surface_points_before_filter": before, "cleaned_points": len(cloud.points),
        "mesh_triangles": len(mesh.triangles), "coordinate_frame": "Z-up, metres",
        "trajectory_source": str(trajectory_path.resolve()),
        "intrinsic_source":str(projection_source),
        "projection_override_used":bool(intrinsic_path or (refinement or {}).get('projection_override_used')),
        "distortion_remapping_applied":False,
        "trajectory_recomputed": refinement is not None,
        "pose_graph_optimised": bool(refinement and refinement["pose_graph_optimised"]),
        "global_correction_failed":correction_failed,
        "refinement": refinement,
        "layout":layout_report,
        "photometric_processing":"Tracking-only bounded gamma and local contrast normalisation. Raw RGB/depth preserved; not measured sensor calibration.",
        "depth_screening":{
            "method":"3x3 support and depth discontinuity screening; no hole filling or material classification",
            "input_valid_pixels":sum(row['input_valid_pixels'] for row in depth_quality),
            "excluded_pixels":sum(row['excluded_pixels'] for row in depth_quality),
            "limitations":"Heuristic spatial screening can omit thin structures. Missing depth remains unknown, not free space."
        },
        "limitations": "Registration and revisit checks are heuristic. Mesh appearance is not proof of measured accuracy or room completeness; skipped frames can leave missing areas.",
    }
    (out / "processing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Final room processing\n\n"
        "room_mesh.ply is the reconstructed coloured surface. room_cleaned.ply is the filtered point cloud.\n"
        "Open either in CloudCompare, MeshLab or Open3D. Coordinate up is +Z.\n"
        "See trajectory_used.txt and refinement_report.json (when present) for the poses actually used.\n"
        "Revisit optimisation is applied only when verification checks pass; no surveyed accuracy is claimed.\n"
        "coverage_review.ply: red=one viewpoint bin, amber=two, green=three or more.\n"
        "layout_surfaces.ply: fitted candidates, red walls need more consistent observations.\n"
        "floorplan.png / floorplan.svg show observed layout; rescan_tasks.json lists follow-up capture tasks.\n"
        "Revisit red/amber surfaces with overlap from shifted camera positions. Unseen surfaces are unknown.\n"
        "Missing or distorted surfaces may remain. See processing_report.json for provenance.\n",
        encoding="utf-8",
    )
    print(f"Final room evidence: {out}", flush=True)
    return out


def review(out: Path):
    mesh = o3d.io.read_triangle_mesh(str(out / "room_mesh.ply"))
    mesh.compute_vertex_normals()
    coverage = o3d.io.read_point_cloud(str(out / "coverage_review.ply"))
    layout = None
    if (out/'layout_surfaces.ply').is_file():
        layout = o3d.io.read_triangle_mesh(str(out/'layout_surfaces.ply'))
        layout.compute_vertex_normals()
        layout_report = json.loads((out/'layout_report.json').read_text(encoding='utf-8'))
        transform = np.asarray(layout_report['scan_to_layout_transform'])
        mesh.transform(transform)
        coverage.transform(transform)
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    report = json.loads((out / "processing_report.json").read_text(encoding="utf-8"))
    partial = bool(report.get("refinement") and report["refinement"]["validated_frame_fraction"] < .7)
    correction_failed = bool(report.get('global_correction_failed') or
                            (report.get('refinement') or {}).get('optimisation_reason') ==
                            'graph_proposal_rejected_by_consistency_checks')
    label = ('UNVERIFIED - GLOBAL CORRECTION FAILED' if correction_failed else
             'PARTIAL room - tracking gaps' if partial else 'Unverified room review')
    if not viewer.create_window(f"{label} - C: coverage | L: fitted layout | Q: close", 1280, 800):
        raise RuntimeError("Could not open room review")
    for _ in range(10):
        viewer.poll_events()
        viewer.update_renderer()
    viewer.add_geometry(mesh)
    control = viewer.get_view_control()
    control.set_lookat(mesh.get_center())
    control.set_front([0, -1, 0])
    control.set_up([0, 0, 1])
    control.set_zoom(.55)
    options = viewer.get_render_option()
    options.background_color = np.array([.12, .14, .17])
    options.mesh_show_back_face = True
    options.point_size = 5
    geometries = {'mesh':mesh,'coverage':coverage,'layout':layout}
    state = ['mesh']
    def show(vis,name):
        if geometries[name] is None:
            print('No supported layout surfaces found; inspect layout_report.json.',flush=True)
            return False
        vis.remove_geometry(geometries[state[0]],reset_bounding_box=False)
        vis.add_geometry(geometries[name],reset_bounding_box=False)
        state[0] = name
        return True
    def toggle(vis):
        return show(vis,'mesh' if state[0]=='coverage' else 'coverage')
    viewer.register_key_callback(ord("C"), toggle)
    viewer.register_key_callback(ord('L'),lambda vis: show(vis,'mesh' if state[0]=='layout' else 'layout'))
    print('L toggles fitted layout. Red fitted walls: inconsistent plane fit; rescan recommended. Layout is an interpretation, not measured raw geometry.',flush=True)
    print("C toggles surface / coverage. Red: rescan; amber: add view; green: more views, not guaranteed accuracy.", flush=True)
    try:
        viewer.run()
    finally:
        viewer.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses every tracked frame")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--original-poses", action="store_true", help="Comparison only: bypass trajectory refinement")
    parser.add_argument("--poses", type=Path, help="Use an explicitly selected saved trajectory")
    parser.add_argument("--intrinsic",type=Path,help="Experimental replay projection; recomputes poses with the same projection")
    args = parser.parse_args()
    if args.voxel <= 0 or args.max_frames < 0:
        parser.error("voxel must be positive and max-frames nonnegative")
    dataset = args.dataset
    if dataset is None:
        base = Path(__file__).resolve().parent / "output" / "live_mappings"
        candidates = [p for p in base.glob("live_*") if (p / "live_trajectory.txt").is_file()]
        if not candidates:
            parser.error("No completed live scans found")
        dataset = max(candidates, key=lambda p: p.name)
    out = finalize(dataset, args.voxel, args.max_frames, refine=not args.original_poses,
                   trajectory_path=args.poses,intrinsic_path=args.intrinsic)
    if args.view:
        review(out)
