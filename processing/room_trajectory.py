"""Recompute validated poses and conservatively optimise revisit constraints."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from .validated_registration import make_frame, matches_between, register_pair, rotation_degrees
from .projection_intrinsic import load_projection


def pose_lines(indices, poses):
    return "\n".join(f"{i} " + " ".join(f"{x:.9f}" for x in p.ravel()) for i,p in zip(indices,poses)) + "\n"


def optimise_trajectory(dataset: Path, stride: int = 3, max_frames: int = 0,
                        intrinsic_path: Path | None = None):
    dataset = dataset.resolve()
    indices = sorted(int(p.stem) for p in (dataset / "depth").glob("*.png")
                     if (dataset / "color" / f"{int(p.stem):05d}.jpg").is_file())[::stride]
    if max_frames:
        indices = indices[:max_frames]
    if not indices:
        raise ValueError("No complete RGB-D frames")
    out = dataset / "trajectory_refinement" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    intrinsic,projection_source = load_projection(dataset,intrinsic_path)
    if not o3d.io.write_pinhole_camera_intrinsic(str(out/'intrinsic_used.json'),intrinsic):
        raise RuntimeError('Could not archive trajectory projection')
    reg = o3d.pipelines.registration
    graph = reg.PoseGraph()
    accepted_indices, poses, logs, loops = [], [], [], []
    anchors = []
    reference = None
    reference_id = 0
    last_valid_frame = None
    failures = recoveries = 0
    started = time.perf_counter()

    def load(index):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(dataset / "color" / f"{index:05d}.jpg")),
            o3d.io.read_image(str(dataset / "depth" / f"{index:05d}.png")),
            depth_scale=1000, depth_trunc=5, convert_rgb_to_intensity=False,
        )
        return make_frame(rgbd,intrinsic)

    for number, index in enumerate(indices):
        current = load(index)
        source_id = reference_id
        if reference is None:
            ok, transform, information, details = True,np.eye(4),np.eye(6),{"reason":"initial"}
        else:
            ok, transform, information, details = register_pair(reference,current)
            if not ok and reference_id != len(poses)-1:
                advanced,t,info,d = register_pair(last_valid_frame,current)
                if advanced:
                    ok,transform,information,details = advanced,t,info,{**d,'reason':'validated_recent_reference'}
                    source_id = len(poses)-1
            if not ok:
                # Re-localise only with visually AND geometrically verified
                # stored views; never hold a pose and rebase the image alone.
                ranked = sorted(((len(matches_between(frame,current)), node,frame)
                                 for node,frame in anchors), key=lambda x:x[0], reverse=True)
                for score,node,frame in ranked[:3]:
                    if score < 25:
                        break
                    recovered,t,info,d = register_pair(frame,current,loop=True)
                    if recovered:
                        ok, transform, information, details = recovered,t,info,d
                        source_id = node
                        details["reason"] = "relocalised_validated_view"
                        recoveries += 1
                        break
        logs.append({"frame":index, "reference_node":source_id, **details, "accepted":bool(ok)})
        if not ok:
            failures += 1
            if number % 20 == 0:
                print(f"Refinement {number+1}/{len(indices)}: tracking unavailable; retaining validated reference", flush=True)
            continue
        pose = np.eye(4) if not poses else poses[source_id] @ np.linalg.inv(transform)
        target_id = len(poses)
        graph.nodes.append(reg.PoseGraphNode(pose))
        if poses:
            graph.edges.append(reg.PoseGraphEdge(source_id,target_id,transform,information,uncertain=False))
        poses.append(pose)
        accepted_indices.append(index)
        last_valid_frame = current
        # Attach small movements to a retained reference rather than repeatedly
        # multiplying noisy near-identity estimates. Every accepted frame still
        # gets its own pose and graph edge. Promote a new reference before the
        # baseline becomes too large for the sequential registration checks.
        if (reference is None or source_id != reference_id
                or np.linalg.norm(transform[:3,3]) >= .08
                or rotation_degrees(transform[:3,:3]) >= 5
                or target_id-reference_id >= 15):
            reference = current
            reference_id = target_id
        # Match visually similar stored keyframes, independent of drifted
        # positional proximity. Require both depth and bidirectional overlap.
        if target_id % 10 == 0:
            ranked = sorted(((len(matches_between(frame,current)), node,frame)
                             for node,frame in anchors if target_id-node >= 30),
                            key=lambda x:x[0], reverse=True)
            # Nearby, visually similar frames must not consume the entire
            # retrieval budget: they cannot constrain room-scale drift.
            distant = [item for item in ranked if target_id-item[1] >= 300][:3]
            candidates = ranked[:3] + [item for item in distant
                                      if item[1] not in {r[1] for r in ranked[:3]}]
            for score,node,frame in candidates:
                if score < 25:
                    break
                valid,t,info,d = register_pair(frame,current,loop=True)
                if not valid:
                    continue
                graph.edges.append(reg.PoseGraphEdge(node,target_id,t,info,uncertain=True))
                loops.append({"source_frame":accepted_indices[node], "target_frame":index,
                              "source_node":node,"target_node":target_id, **d})
            anchors.append((target_id,current))
            if len(anchors) > 100:
                # Retain distributed historic views as well as recent ones.
                # Evict a redundant old view instead of every early anchor;
                # otherwise a late return cannot find the original corner.
                removable = range(1, len(anchors)-30)
                drop = min(removable, key=lambda i: anchors[i+1][0]-anchors[i-1][0])
                del anchors[drop]
        if number % 20 == 0 or number == len(indices)-1:
            print(f"Refinement {number+1}/{len(indices)}: {len(poses)} validated poses, {len(loops)} revisit constraints", flush=True)

    if len(poses) < 2:
        raise RuntimeError("Fewer than two frames could be aligned safely; rescan a textured overlapping view")
    original = [p.copy() for p in poses]
    optimised = False
    optimisation_reason = "no_validated_revisit_constraints"
    retained_loops = 0
    graph_checks = None
    o3d.io.write_pose_graph(str(out/'pose_graph_before.json'),graph)
    if loops:
        reg.global_optimization(
            graph, reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(),
            reg.GlobalOptimizationOption(max_correspondence_distance=.06,edge_prune_threshold=.25,
                                         preference_loop_closure=1.,reference_node=0),
        )
        proposal = [np.asarray(node.pose).copy() for node in graph.nodes]
        retained_loops = sum(edge.uncertain for edge in graph.edges)
        shifts = [np.linalg.norm(a[:3,3]-b[:3,3]) for a,b in zip(original,proposal)]
        rotations = [rotation_degrees(a[:3,:3].T @ b[:3,:3]) for a,b in zip(original,proposal)]
        # Reject a graph that collapses a room through a large false revisit.
        edge_errors = []
        for edge in graph.edges:
            predicted = np.linalg.inv(proposal[edge.target_node_id]) @ proposal[edge.source_node_id]
            error = np.linalg.inv(edge.transformation) @ predicted
            if not edge.uncertain:
                edge_errors.append((np.linalg.norm(error[:3,3]),rotation_degrees(error[:3,:3])))
        graph_checks = {
            'maximum_pose_shift_m':float(max(shifts)),
            'maximum_pose_rotation_deg':float(max(rotations)),
            'maximum_certain_edge_translation_error_m':float(max((t for t,r in edge_errors),default=0)),
            'maximum_certain_edge_rotation_error_deg':float(max((r for t,r in edge_errors),default=0)),
            'limits':{'pose_shift_m':1.5,'pose_rotation_deg':30,
                      'certain_edge_translation_m':.08,'certain_edge_rotation_deg':5},
        }
        o3d.io.write_pose_graph(str(out/'pose_graph_proposal.json'),graph)
        (out/'trajectory_proposal_NOT_VERIFIED.txt').write_text(pose_lines(accepted_indices,proposal),encoding='utf-8')
        safe = (retained_loops > 0 and all(np.isfinite(p).all() for p in proposal)
                and max(shifts) <= 1.5 and max(rotations) <= 30
                and all(t <= .08 and r <= 5 for t,r in edge_errors))
        if safe:
            poses = proposal
            optimised = True
            optimisation_reason = "validated_revisit_graph_optimised"
        else:
            optimisation_reason = "graph_proposal_rejected_by_consistency_checks"
            print('WARNING: global trajectory correction rejected; exporting uncorrected poses. '
                  'Room layout may be drifted. Checks: '+json.dumps(graph_checks),flush=True)
    trajectory = out / "trajectory.txt"
    trajectory.write_text(pose_lines(accepted_indices,poses),encoding="utf-8")
    (out / "trajectory_before_graph.txt").write_text(pose_lines(accepted_indices,original),encoding="utf-8")
    (out / "tracking_quality.jsonl").write_text("\n".join(json.dumps(row) for row in logs)+"\n",encoding="utf-8")
    (out / "loop_constraints.json").write_text(json.dumps(loops,indent=2),encoding="utf-8")
    report = {
        "source_dataset":str(dataset),"tested_frames":len(indices),"validated_poses":len(poses),
        "intrinsic_source":str(projection_source),"projection_override_used":intrinsic_path is not None,
        "rejected_frames":failures,"relocalisations":recoveries,
        "accepted_revisit_constraints":len(loops),"retained_revisit_constraints":retained_loops,
        "pose_graph_optimised":optimised,"optimisation_reason":optimisation_reason,
        "graph_checks":graph_checks,
        "processing_seconds":time.perf_counter()-started,
        "validated_frame_fraction":len(poses)/len(indices),
        "registration_method":"depth-masked ORB, metric RANSAC, checked ICP; verified revisit pose graph",
        "candidate_descriptor_limit":0,"frame_stride":stride,
        "revisit_retrieval_policy":"top 3 overall plus top 3 >=300 nodes apart; distributed historic anchors",
        "reference_policy":"retained keyframe: 0.08m / 5deg / 15 accepted nodes; all tested frames retained when validated",
        "limitations":"Feature/depth and graph checks are heuristic. No measured room accuracy or completeness is guaranteed. Rejected frames are excluded rather than fused at an unknown pose.",
    }
    (out / "refinement_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Trajectory refinement: {out}; {len(poses)}/{len(indices)} frames validated",flush=True)
    return trajectory,report
