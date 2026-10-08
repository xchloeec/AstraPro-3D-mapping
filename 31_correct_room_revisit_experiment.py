"""Independent long-revisit correction with held-out constraints and evidence."""
from pathlib import Path
from datetime import datetime
import importlib.util
import json
import shutil
import cv2
import numpy as np
import open3d as o3d
from processing.room_trajectory import pose_lines
from processing.validated_registration import rotation_degrees


def run():
    root = Path(__file__).resolve().parent
    dataset = root / 'output/live_mappings/live_20261006_183452'
    previous = dataset / 'trajectory_refinement/20261008_103429_009412'
    measured = json.loads((previous / 'long_revisit_audit.json').read_text())
    candidates = [r for r in measured['results'] if r['accepted']]
    targets = sorted(set(r['target_frame'] for r in candidates))
    if len(targets) < 4:
        raise SystemExit('Insufficient independently measured revisits; no correction applied.')
    held_targets = set(targets[1::3])
    held = [r for r in candidates if r['target_frame'] in held_targets]
    fitted = [r for r in candidates if r['target_frame'] not in held_targets]
    lines = (previous / 'trajectory.txt').read_text().splitlines()
    ids = [int(line.split()[0]) for line in lines]
    lookup = {frame: i for i, frame in enumerate(ids)}
    original = [np.array(line.split()[1:], dtype=float).reshape(4, 4) for line in lines]
    reg = o3d.pipelines.registration
    graph = o3d.io.read_pose_graph(str(previous / 'pose_graph_proposal.json'))
    for row in fitted:
        graph.edges.append(reg.PoseGraphEdge(lookup[row['source_frame']], lookup[row['target_frame']],
                                            np.array(row['transform']), np.array(row['information']), uncertain=True))
    # Large real drift is immediately pruned by robust optimisation if started
    # at the drifted poses. Initialise a separate experiment from a measured
    # long revisit, spreading its rigid correction smoothly along the path.
    # Held-out revisit targets are never used to construct this initialisation.
    seed = max(fitted, key=lambda r: (r['source_frame'] == 0, r['fitness']))
    a, b = lookup[seed['source_frame']], lookup[seed['target_frame']]
    desired = original[a] @ np.linalg.inv(np.array(seed['transform']))
    correction = desired @ np.linalg.inv(original[b])
    rotation_vector = cv2.Rodrigues(correction[:3, :3])[0]
    for i, node in enumerate(graph.nodes):
        fraction = float(np.clip((i-a)/(b-a), 0., 1.))
        incremental = np.eye(4)
        incremental[:3, :3] = cv2.Rodrigues(rotation_vector*fraction)[0]
        incremental[:3, 3] = correction[:3, 3]*fraction
        node.pose = incremental @ original[i]
    out = dataset / 'trajectory_refinement' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir()
    o3d.io.write_pose_graph(str(out / 'pose_graph_before.json'), graph)
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(),
                            reg.GlobalOptimizationOption(max_correspondence_distance=.06,
                                                         edge_prune_threshold=.25,
                                                         preference_loop_closure=1., reference_node=0))
    proposal = [np.array(node.pose) for node in graph.nodes]
    def residual(a, b, transform):
        error = np.linalg.inv(transform) @ (np.linalg.inv(proposal[b]) @ proposal[a])
        return dict(translation_m=float(np.linalg.norm(error[:3, 3])), rotation_deg=rotation_degrees(error[:3, :3]))
    validation = [dict(source_frame=r['source_frame'], target_frame=r['target_frame'],
                       **residual(lookup[r['source_frame']], lookup[r['target_frame']], np.array(r['transform']))) for r in held]
    certain = [residual(e.source_node_id, e.target_node_id, e.transformation) for e in graph.edges if not e.uncertain]
    checks = dict(held_out=validation, fitted_count=len(fitted), held_out_count=len(held),
                  initialisation_seed_frames=[seed['source_frame'], seed['target_frame']],
                  maximum_certain_translation_m=max(r['translation_m'] for r in certain),
                  maximum_certain_rotation_deg=max(r['rotation_deg'] for r in certain),
                  maximum_pose_shift_m=max(float(np.linalg.norm(a[:3, 3]-b[:3, 3])) for a, b in zip(original, proposal)),
                  limits=dict(certain_translation_m=.08, certain_rotation_deg=5,
                              held_out_translation_m=.10, held_out_rotation_deg=3))
    passed = (len(held) >= 2 and all(np.isfinite(p).all() for p in proposal)
              and all(r['translation_m'] <= .08 and r['rotation_deg'] <= 5 for r in certain)
              and all(r['translation_m'] <= .10 and r['rotation_deg'] <= 3 for r in validation))
    checks['passed'] = passed
    (out / 'revisit_validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    o3d.io.write_pose_graph(str(out / 'pose_graph_proposal.json'), graph)
    (out / 'trajectory_proposal_NOT_VERIFIED.txt').write_text(pose_lines(ids, proposal), encoding='utf-8')
    print(json.dumps(checks, indent=2), flush=True)
    if not passed:
        raise SystemExit('Correction failed held-out/edge checks. Existing model unchanged; diagnostics saved.')
    shutil.copy2(previous / 'intrinsic_used.json', out / 'intrinsic_used.json')
    trajectory = out / 'trajectory.txt'
    trajectory.write_text(pose_lines(ids, proposal), encoding='utf-8')
    report = json.loads((previous / 'refinement_report.json').read_text())
    report.update(optimisation_reason='experimental_long_revisit_held_out_checks_passed',
                  revisit_validation=checks, experiment_parent=str(previous),
                  limitation='Experiment uses correlated views from one recording. Held-out constraints '
                  'support trajectory consistency but do not certify room accuracy or completeness.')
    (out / 'refinement_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('finalize_room', root / '18_finalize_room_scan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.finalize(dataset, refine=False, trajectory_path=trajectory,
                             intrinsic_path=out / 'intrinsic_used.json')
    shutil.copy2(out / 'revisit_validation.json', result / 'revisit_validation.json')
    print(f'Experimental room evidence: {result}', flush=True)


if __name__ == '__main__':
    run()
