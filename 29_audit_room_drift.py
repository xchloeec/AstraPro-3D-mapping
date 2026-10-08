"""Audit saved graph constraints without modifying the scan or trajectory."""
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from processing.validated_registration import rotation_degrees


def audit(folder):
    graph = o3d.io.read_pose_graph(str(folder / 'pose_graph_proposal.json'))
    lines = (folder / 'trajectory.txt').read_text().splitlines()
    frames = [int(line.split()[0]) for line in lines if line.strip()]
    rows = []
    for edge in graph.edges:
        a, b = edge.source_node_id, edge.target_node_id
        predicted = np.linalg.inv(graph.nodes[b].pose) @ graph.nodes[a].pose
        error = np.linalg.inv(edge.transformation) @ predicted
        rows.append(dict(source_frame=frames[a], target_frame=frames[b],
                         frame_gap=abs(frames[b]-frames[a]),
                         revisit=bool(edge.uncertain),
                         translation_error_m=float(np.linalg.norm(error[:3, 3])),
                         rotation_error_deg=rotation_degrees(error[:3, :3])))
    loops = [r for r in rows if r['revisit']]
    report = dict(source=str(folder), edges=rows,
                  revisit_count=len(loops),
                  longest_revisit_gap=max((r['frame_gap'] for r in loops), default=0),
                  long_revisits=[r for r in loops if r['frame_gap'] >= 300],
                  limitation='Graph residuals do not verify physical wall alignment. '
                  'A small residual can coexist with accumulated drift when long revisits are missing.')
    destination = folder / 'drift_audit.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved: {destination}')
    print(f"Revisits: {len(loops)}; longest gap: {report['longest_revisit_gap']} frames; "
          f"gaps >=300: {len(report['long_revisits'])}")
    for row in sorted(loops, key=lambda r: r['translation_error_m'], reverse=True)[:8]:
        print(row)
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parent / 'output' / 'live_mappings'
    folders = sorted(p.parent for p in root.glob('*/trajectory_refinement/*/refinement_report.json')
                     if (p.parent / 'pose_graph_proposal.json').is_file())
    if not folders:
        raise SystemExit('No saved optimised graph found.')
    audit(folders[-1])
