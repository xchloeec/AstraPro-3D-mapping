"""Search long revisits in existing evidence; never force room geometry."""
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from processing.validated_registration import make_frame, matches_between, register_pair, rotation_degrees


def run():
    root = Path(__file__).resolve().parent
    dataset = root / 'output/live_mappings/live_20261006_183452'
    folder = dataset / 'trajectory_refinement/20261008_103429_009412'
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(str(folder / 'intrinsic_used.json'))
    lines = (folder / 'trajectory.txt').read_text().splitlines()
    ids = [int(line.split()[0]) for line in lines]
    poses = [np.array(line.split()[1:], dtype=float).reshape(4, 4) for line in lines]
    # Inspect the beginning/end densely because a return to the starting
    # corner may occupy only a few frames, missed by the coarse room sweep.
    samples = sorted(set(ids[::30] + ids[:100:5] + ids[-100::5] + [ids[-1]]))
    frames = {}
    results = []
    retrieval = []
    for index in samples:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(dataset / 'color' / f'{index:05d}.jpg')),
            o3d.io.read_image(str(dataset / 'depth' / f'{index:05d}.png')),
            depth_scale=1000, depth_trunc=5, convert_rgb_to_intensity=False)
        current = make_frame(rgbd, intrinsic)
        ranked = sorted(((len(matches_between(frame, current)), old) for old, frame in frames.items()
                         if index-old >= 300), reverse=True)
        retrieval.append(dict(target_frame=index, best_match_count=ranked[0][0] if ranked else 0))
        for score, old in ranked[:3]:
            if score < 25:
                continue
            ok, transform, info, details = register_pair(frames[old], current, loop=True)
            predicted = np.linalg.inv(poses[ids.index(index)]) @ poses[ids.index(old)]
            error = np.linalg.inv(transform) @ predicted
            results.append(dict(details, source_frame=old, target_frame=index, accepted=ok,
                                graph_disagreement_m=float(np.linalg.norm(error[:3, 3])),
                                graph_disagreement_deg=rotation_degrees(error[:3, :3]),
                                transform=transform.tolist(), information=info.tolist()))
        frames[index] = current
        print(f'Long revisit search {index}/{ids[-1]}: {sum(r["accepted"] for r in results)} verified candidates', flush=True)
    out = folder / 'long_revisit_audit.json'
    out.write_text(json.dumps(dict(results=results, retrieval=retrieval, sampled_frames=samples,
                                  limitation='Diagnostic candidates only; existing poses and model unchanged.'), indent=2), encoding='utf-8')
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    run()
