"""Read-only RGB overlap audit; does not certify calibration or completeness."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    dataset = args.dataset.resolve()
    paths = sorted((dataset / 'color').glob('*.jpg'), key=lambda p: int(p.stem))
    if len(paths) < 2:
        raise ValueError('Need at least two saved RGB images')
    cv2.setNumThreads(4)
    cv2.setRNGSeed(7)
    orb = cv2.ORB_create(nfeatures=3000)
    cache = {}

    def features(index):
        if index not in cache:
            gray = cv2.imread(str(paths[index]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f'Unreadable image: {paths[index]}')
            keys, descriptors = orb.detectAndCompute(gray, None)
            cache[index] = (keys, descriptors)
        return cache[index]

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def pair(left, right):
        keys_a, desc_a = features(left)
        keys_b, desc_b = features(right)
        row = {'from_frame': int(paths[left].stem), 'to_frame': int(paths[right].stem),
               'features_a': len(keys_a), 'features_b': len(keys_b),
               'mutual_ratio_matches': 0, 'fundamental_inliers': 0,
               'homography_inliers': 0}
        if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 2:
            return row
        forward = [m for m, n in matcher.knnMatch(desc_a, desc_b, k=2) if m.distance < .75*n.distance]
        reverse = {(m.trainIdx, m.queryIdx) for m, n in matcher.knnMatch(desc_b, desc_a, k=2)
                   if m.distance < .75*n.distance}
        matches = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse]
        row['mutual_ratio_matches'] = len(matches)
        if len(matches) >= 8:
            a = np.float32([keys_a[m.queryIdx].pt for m in matches])
            b = np.float32([keys_b[m.trainIdx].pt for m in matches])
            _, mask_f = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, .999)
            _, mask_h = cv2.findHomography(a, b, cv2.RANSAC, 3.)
            row['fundamental_inliers'] = int(mask_f.sum()) if mask_f is not None else 0
            row['homography_inliers'] = int(mask_h.sum()) if mask_h is not None else 0
        return row

    endpoints = list(range(args.stride, len(paths), args.stride))
    rows = []
    for number, index in enumerate(endpoints):
        for name, left in (('adjacent', index-1), ('sampled', index-args.stride)):
            rows.append({'comparison': name, **pair(left, index)})
        if (number+1) % 25 == 0:
            print(f'Audited {number+1}/{len(endpoints)} endpoints', flush=True)

    def stats(values):
        a = np.asarray(values, dtype=float)
        return dict(zip(('minimum', 'p10', 'median', 'p90', 'maximum'),
                        np.quantile(a, [0, .1, .5, .9, 1]).tolist())) if len(a) else None

    summary = {'source_dataset': str(dataset), 'raw_rgb_frames': len(paths),
               'stride': args.stride, 'paired_endpoints': len(endpoints),
               'method': 'ORB 3000, mutual Lowe ratio 0.75, fundamental RANSAC 1.5px, homography RANSAC 3px',
               'heuristic_low_support_threshold': 20,
               'limitations': 'Inlier counts are image-match diagnostics, not verified overlap area, calibrated poses, scale, accuracy or room completeness.',
               'groups': {}}
    for name in ('adjacent', 'sampled'):
        group = [r for r in rows if r['comparison'] == name]
        summary['groups'][name] = {
            'mutual_matches': stats([r['mutual_ratio_matches'] for r in group]),
            'fundamental_inliers': stats([r['fundamental_inliers'] for r in group]),
            'low_support_pairs': sum(r['fundamental_inliers'] < 20 for r in group),
            'low_support_frame_ids': [r['to_frame'] for r in group if r['fundamental_inliers'] < 20]}
    meta = dataset / 'capture_metadata.json'
    summary['capture_metadata'] = json.loads(meta.read_text()) if meta.exists() else None
    for filename, fields in (
        ('pair_timing.csv', ('offset_ms',)),
        ('lighting_quality.csv', ('median_luma', 'dark_fraction', 'laplacian_variance')),
        ('frame_quality.csv', ('valid_depth_percentage',))):
        path = dataset / filename
        if path.exists():
            with path.open(newline='') as stream:
                data = list(csv.DictReader(stream))
            summary[filename] = {field: stats([float(r[field]) for r in data if r.get(field)])
                                 for field in fields}
    out = Path(__file__).resolve().parent / 'report_evidence' / 'recording_audit' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out / 'audit_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out / 'pair_diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out, flush=True)


if __name__ == '__main__':
    main()
