"""Fit RGB intrinsics from observed checkerboard corners, not room priors."""
import cv2
import numpy as np


def fit_checkerboard(corners, image_size, square_mm, pattern=(9, 6)):
    if len(corners) < 20 or square_mm <= 0:
        raise ValueError('Need 20 views and a positive measured square size')
    objects = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    objects[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objects *= square_mm / 1000
    observations = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in corners]
    if any(len(p) != len(objects) or not np.isfinite(p).all() for p in observations):
        raise ValueError('Invalid checkerboard observations')
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        [objects]*len(observations), observations, tuple(image_size), None, None)
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Non-finite calibration')
    per_view = []
    normals = []
    for observed, rotation, translation in zip(observations, rotations, translations):
        projected, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
        per_view.append(float(np.sqrt(np.mean(np.sum((observed-projected)**2, axis=2)))))
        normals.append(cv2.Rodrigues(rotation)[0][:, 2])
    centroids = np.array([p[:, 0].mean(axis=0) for p in observations])
    cells = {(min(3, max(0, int(x/image_size[0]*4))),
              min(2, max(0, int(y/image_size[1]*3)))) for x, y in centroids}
    normals = np.asarray(normals)
    tilt_span = np.degrees(np.ptp(np.arctan2(normals[:, :2], normals[:, 2:3]), axis=0))
    reasons = []
    if rms > 1.: reasons.append('Reprojection RMS exceeds heuristic 1 pixel')
    if len(cells) < 4: reasons.append('Board centres cover fewer than 4 of 12 image cells')
    if min(tilt_span) < 10.: reasons.append('Insufficient tilt diversity in one or both axes')
    return {'camera_matrix': matrix.tolist(), 'distortion_coefficients': distortion.ravel().tolist(),
            'image_size': list(image_size), 'pattern_inner_corners': list(pattern),
            'square_size_mm': square_mm, 'views': len(corners), 'rms_pixels': float(rms),
            'per_view_rms_pixels': per_view, 'board_centre_cells': sorted(cells),
            'tilt_span_degrees': tilt_span.tolist(),
            'status': 'needs_more_views_or_review' if reasons else 'estimated_passes_basic_checks',
            'review_reasons': reasons,
            'limitations': 'RGB intrinsic/distortion estimate only. Reprojection and diversity checks do not certify metric accuracy, depth registration, synchronisation or full-room coverage.'}
