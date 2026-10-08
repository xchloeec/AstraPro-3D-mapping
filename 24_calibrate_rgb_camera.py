"""Capture the printed 9x6-inner-corner board for RGB calibration at 640x480."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from camera.rgb_camera import RGBCamera
from processing.camera_calibration import fit_checkerboard


def fit_saved_session(folder):
    folder = folder.resolve()
    metadata = json.loads((folder/'session.json').read_text(encoding='utf-8'))
    with np.load(folder/'observations.npz', allow_pickle=False) as data:
        result = fit_checkerboard(data['corners'], tuple(int(v) for v in data['image_size']),
                                  float(metadata['square_mm']))
    result.update(camera_index=metadata['camera_index'], backend='Not recorded in original session',
                  square_size_source='Measured print size recorded in capture session',
                  observations=metadata['observations'], fit_method='offline saved checkerboard corners')
    (folder/'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    image = cv2.imread(str(folder/metadata['observations'][-1]['image']))
    if image is not None:
        preview = cv2.undistort(image, np.asarray(result['camera_matrix']), np.asarray(result['distortion_coefficients']))
        cv2.imwrite(str(folder/'undistorted_preview.png'), preview)
    metadata.update(calibration_saved=True, offline_fit=True)
    (folder/'session.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"RMS {result['rms_pixels']:.3f}px; {result['status']}", flush=True)
    print(result['review_reasons'], flush=True)
    print(f'Calibration evidence: {folder}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--square-mm', type=float, default=17.1,
                        help='Actual square size in mm (default: 17.1 for the current measured print)')
    parser.add_argument('--fit-saved', type=Path,
                        help='Fit a saved session without opening the camera; uses its recorded square size')
    args = parser.parse_args()
    if args.fit_saved is not None:
        fit_saved_session(args.fit_saved)
        return
    if args.square_mm <= 0: parser.error('Square size must be positive')
    out = Path(__file__).resolve().parent/'report_evidence'/'camera_calibration'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    camera = RGBCamera(args.camera_index, requested_width=640, requested_height=480)
    corners = []
    records = []
    result = None
    message = 'Show the whole board. SPACE saves; C calibrates after 20+ views; Q exits.'
    print(message, flush=True)
    print(f'Printed square size: {args.square_mm:g} mm (current print default: 17.1 mm).', flush=True)
    print('Move the camera around the fixed board: vary tilt, distance and image position.', flush=True)
    print('This does not change Stage 17/18 camera parameters.', flush=True)
    try:
        camera.start()
        while True:
            frame = camera.read()
            if frame.shape[:2] != (480, 640):
                raise RuntimeError(f'Expected 640x480, got {frame.shape[1]}x{frame.shape[0]}; calibration must match scanning mode')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, detected = cv2.findChessboardCornersSB(gray, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
            display = frame.copy()
            if found: cv2.drawChessboardCorners(display, (9, 6), detected, found)
            cv2.putText(display, f'Views: {len(corners)} | SPACE save | C fit | Q exit',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 0), 1)
            cv2.putText(display, 'Board detected' if found else 'Whole board needed - improve angle/light',
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 0) if found else (0, 100, 255), 1)
            cv2.imshow('Stage 24 - RGB calibration', display)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27) or cv2.getWindowProperty('Stage 24 - RGB calibration', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                if not found:
                    print('Not saved: checkerboard not detected.', flush=True); continue
                area = cv2.contourArea(cv2.convexHull(detected)) / (640*480)
                if area < .03:
                    print('Not saved: move closer so the board occupies more of the image.', flush=True); continue
                normalised = detected.reshape(-1, 2)/np.array([640, 480])
                duplicate = any(np.sqrt(np.mean(np.sum((normalised-p.reshape(-1, 2)/[640, 480])**2, axis=1))) < .025 for p in corners)
                if duplicate:
                    print('Not saved: move to a different position, distance or tilt.', flush=True); continue
                path = out/f'view_{len(corners):03d}.png'
                if not cv2.imwrite(str(path), frame): raise IOError(f'Failed to save {path}')
                corners.append(detected.copy())
                records.append({'image': path.name, 'board_corner_hull_fraction': float(area)})
                np.savez_compressed(out/'observations.npz', corners=np.asarray(corners), image_size=[640,480])
                print(f'Saved {len(corners)} views. Aim for 25-30 varied views.', flush=True)
            if key == ord('c'):
                if len(corners) < 20:
                    print('Need at least 20 different views; aim for 25-30.', flush=True); continue
                result = fit_checkerboard(corners, (640,480), args.square_mm)
                result.update(camera_index=args.camera_index, backend=camera.active_backend_name,
                              square_size_source='User-supplied measured printed square size', observations=records)
                (out/'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                preview = cv2.undistort(frame, np.asarray(result['camera_matrix']), np.asarray(result['distortion_coefficients']))
                cv2.imwrite(str(out/'undistorted_preview.png'), preview)
                print(f"RMS {result['rms_pixels']:.3f}px; {result['status']}", flush=True)
                print(result['review_reasons'], flush=True)
                print(f'Result saved: {out}. Add views and press C again if review is needed.', flush=True)
    finally:
        camera.close()
        cv2.destroyAllWindows()
        (out/'session.json').write_text(json.dumps({'views_saved': len(corners), 'square_mm': args.square_mm,
                'camera_index': args.camera_index, 'calibration_saved': result is not None,
                'scan_mode': [640,480], 'observations': records}, indent=2), encoding='utf-8')
        print(f'Calibration evidence: {out}', flush=True)


if __name__ == '__main__': main()
