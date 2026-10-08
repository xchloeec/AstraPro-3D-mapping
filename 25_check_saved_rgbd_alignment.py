"""Diagnostic edge alignment only; not a spatial calibration certificate."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    paths = sorted((dataset/'color').glob('*.jpg'), key=lambda p:int(p.stem))[::40]
    out = Path(__file__).resolve().parent/'report_evidence'/'alignment_audit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    totals = {}
    frames = []
    for path in paths:
        rgb = cv2.imread(str(path))
        depth = cv2.imread(str(dataset/'depth'/f'{path.stem}.png'), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
            continue
        z = depth.astype(np.float32)/1000
        valid = ((z>=.4)&(z<=5)).astype(np.uint8)
        supported = cv2.erode(valid,np.ones((3,3),np.uint8))>0
        differences = np.maximum(np.abs(z-np.roll(z,1,axis=0)),np.abs(z-np.roll(z,1,axis=1)))
        edge = supported & (differences>.12+.02*z)
        # Exclude roll boundaries and retain identical samples for every shift.
        edge[:22,:]=False; edge[-22:,:]=False
        edge[:,:22]=False; edge[:,-22:]=False
        y,x = np.nonzero(edge)
        if len(y)<80:continue
        rgb_edges = cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),60,120)
        distances = cv2.distanceTransform((rgb_edges==0).astype(np.uint8),cv2.DIST_L2,3)
        for dy in range(-20,21,2):
            for dx in range(-20,21,2):
                xx=x+dx; yy=y+dy
                mask=(xx>=0)&(xx<depth.shape[1])&(yy>=0)&(yy<depth.shape[0])
                values=distances[yy[mask],xx[mask]]
                score=float(np.mean(np.minimum(values,15))) if len(values) else 15.
                totals.setdefault((dx,dy),[]).append(score)
        frames.append({'frame':int(path.stem),'depth_edge_pixels':len(y),
                       'median_nearest_rgb_edge_px':float(np.median(distances[y,x])),
                       'fraction_within_3px':float(np.mean(distances[y,x]<=3))})
        if len(frames)<=6:
            overlay=rgb.copy()
            overlay[rgb_edges>0]=[0,255,0]
            overlay[edge]=[255,0,255]
            cv2.putText(overlay,'Green: RGB edges | Magenta: depth steps',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.imwrite(str(out/f'overlay_{path.stem}.png'),overlay)
    candidates=sorted(({'dx':dx,'dy':dy,'mean_clipped_edge_distance_px':float(np.mean(scores))}
                        for (dx,dy),scores in totals.items()),key=lambda r:r['mean_clipped_edge_distance_px'])
    zero=next((c for c in candidates if c['dx']==c['dy']==0),None)
    report={'dataset':str(dataset),'sampled_rgb_frames':len(paths),'evaluated_frames':len(frames),
            'status':'alignment_not_certified','frames':frames,'zero_shift':zero,'best_diagnostic_shifts':candidates[:10],
            'limitations':'RGB texture edges are not labelled depth boundaries; nearest-edge scores can match unrelated texture or reflections. Translation sweep does not estimate full camera intrinsics/extrinsics, distortion or timing. Do not apply the best shift as calibration.'}
    (out/'alignment_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(out)
    print(json.dumps({'evaluated_frames':len(frames),'zero_shift':zero,'best':candidates[:3]},indent=2))


if __name__=='__main__':main()
