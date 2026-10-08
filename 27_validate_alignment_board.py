"""Compare projection hypotheses on measured-board RGB/depth observations."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from processing.board_depth import corner_depths


def residual(objects, points):
    a=objects-objects.mean(axis=0); b=points-points.mean(axis=0)
    u,_,vt=np.linalg.svd(a.T@b)
    rotation=vt.T@u.T
    if np.linalg.det(rotation)<0:
        vt[-1]*=-1; rotation=vt.T@u.T
    rotated=a@rotation.T
    scale=float(np.sum(rotated*b)/np.sum(a*a))
    rigid_error=float(np.sqrt(np.mean(np.sum((rotated-b)**2,axis=1))))
    scaled_error=float(np.sqrt(np.mean(np.sum((scale*rotated-b)**2,axis=1))))
    return rigid_error,scale,scaled_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',nargs='?',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    folders=sorted((root/'report_evidence'/'alignment_board').glob('*/session.json'))
    folder=args.session or (folders[-1].parent if folders else None)
    if folder is None:raise ValueError('No Stage 26 session')
    meta=json.loads((folder/'session.json').read_text(encoding='utf-8'))
    observed=np.load(folder/'observations.npz',allow_pickle=False)['corners']
    rgb_k=np.asarray(meta['rgb_calibration']['camera_matrix'])
    rgb_d=np.asarray(meta['rgb_calibration']['distortion_coefficients'])
    objects=np.zeros((54,3));objects[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)*meta['square_mm']/1000
    candidates={name:[] for name in ('old_depth_fov','rgb_calibrated_pinhole','rgb_calibrated_undistorted_rays')}
    skipped=[]
    for corners,record in zip(observed,meta['observations']):
        index=record['frame']
        depth=cv2.imread(str(folder/'depth'/f'{index:05d}.png'),cv2.IMREAD_UNCHANGED)
        if depth is None:raise ValueError(f'Missing depth frame {index}')
        pixels=corners.reshape(-1,2)
        z=corner_depths(depth,corners);valid=np.isfinite(z)
        if valid.sum()<43:
            skipped.append({'frame':index,'valid_corners':int(valid.sum()),
                'overall_nonzero_depth_fraction':float(np.mean(depth>0))});continue
        old=record['depth_fov_intrinsic']
        old_rays=np.column_stack(((pixels[:,0]-old['cx'])/old['fx'],(pixels[:,1]-old['cy'])/old['fy']))
        rgb_rays=np.column_stack(((pixels[:,0]-rgb_k[0,2])/rgb_k[0,0],(pixels[:,1]-rgb_k[1,2])/rgb_k[1,1]))
        undistorted=cv2.undistortPoints(corners,rgb_k,rgb_d).reshape(-1,2)
        for name,rays in zip(candidates,(old_rays,rgb_rays,undistorted)):
            points=np.column_stack((rays*z[:,None],z))[valid]
            error,scale,scaled_error=residual(objects[valid],points)
            candidates[name].append({'frame':index,'valid_corners':int(valid.sum()),
                'rigid_rms_mm':error*1000,'measured_to_printed_scale':scale,
                'similarity_rms_mm':scaled_error*1000})
    summary={}
    for name,rows in candidates.items():
        if not rows:continue
        summary[name]={'evaluated_views':len(rows),
            'rigid_rms_mm_quartiles':np.quantile([r['rigid_rms_mm'] for r in rows],[.25,.5,.75]).tolist(),
            'scale_quartiles':np.quantile([r['measured_to_printed_scale'] for r in rows],[.25,.5,.75]).tolist()}
    status='experimental_projection_comparison_not_certified' if summary else 'no_valid_board_depth'
    report={'source_session':str(folder.resolve()),'status':status,
        'square_mm':meta['square_mm'],'summary':summary,'per_view':candidates,'skipped':skipped,
        'method':'Median valid depth in 5x5 corner neighbourhoods; rigid and similarity fit against measured planar checkerboard. Same accepted corners for all projection hypotheses.',
        'limitations':'Assumes recorded registered depth represents axial Z for the tested projection. Depth bias/noise, registration, paper scale/flatness and timing may confound results. Not a full RGB/depth extrinsic calibration, gravity alignment, room-completeness or navigation certification. No parameters are applied automatically.'}
    (folder/'projection_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Skipped views:',len(skipped));print(folder)


if __name__=='__main__':main()
