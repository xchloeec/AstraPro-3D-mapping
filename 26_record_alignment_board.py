"""Record known-size checkerboard RGB/depth pairs for projection validation."""
import argparse
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
import time
import cv2
import numpy as np
from camera.rgbd_adapter import RGBDFrameAdapter
from camera.rgbd_camera import RGBDCamera
from camera.isolated_rgb_camera import IsolatedRGBCamera
from processing.board_depth import corner_depths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-index',type=int,default=0)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--rgb-backend',choices=('DSHOW','MSMF'),default='DSHOW')
    parser.add_argument('--probe',action='store_true',help='Check startup and a few frames without showing a capture window')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    if not args.worker:
        attempts=[]
        backends=[args.rgb_backend,'MSMF' if args.rgb_backend=='DSHOW' else 'DSHOW']
        result=1
        for backend in backends:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',
                     '--rgb-backend',backend,'--camera-index',str(args.camera_index)]
            if args.calibration:command+=['--calibration',str(args.calibration)]
            if args.probe:command+=['--probe']
            print(f'Starting isolated camera worker with {backend}...',flush=True)
            result=subprocess.run(command).returncode
            attempts.append({'backend':backend,'returncode':result,'hex_status':f'0x{result & 0xffffffff:08X}'})
            if result==0:break
            print(f'Camera worker failed: {attempts[-1]["hex_status"]}; trying alternate backend if available.',flush=True)
            time.sleep(2)
        diagnostics=root/'report_evidence'/'camera_startup'
        diagnostics.mkdir(parents=True,exist_ok=True)
        (diagnostics/f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.json').write_text(
            json.dumps({'probe_only':args.probe,'attempts':attempts,'successful':result==0},indent=2),encoding='utf-8')
        if result!=0:print('Camera startup failed. Close other camera apps and reconnect the Astra before retrying.',flush=True)
        raise SystemExit(0 if result==0 else 1)
    files=sorted((root/'report_evidence'/'camera_calibration').glob('*/calibration.json'))
    source=args.calibration or (files[-1] if files else None)
    if source is None:raise RuntimeError('Complete Stage 24 calibration first')
    calibration=json.loads(source.read_text(encoding='utf-8'))
    if calibration['image_size'] != [640,480]:raise ValueError('Calibration must be 640x480')
    if calibration.get('camera_index') != args.camera_index:raise ValueError('Calibration camera index differs')
    adapter=RGBDFrameAdapter(RGBDCamera(rgb_camera_index=args.camera_index,
                            rgb_backends=(cv2.CAP_DSHOW if args.rgb_backend=='DSHOW' else cv2.CAP_MSMF,)))
    adapter.camera.rgb_camera=IsolatedRGBCamera(args.camera_index,args.rgb_backend)
    if args.probe:
        try:
            adapter.start()
            offsets=[]
            for number in range(30):
                frame=adapter.read();offsets.append(frame.timestamp_difference_ms)
                if number%5==0:
                    cv2.findChessboardCornersSB(cv2.cvtColor(frame.colour_bgr,cv2.COLOR_BGR2GRAY),
                        (9,6),cv2.CALIB_CB_NORMALIZE_IMAGE)
            print(f'Startup probe passed: {args.rgb_backend}; 30 paired reads with board detection; RGB {frame.colour_bgr.shape}; depth {frame.depth_mm.shape}; nonzero depth {np.mean(frame.depth_mm>0):.1%}; median software offset {np.median(offsets):.2f} ms',flush=True)
        finally:adapter.close()
        return
    out=root/'report_evidence'/'alignment_board'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (out/'color').mkdir(parents=True);(out/'depth').mkdir()
    views=[];records=[];metadata={}
    title='Stage 26 - alignment board - SPACE save / Q finish'
    print('Fix the board flat. Hold camera still before SPACE. Save 10-15 varied views; Q finishes.',flush=True)
    print(f"Using square size {calibration['square_size_mm']} mm from {source}",flush=True)
    print('This records evidence only; no mapping parameters are changed.',flush=True)
    print('Start around 60-90 cm away. Save only when board depth shows at least 43/54 valid corners.',flush=True)
    try:
        adapter.start()
        metadata={'backend':adapter.camera.rgb_camera.active_backend_name,
                  'uvc_isolated_process':True,
                  'depth_to_colour_registration_enabled':adapter.camera.registration_enabled,
                  'hardware_synchronised':False,'timestamp_source':'Nearest software arrival timestamps'}
        while True:
            frame=adapter.read()
            rgb=frame.colour_bgr;depth=frame.depth_mm
            if rgb.shape[:2] != (480,640) or depth.shape != (480,640):
                raise ValueError('RGB and registered depth must both be 640x480')
            found,corners=cv2.findChessboardCornersSB(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),
                (9,6),cv2.CALIB_CB_NORMALIZE_IMAGE)
            display=rgb.copy()
            if found:cv2.drawChessboardCorners(display,(9,6),corners,True)
            valid_corners=int(np.isfinite(corner_depths(depth,corners)).sum()) if found else 0
            cv2.putText(display,f'Pairs: {len(views)} / aim 10-15 | SPACE save | Q finish',
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,0),1)
            cv2.putText(display,'Hold still; entire board visible' if found else 'Board not detected',
                (8,44),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,0) if found else (0,100,255),1)
            cv2.putText(display,f'Board depth: {valid_corners}/54 (need 43+) - move further if low',
                (8,66),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0) if valid_corners>=43 else (0,100,255),1)
            heat=cv2.applyColorMap(np.clip(depth.astype(float)/2000*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
            heat[depth==0]=0
            cv2.putText(heat,'Registered depth | black = missing',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.imshow(title,np.hstack((display,heat)))
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key != 32:continue
            if not found:
                print('Not saved: full board not detected.',flush=True);continue
            if valid_corners<43:
                print(f'Not saved: board depth only {valid_corners}/54 valid corners. Move further away, avoid glare and keep the depth camera unobstructed.',flush=True);continue
            if frame.timestamp_difference_ms>25:
                print('Not saved: timestamp separation >25ms; hold still and retry.',flush=True);continue
            normalised=corners.reshape(-1,2)/[640,480]
            if any(np.sqrt(np.mean(np.sum((normalised-p.reshape(-1,2)/[640,480])**2,axis=1)))<.025 for p in views):
                print('Not saved: change distance, image position or tilt.',flush=True);continue
            index=len(views)
            if not cv2.imwrite(str(out/'color'/f'{index:05d}.png'),rgb):raise IOError('RGB save failed')
            if not cv2.imwrite(str(out/'depth'/f'{index:05d}.png'),depth):raise IOError('Depth save failed')
            views.append(corners.copy())
            records.append({'frame':index,'depth_timestamp_ns':frame.depth_timestamp_ns,
                'colour_timestamp_ns':frame.colour_timestamp_ns,'offset_ms':frame.timestamp_difference_ms,
                'valid_board_depth_corners':valid_corners,
                'depth_fov_intrinsic':{'fx':frame.intrinsics.fx,'fy':frame.intrinsics.fy,
                    'cx':frame.intrinsics.cx,'cy':frame.intrinsics.cy}})
            np.savez_compressed(out/'observations.npz',corners=np.asarray(views),image_size=[640,480])
            print(f'Saved pair {len(views)}; hold still at each new viewpoint.',flush=True)
    finally:
        adapter.close();cv2.destroyAllWindows()
        (out/'session.json').write_text(json.dumps({'views_saved':len(views),
            'calibration_source':str(source.resolve()),'rgb_calibration':calibration,
            'square_mm':calibration['square_size_mm'],'camera_index':args.camera_index,
            'status':'captured_not_yet_validated','capture_metadata':metadata,'observations':records,
            'limitations':'Joint observations for later projection checking. No calibration correction or hardware synchronisation certified.'},indent=2),encoding='utf-8')
        print(f'Alignment board evidence: {out}',flush=True)


if __name__=='__main__':main()
