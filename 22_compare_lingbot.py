"""Isolated RGB-only LingBot-Map experiment; predicted scale is unverified."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import time
import subprocess
import hashlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--stride',type=int,default=10)
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--image-size',type=int,default=392)
    parser.add_argument('--keyframe-interval',type=int,default=4)
    parser.add_argument('--confidence',type=float,default=1.5)
    parser.add_argument('--reference-window',type=int,default=8)
    args=parser.parse_args()
    if args.stride<1 or args.max_frames<0 or args.keyframe_interval<1 or args.reference_window<1:
        parser.error('Invalid sampling settings')
    root=Path(__file__).resolve().parent
    source=root/'external'/'lingbot-map'
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    dataset=args.dataset.resolve()
    paths=sorted((dataset/'color').glob('*.jpg'))[::args.stride]
    if args.max_frames:paths=paths[:args.max_frames]
    if len(paths)<2:raise ValueError('Need at least two recorded RGB frames')
    out=root/'report_evidence'/'lingbot_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_dataset':str(dataset),'input_frames':[int(p.stem) for p in paths],
            'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
            'settings':vars(args)|{'dataset':str(dataset)},
            'geometry_source':'RGB-only neural depth and camera prediction; no measured depth supplied',
            'scale':'UNVERIFIED model units, not measured metres',
            'runtime_config':{'backend':'PyTorch SDPA','weight_dtype':'BF16 encoder; FP32 heads and LayerNorm','autocast_dtype':'bfloat16',
                              'scale_frames':2,'reference_window':args.reference_window,'camera_iterations':4,
                              'input_storage':'CPU; official model moves individual frames to GPU'},
            'status':'started'}
    def save(): (out/'comparison_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable in test environment')
        report['gpu']=torch.cuda.get_device_name(0)
        report['torch_version']=torch.__version__
        weight_path=source/'weights'/'lingbot-map.pt'
        with weight_path.open('rb') as stream:
            report['checkpoint_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        print('Loading RGB frames:',len(paths),flush=True)
        images=load_and_preprocess_images([str(p) for p in paths],mode='crop',image_size=args.image_size,patch_size=14)
        report['input_shape']=list(images.shape)
        print('Building model / loading official checkpoint on CPU',flush=True)
        # Checkpoint positional grid is 518; the official encoder interpolates
        # that grid for the actual input resolution at runtime.
        model=GCTStream(img_size=518,patch_size=14,use_sdpa=True,enable_3d_rope=True,
                        kv_cache_sliding_window=args.reference_window,kv_cache_scale_frames=2,
                        camera_num_iterations=4)
        checkpoint=torch.load(weight_path,map_location='cpu',weights_only=True,mmap=True)
        state=checkpoint.get('model',checkpoint)
        incompatible=model.load_state_dict(state,strict=False,assign=True)
        report['missing_keys']=list(incompatible.missing_keys)
        report['unexpected_keys']=list(incompatible.unexpected_keys)
        if incompatible.missing_keys:
            raise RuntimeError('Checkpoint is missing model weights; aborting instead of using random parameters')
        del state,checkpoint
        # Camera/depth heads require FP32 under the official forward path.
        # Keep normalisation FP32 too; only the large encoder uses BF16 weights.
        model.aggregator.to(dtype=torch.bfloat16)
        for layer in model.aggregator.modules():
            if isinstance(layer,torch.nn.LayerNorm):layer.float()
        model=model.to(device='cuda').eval()
        torch.cuda.reset_peak_memory_stats()
        print('Running inference with CPU prediction offload',flush=True)
        started=time.perf_counter()
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
            pred=model.inference_streaming(images,num_scale_frames=2,
                                          keyframe_interval=args.keyframe_interval,
                                          output_device=torch.device('cpu'))
        report['inference_seconds']=time.perf_counter()-started
        report['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
        extrinsics,intrinsics=pose_encoding_to_extri_intri(pred['pose_enc'].float(),images.shape[-2:])
        extrinsics=extrinsics[0].numpy();intrinsics=intrinsics[0].numpy()
        depth=pred['depth'][0].float().numpy()[...,0]
        confidence=pred['depth_conf'][0].float().numpy()
        rgb=images.permute(0,2,3,1).numpy()
        # Post-inference comparison only: recorded depth is never model input.
        # These ratios are not ground-truth accuracy or calibrated model scale.
        import cv2
        depth_agreement=[]
        for path,d,conf in zip(paths,depth,confidence):
            measured=cv2.imread(str(dataset/'depth'/f'{path.stem}.png'),cv2.IMREAD_UNCHANGED)
            if measured is None:continue
            new_height=round(measured.shape[0]*(args.image_size/measured.shape[1])/14)*14
            measured=cv2.resize(measured,(args.image_size,new_height),interpolation=cv2.INTER_NEAREST).astype(np.float32)/1000
            if new_height>args.image_size:
                start=(new_height-args.image_size)//2;measured=measured[start:start+args.image_size]
            if measured.shape!=d.shape:continue
            mask=(measured>=.4)&(measured<=5)&np.isfinite(d)&(d>0)&(conf>=args.confidence)
            if mask.sum()<100:continue
            ratios=measured[mask]/d[mask]
            depth_agreement.append({'frame':int(path.stem),'pixels':int(mask.sum()),
                                    'median_recorded_depth_to_model_ratio':float(np.median(ratios)),
                                    'ratio_quartiles':np.quantile(ratios,[.25,.75]).tolist()})
        report['recorded_depth_comparison']={
            'frames':depth_agreement,
            'limitations':'Post-inference consistency only. Recorded depth and RGB registration are not surveyed ground truth. Ratios do not establish metric accuracy.'}
        np.savez_compressed(out/'predictions.npz',depth=depth,confidence=confidence,
                            world_to_camera=extrinsics,intrinsics=intrinsics)
        points=[];colours=[];poses=[]
        for index,(d,conf,k,e) in enumerate(zip(depth,confidence,intrinsics,extrinsics)):
            y,x=np.mgrid[0:d.shape[0]:3,0:d.shape[1]:3]
            z=d[::3,::3];valid=np.isfinite(z)&(z>0)&np.isfinite(conf[::3,::3])&(conf[::3,::3]>=args.confidence)
            camera=np.stack([(x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z],axis=-1)[valid]
            matrix=np.eye(4);matrix[:3]=e;pose=np.linalg.inv(matrix);poses.append(pose)
            points.append(camera@pose[:3,:3].T+pose[:3,3])
            colours.append(rgb[index,::3,::3][valid])
        points=np.concatenate(points);colours=np.clip(np.concatenate(colours)*255,0,255).astype(np.uint8)
        if not len(points) or not np.isfinite(points).all():raise RuntimeError('Invalid predicted geometry')
        # Binary PLY with explicit model-unit provenance in its header.
        records=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
        for axis,name in enumerate(('x','y','z')):records[name]=points[:,axis]
        for axis,name in enumerate(('red','green','blue')):records[name]=colours[:,axis]
        header=('ply\nformat binary_little_endian 1.0\ncomment RGB-only predictions; unverified model units\n'
                f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        with (out/'predicted_cloud.ply').open('wb') as stream:stream.write(header.encode());stream.write(records.tobytes())
        np.save(out/'camera_to_world.npy',np.asarray(poses))
        report.update(status='completed',points=len(points),metric_accuracy_verified=False)
        (out/'README.md').write_text('RGB-only neural prediction experiment. Scale is unverified.\n'
                                    'This is not a calibrated metric rescue map; compare with recorded depth and surveyed dimensions.\n',encoding='utf-8')
        print('Comparison evidence:',out,flush=True)
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:save()


if __name__=='__main__':main()
