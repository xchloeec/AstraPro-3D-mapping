"""Stage 16 - conservative visual/depth loop constraints and pose-graph optimisation."""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


def q_to_r(x, y, z, w):
    q=np.array([x,y,z,w],float); q/=np.linalg.norm(q); x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def r_to_q(r):
    # Open3D utility returns quaternion in w,x,y,z order.
    q=o3d.geometry.get_rotation_matrix_from_quaternion
    trace=float(np.trace(r))
    if trace>0:
        s=math.sqrt(trace+1)*2; return ((r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s,.25*s)
    axis=int(np.argmax(np.diag(r)))
    if axis==0:
        s=math.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2; return (.25*s,(r[0,1]+r[1,0])/s,(r[0,2]+r[2,0])/s,(r[2,1]-r[1,2])/s)
    if axis==1:
        s=math.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2; return ((r[0,1]+r[1,0])/s,.25*s,(r[1,2]+r[2,1])/s,(r[0,2]-r[2,0])/s)
    s=math.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2; return ((r[0,2]+r[2,0])/s,(r[1,2]+r[2,1])/s,.25*s,(r[1,0]-r[0,1])/s)


class PoseGraphOptimizer:
    def __init__(self,dataset:Path):
        self.d=dataset.resolve(); self.out=self.d/'stage_16_pose_graph'; self.out.mkdir(exist_ok=True)
        self.assoc=[x.split() for x in (self.d/'associations.txt').read_text().splitlines() if x and not x.startswith('#')]
        info=json.loads((self.d/'camera_info.json').read_text()); self.fx=info['fx']/2; self.fy=info['fy']/2; self.cx=info['cx']/2; self.cy=info['cy']/2
        self.K=np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]],float)

    def run(self):
        rows=list(csv.DictReader((self.d/'stage_14_odometry'/'trajectory.csv').open()))
        poses=[]; timestamps=[]; indices=[]
        for row in rows:
            p=np.eye(4); p[:3,:3]=q_to_r(*[float(row[k]) for k in ('qx','qy','qz','qw')]); p[:3,3]=[float(row[k]) for k in ('x','y','z')]
            poses.append(p); timestamps.append(float(row['timestamp'])); indices.append(int(row['index']))
        graph=o3d.pipelines.registration.PoseGraph()
        for p in poses: graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(p))
        for i in range(len(poses)-1):
            t=np.linalg.inv(poses[i+1])@poses[i]
            graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i,i+1,t,np.eye(6)*100,uncertain=False))

        orb=cv2.ORB_create(3000,fastThreshold=10); cache={}
        def load(n):
            if n not in cache:
                dep=cv2.resize(cv2.imread(str(self.d/self.assoc[indices[n]][1]),-1),(320,240),interpolation=cv2.INTER_NEAREST)
                gray=cv2.resize(cv2.imread(str(self.d/self.assoc[indices[n]][3]),0),(320,240),interpolation=cv2.INTER_AREA)
                kp,des=orb.detectAndCompute(gray,None); cache[n]=(dep,kp,des)
            return cache[n]
        tested=accepted=0; accepted_rows=[]; bf=cv2.BFMatcher(cv2.NORM_HAMMING)
        keyframes=list(range(0,len(poses),5))
        for ai,i in enumerate(keyframes):
            for j in keyframes[ai+1:]:
                if j-i<20 or np.linalg.norm(poses[i][:3,3]-poses[j][:3,3])>1.2: continue
                tested+=1; di,ki,xi=load(i); _,kj,xj=load(j)
                if xi is None or xj is None: continue
                good=[m for m,n in bf.knnMatch(xi,xj,k=2) if m.distance<.70*n.distance]
                obj=[]; img=[]
                for m in good:
                    u,v=ki[m.queryIdx].pt; px,py=round(u),round(v)
                    if 0<=px<320 and 0<=py<240:
                        z=float(di[py,px])/1000
                        if .4<=z<=6: obj.append(((u-self.cx)*z/self.fx,(v-self.cy)*z/self.fy,z)); img.append(kj[m.trainIdx].pt)
                if len(obj)<25: continue
                ok,rv,tv,inl=cv2.solvePnPRansac(np.float32(obj),np.float32(img),self.K,None,iterationsCount=800,reprojectionError=2.5,confidence=.999)
                nin=0 if inl is None else len(inl)
                if not ok or nin<20 or nin/len(obj)<.35: continue
                r,_=cv2.Rodrigues(rv); measured=np.eye(4); measured[:3,:3]=r; measured[:3,3]=tv.ravel()
                predicted=np.linalg.inv(poses[j])@poses[i]; delta=np.linalg.inv(predicted)@measured
                trans=float(np.linalg.norm(delta[:3,3])); angle=math.degrees(math.acos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
                if trans>.35 or angle>15: continue
                graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i,j,measured,np.eye(6)*min(100,nin),uncertain=True)); accepted+=1
                accepted_rows.append({'source':indices[i],'target':indices[j],'matches':len(obj),'inliers':nin,'correction_m':trans,'correction_deg':angle})
        print(f'Stage 16 candidates tested: {tested}',flush=True); print(f'Validated loop constraints: {accepted}',flush=True)
        if accepted==0:
            print('No safe loop constraint found; original trajectory preserved.',flush=True); return 1
        option=o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=.07,edge_prune_threshold=.25,preference_loop_closure=1.0,reference_node=0)
        o3d.pipelines.registration.global_optimization(graph,o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),option)
        fields=['index','timestamp','x','y','z','qx','qy','qz','qw']; output=[]
        for idx,ts,node in zip(indices,timestamps,graph.nodes):
            p=node.pose; qx,qy,qz,qw=r_to_q(p[:3,:3]); output.append(dict(zip(fields,[idx,ts,*p[:3,3],qx,qy,qz,qw])))
        with (self.out/'trajectory.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(output)
        with (self.out/'loop_constraints.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(accepted_rows[0])); w.writeheader(); w.writerows(accepted_rows)
        summary={'tested_candidates':tested,'accepted_constraints':accepted,'input_poses':len(poses)}
        (self.out/'pose_graph_summary.json').write_text(json.dumps(summary,indent=2))
        print(f'Optimized trajectory: {self.out / "trajectory.csv"}',flush=True); return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('dataset',nargs='?',type=Path); a=ap.parse_args(); root=Path(__file__).resolve().parent
    d=a.dataset or sorted((root/'output'/'rgbd_datasets').glob('dataset_*'),key=lambda p:p.stat().st_mtime)[-1]
    raise SystemExit(PoseGraphOptimizer(d).run())
