"""Fit observed wall candidates and floor; never replace a room with a box."""
from pathlib import Path
import json
import numpy as np
import open3d as o3d


def align_normal(normal):
    normal = np.asarray(normal,float)
    if normal[2] < 0:
        normal = -normal
    v = np.cross(normal,[0,0,1])
    cosine = normal[2]
    skew = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew/(1+cosine)


def patch_mesh(vertices,colour):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    mesh.paint_uniform_color(colour)
    mesh.compute_vertex_normals()
    return mesh


def extract_layout(cloud, out: Path, source_model: Path | None = None):
    """Outputs fitted candidates, separate from measured geometry and furniture."""
    out.mkdir(parents=True,exist_ok=True)
    sampled = cloud.voxel_down_sample(.04)
    points = np.asarray(sampled.points)
    if len(points) < 500:
        raise ValueError("Insufficient surface points for layout analysis")
    remaining = sampled
    planes = []
    o3d.utility.random.seed(42)
    for _ in range(24):
        if len(remaining.points) < 500:
            break
        equation,indices = remaining.segment_plane(distance_threshold=.025,ransac_n=3,num_iterations=1500)
        if len(indices) < 500:
            break
        support = np.asarray(remaining.points)[indices].copy()
        equation = np.asarray(equation,float)
        equation /= np.linalg.norm(equation[:3])
        planes.append((equation,support))
        remaining = remaining.select_by_index(indices,invert=True)
    floor = None
    low_height = np.quantile(points[:,2],.15)
    for equation,support in planes:
        if abs(equation[2]) < .9:
            continue
        span = np.quantile(support[:,:2],.98,axis=0)-np.quantile(support[:,:2],.02,axis=0)
        if np.prod(span) < 1.5 or np.median(support[:,2]) > low_height+.15:
            continue
        if floor is None or np.median(support[:,2]) < np.median(floor[1][:,2]):
            floor = equation,support
    transform = np.eye(4)
    if floor is not None:
        equation,_ = floor
        transform[:3,:3] = align_normal(equation[:3])
        point_on_floor = -equation[3]*equation[:3]
        transform[:3,3] = -transform[:3,:3]@point_on_floor
    def level(p):
        return p@transform[:3,:3].T+transform[:3,3]
    walls = []
    for equation,support in planes:
        normal = transform[:3,:3]@equation[:3]
        p = level(support)
        if abs(normal[2]) > .15:
            continue
        normal = normal[:2]/np.linalg.norm(normal[:2])
        tangent = np.array([-normal[1],normal[0]])
        along = p[:,:2]@tangent
        order = np.argsort(along)
        # Disconnected patches on the same plane must not become one wall
        # across an unseen gap, recess or doorway.
        splits = np.flatnonzero(np.diff(along[order]) > .3)+1
        for group in np.split(order,splits):
            if len(group) < 300:
                continue
            patch = p[group]
            length = np.quantile(along[group],.98)-np.quantile(along[group],.02)
            height = np.quantile(patch[:,2],.98)-np.quantile(patch[:,2],.02)
            if length < 1. or height < 1.2 or length*height < 1.5:
                continue
            walls.append({"normal":normal,"support":patch,"length":float(length)})
    # Orthogonal regularisation only where detected wall normals agree.
    consensus = 0.
    yaw = 0.
    if walls:
        votes = sum(w['length']*np.exp(4j*np.arctan2(w['normal'][1],w['normal'][0])) for w in walls)
        consensus = float(abs(votes)/sum(w['length'] for w in walls))
        yaw = float(np.angle(votes)/4)
    fitted = o3d.geometry.TriangleMesh()
    report_walls = []
    for number,wall in enumerate(walls,1):
        normal,p = wall['normal'],wall['support']
        angle = np.arctan2(normal[1],normal[0])
        snapped = yaw + round((angle-yaw)/(np.pi/2))*(np.pi/2)
        regularised = consensus >= .8 and abs(angle-snapped) < np.deg2rad(7)
        if regularised:
            proposal = np.array([np.cos(snapped),np.sin(snapped)])
            residual = np.abs(p[:,:2]@proposal-np.median(p[:,:2]@proposal))
            if np.quantile(residual,.9) <= .04:
                normal = proposal
            else:
                regularised = False
        tangent = np.array([-normal[1],normal[0]])
        offset = float(np.median(p[:,:2]@normal))
        lo,hi = np.quantile(p[:,:2]@tangent,[.02,.98])
        zlo,zhi = np.quantile(p[:,2],[.02,.98])
        start,end = normal*offset+tangent*lo,normal*offset+tangent*hi
        vertices = [[*start,zlo],[*end,zlo],[*end,zhi],[*start,zhi]]
        errors = np.abs(p[:,:2]@normal-offset)
        weak_fit = bool(np.median(errors) > .03)
        fitted += patch_mesh(vertices,[.86,.48,.40] if weak_fit else [.72,.80,.88])
        report_walls.append({"id":number,"type":"observed_vertical_plane_candidate",
            "start_xy_m":start.tolist(),"end_xy_m":end.tolist(),
            "height_range_m":[float(zlo),float(zhi)],"support_points":len(p),
            "median_fit_error_m":float(np.median(errors)),"orthogonally_regularised":bool(regularised),
            "rescan_recommended":weak_fit})
    if floor is not None:
        p = level(floor[1])
        # Only occupied floor cells: a bounding rectangle would erase recesses
        # or invent floor where the camera has no observation.
        size = .15
        for cell in np.unique(np.floor(p[:,:2]/size).astype(int),axis=0):
            x,y = cell*size
            fitted += patch_mesh([[x,y,0],[x+size,y,0],
                                  [x+size,y+size,0],[x,y+size,0]],[.55,.58,.60])
    if len(fitted.triangles):
        if not o3d.io.write_triangle_mesh(str(out/'layout_surfaces.ply'),fitted):
            raise OSError("Could not save fitted layout")
    report = {"floor_detected":floor is not None,"wall_candidates":report_walls,
              "source_model":str(source_model.resolve()) if source_model else None,
              "orthogonal_direction_consensus":consensus,
              "scan_to_layout_transform":transform.tolist(),"closed_room_boundary_verified":False,
              "representation":"Fitted wall envelopes and occupied floor cells, not raw measured triangles. Wall envelopes may span openings; floor cells are not a verified room boundary.",
              "limitations":"Tall furniture can resemble walls. Openings and unseen surfaces are not inferred. Gaps and recesses are retained; no rectangular bounding room is invented. Dimensions are unverified."}
    (out/'layout_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    tasks = [{"wall_id":wall['id'],"reason":"Plane fit is inconsistent with a clean vertical surface",
              "start_xy_m":wall['start_xy_m'],"end_xy_m":wall['end_xy_m'],
              "action":"Revisit this observed wall from overlapping shifted views; include its corners and floor junction."}
             for wall in report_walls if wall['rescan_recommended']]
    if floor is None:
        tasks.append({"reason":"No supported floor candidate","action":"Capture the floor and wall-floor junctions from several overlapping positions."})
    tasks.append({"reason":"Closed room boundary not verified",
                  "action":"Include every wall, recess and partition corner; revisit the starting view. Compare the floor plan with the actual room before marking capture complete."})
    (out/'rescan_tasks.json').write_text(json.dumps({"complete_room_layout_verified":False,"tasks":tasks},indent=2),encoding='utf-8')
    render_floorplan(level(points),report_walls,out/'floorplan.svg')
    render_floorplan_png(level(points),report_walls,out/'floorplan.png')
    print(f"Layout: floor {'detected' if floor is not None else 'unknown'}, {len(walls)} wall candidates; boundary completeness unverified",flush=True)
    return report


def render_floorplan(points,walls,path):
    lo,hi = np.quantile(points[:,:2],[.01,.99],axis=0)
    for wall in walls:
        for key in ('start_xy_m','end_xy_m'):
            lo = np.minimum(lo,wall[key]); hi = np.maximum(hi,wall[key])
    extent = np.maximum(hi-lo,.1)
    scale = min(900/extent[0],650/extent[1])
    def xy(p):
        return 70+(p[0]-lo[0])*scale,740-(p[1]-lo[1])*scale
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="850" viewBox="0 0 1100 850">',
             '<rect width="1100" height="850" fill="#f4f6f8"/>',
             '<text x="40" y="35" font-family="sans-serif" font-size="22">Observed room layout — candidate walls</text>',
             '<text x="40" y="62" font-family="sans-serif" font-size="14">Gaps remain unknown. Fitted geometry; not a verified closed floor plan.</text>']
    for p in points[::max(1,len(points)//12000)]:
        x,y = xy(p)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r=".7" fill="#b6c1ca"/>')
    for wall in walls:
        a,b = xy(wall['start_xy_m']),xy(wall['end_xy_m'])
        colour = '#b34b38' if wall.get('rescan_recommended') else '#205378'
        parts.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{colour}" stroke-width="5"/>')
        parts.append(f'<text x="{(a[0]+b[0])/2:.2f}" y="{(a[1]+b[1])/2-8:.2f}" font-family="sans-serif" font-size="14">W{wall["id"]}</text>')
    parts.append('<text x="40" y="810" font-family="sans-serif" font-size="14">Coordinate units: metres. Furniture and recesses require review; measured scale remains unverified.</text></svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')


def render_floorplan_png(points,walls,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axis = plt.subplots(figsize=(11,8),dpi=140)
    figure.set_facecolor('#f4f6f8'); axis.set_facecolor('#f4f6f8')
    sample = points[::max(1,len(points)//15000)]
    axis.scatter(sample[:,0],sample[:,1],s=.5,c='#bbc5ce',rasterized=True)
    for wall in walls:
        a,b = np.array(wall['start_xy_m']),np.array(wall['end_xy_m'])
        colour = '#b34b38' if wall.get('rescan_recommended') else '#205378'
        axis.plot([a[0],b[0]],[a[1],b[1]],color=colour,linewidth=3)
        centre = (a+b)/2
        axis.annotate(f"W{wall['id']}",centre,xytext=(5,5),textcoords='offset points',color=colour)
    axis.set_aspect('equal',adjustable='datalim')
    axis.set_xlabel('Layout X (m, unverified scale)');axis.set_ylabel('Layout Y (m)')
    axis.set_title('Observed room layout: candidate walls\nRed: rescan recommended | gaps and unseen structure remain unknown',loc='left',fontsize=12,pad=15)
    axis.spines[['top','right']].set_visible(False)
    figure.text(.1,.02,'Fitted interpretation, not a verified closed floor plan. Tall furniture may resemble walls.',fontsize=9,color='#555555')
    figure.tight_layout(rect=[0,.05,1,1])
    figure.savefig(path);plt.close(figure)
