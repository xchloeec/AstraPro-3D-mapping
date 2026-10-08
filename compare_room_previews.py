"""Render two saved room surfaces with one common camera for visual comparison."""
import argparse
from pathlib import Path
import time
import numpy as np
import open3d as o3d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path)
    parser.add_argument('after',type=Path)
    args=parser.parse_args()
    meshes=[]
    for folder in (args.before,args.after):
        mesh=o3d.io.read_triangle_mesh(str(folder/'room_mesh.ply'))
        if not len(mesh.triangles):raise ValueError(f'No room mesh: {folder}')
        mesh.compute_vertex_normals();meshes.append(mesh)
    lower=np.minimum(meshes[0].get_min_bound(),meshes[1].get_min_bound())
    upper=np.maximum(meshes[0].get_max_bound(),meshes[1].get_max_bound())
    common_center=(lower+upper)/2
    viewer=o3d.visualization.Visualizer()
    if not viewer.create_window(width=1280,height=900,visible=False):
        raise RuntimeError('Preview renderer unavailable')
    try:
        for _ in range(10):viewer.poll_events();viewer.update_renderer();time.sleep(.02)
        # One renderer and a joint bound avoids Windows hidden-window viewport
        # inconsistencies across repeated window destruction/recreation.
        for mesh in meshes:viewer.add_geometry(mesh)
        for mesh in meshes:viewer.remove_geometry(mesh,reset_bounding_box=False)
        control=viewer.get_view_control()
        control.set_lookat(common_center)
        control.set_front([.65,-.9,.7]);control.set_up([0,0,1]);control.set_zoom(.65)
        options=viewer.get_render_option()
        options.background_color=np.array([.12,.14,.16])
        options.mesh_show_back_face=True
        for mesh,label in zip(meshes,('before','after')):
            viewer.add_geometry(mesh,reset_bounding_box=False)
            for _ in range(15):viewer.poll_events();viewer.update_renderer();time.sleep(.03)
            destination=args.after/f'{label}_comparison.png'
            viewer.capture_screen_image(str(destination),do_render=True)
            print(destination.resolve())
            viewer.remove_geometry(mesh,reset_bounding_box=False)
    finally:viewer.destroy_window()


if __name__=='__main__':main()
