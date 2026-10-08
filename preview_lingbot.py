"""Render saved neural predictions with the original environment's Open3D."""
from pathlib import Path
import argparse
import time
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence', type=Path)
args = parser.parse_args()
cloud = o3d.io.read_point_cloud(str(args.evidence / 'predicted_cloud.ply'))
if cloud.is_empty():
    raise ValueError('Empty predicted cloud')
# Display uses first-camera up, not a measured gravity or floor alignment.
viewer = o3d.visualization.Visualizer()
if not viewer.create_window(width=1280, height=900, visible=False):
    raise RuntimeError('Unable to create preview renderer')
try:
    viewer.add_geometry(cloud)
    options = viewer.get_render_option()
    options.background_color = np.array([.12, .14, .16])
    options.point_size = 1.5
    control = viewer.get_view_control()
    control.set_lookat(np.median(np.asarray(cloud.points), axis=0))
    control.set_front([.65, -.5, -.65])
    control.set_up([0, -1, 0])
    control.set_zoom(.65)
    for _ in range(20):
        viewer.poll_events()
        viewer.update_renderer()
        time.sleep(.03)
    target = args.evidence / 'predicted_preview.png'
    viewer.capture_screen_image(str(target), do_render=True)
    print(target.resolve())
finally:
    viewer.destroy_window()
