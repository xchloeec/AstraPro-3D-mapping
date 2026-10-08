import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import open3d as o3d
from processing.room_layout import extract_layout
from processing.photometric import normalise_gray


class RoomLayoutTests(unittest.TestCase):
    def test_dark_detail_is_lifted_without_changing_depth(self):
        gray = np.tile(np.arange(5,65,dtype=np.uint8),(120,2))
        enhanced = normalise_gray(gray)
        self.assertGreater(np.median(enhanced),np.median(gray))
        self.assertGreater(np.std(enhanced),0)
        self.assertEqual(enhanced.dtype,np.uint8)
        self.assertEqual(enhanced.shape,gray.shape)

    def test_recess_is_not_replaced_by_rectangle(self):
        corners = np.array([[0,0],[4,0],[4,2],[2,2],[2,4],[0,4]],float)
        points = []
        for a,b in zip(corners,np.roll(corners,-1,axis=0)):
            for s in np.linspace(.01,.99,90):
                xy = a*(1-s)+b*s
                for z in np.linspace(.01,2.8,65):
                    points.append([*xy,z])
        for x in np.linspace(.03,3.97,100):
            for y in np.linspace(.03,3.97,100):
                if x < 2 or y < 2:
                    points.append([x,y,0])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            report = extract_layout(cloud,out)
            self.assertTrue(report['floor_detected'])
            self.assertGreaterEqual(len(report['wall_candidates']),6)
            self.assertFalse(report['closed_room_boundary_verified'])
            mesh = o3d.io.read_triangle_mesh(str(out/'layout_surfaces.ply'))
            vertices = np.asarray(mesh.vertices)
            floor = vertices[np.abs(vertices[:,2])<1e-5]
            # Missing upper-right quadrant is a real recess, not floor.
            self.assertFalse(np.any((floor[:,0]>2.2)&(floor[:,1]>2.2)))
            self.assertTrue((out/'floorplan.svg').is_file())


if __name__ == '__main__':
    unittest.main()
