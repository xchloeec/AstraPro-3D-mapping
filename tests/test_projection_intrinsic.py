"""A calibrated fusion must use the projection fitted by its trajectory."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import open3d as o3d
from processing.projection_intrinsic import load_projection


class ProjectionTests(unittest.TestCase):
    def test_old_poses_cannot_use_new_intrinsics(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            old=o3d.camera.PinholeCameraIntrinsic(640,480,570,570,319.5,239.5)
            new=o3d.camera.PinholeCameraIntrinsic(640,480,600,599,332.6,242.1)
            o3d.io.write_pinhole_camera_intrinsic(str(folder/'intrinsic.json'),old)
            o3d.io.write_pinhole_camera_intrinsic(str(folder/'new.json'),new)
            with self.assertRaisesRegex(ValueError,'recompute trajectory'):
                load_projection(folder,folder/'new.json',folder/'live_trajectory.txt')

    def test_saved_trajectory_projection_used_without_override(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary);poses=folder/'refinement';poses.mkdir()
            old=o3d.camera.PinholeCameraIntrinsic(640,480,570,570,319.5,239.5)
            new=o3d.camera.PinholeCameraIntrinsic(640,480,600,599,332.6,242.1)
            o3d.io.write_pinhole_camera_intrinsic(str(folder/'intrinsic.json'),old)
            o3d.io.write_pinhole_camera_intrinsic(str(poses/'intrinsic_used.json'),new)
            selected,_=load_projection(folder,trajectory=poses/'trajectory.txt')
            self.assertAlmostEqual(selected.intrinsic_matrix[0,0],600)

    def test_resolution_mismatch_rejected(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            o3d.io.write_pinhole_camera_intrinsic(str(folder/'intrinsic.json'),o3d.camera.PinholeCameraIntrinsic(640,480,570,570,319.5,239.5))
            o3d.io.write_pinhole_camera_intrinsic(str(folder/'wrong.json'),o3d.camera.PinholeCameraIntrinsic(320,240,300,300,159.5,119.5))
            with self.assertRaisesRegex(ValueError,'resolution'):
                load_projection(folder,folder/'wrong.json')


if __name__=='__main__':unittest.main()
