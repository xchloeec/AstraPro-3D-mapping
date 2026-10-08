"""Optional Windows UI telemetry must not terminate a reconstruction."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import numpy as np
import open3d as o3d

spec = importlib.util.spec_from_file_location("live_mapper",Path(__file__).resolve().parents[1]/"10_live_mapping_worker.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrackingStatusTests(unittest.TestCase):
    def test_locked_status_preserves_previous_value_and_recovers(self):
        with TemporaryDirectory() as temporary:
            mapper = module.LiveRGBDMapper(Path(temporary))
            self.assertTrue(mapper._write_tracking_status(3,3,True))
            with patch.object(Path,"replace",side_effect=PermissionError("Windows sharing violation")):
                self.assertFalse(mapper._write_tracking_status(6,3,False))
            old = json.loads((Path(temporary)/"tracking_status.json").read_text())
            self.assertEqual(old["processed_frame"],3)
            self.assertFalse((Path(temporary)/"_tracking_status.tmp").exists())
            self.assertTrue(mapper._write_tracking_status(9,9,True))
            self.assertEqual(json.loads((Path(temporary)/"tracking_status.json").read_text())["processed_frame"],9)

    def test_pose_is_checkpointed_before_viewer_exits(self):
        with TemporaryDirectory() as temporary:
            mapper = module.LiveRGBDMapper(Path(temporary))
            mapper._record_pose(12)
            self.assertTrue(mapper.trajectory_path.is_file())
            self.assertTrue(mapper.trajectory_path.read_text().startswith("12 "))

    def test_processing_exception_still_exports_partial_map(self):
        with TemporaryDirectory() as temporary:
            mapper = module.LiveRGBDMapper(Path(temporary))
            intrinsic = o3d.camera.PinholeCameraIntrinsic(20,20,20,20,9.5,9.5)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((20,20,3),dtype=np.uint8)),
                o3d.geometry.Image(np.full((20,20),2000,dtype=np.uint16)),
                depth_scale=1000,depth_trunc=5,convert_rgb_to_intensity=False,
            )
            viewer = Mock(); viewer.poll_events.return_value = True
            with patch.object(mapper,"_wait_for_intrinsic",return_value=intrinsic), \
                 patch.object(mapper,"_create_visualizer",return_value=viewer), \
                 patch.object(mapper,"_find_next_pair",side_effect=[(0,Path("rgb"),Path("depth")),RuntimeError("test processing fault")]), \
                 patch.object(mapper,"_load_rgbd",return_value=rgbd), \
                 patch.object(module,"make_frame",return_value=SimpleNamespace(rgbd=rgbd,intrinsic=intrinsic)):
                self.assertEqual(mapper.run(),1)
            self.assertTrue(mapper.output_path.is_file())
            self.assertTrue(o3d.io.read_point_cloud(str(mapper.output_path)).has_points())
            self.assertTrue(json.loads((Path(temporary)/"live_mapping_summary.json").read_text())["saved_partial_on_error"])

    def _check_reference_poses(self, recent=False):
        with TemporaryDirectory() as temporary:
            mapper = module.LiveRGBDMapper(Path(temporary))
            mapper.finished_flag.write_text('capture finished')
            intrinsic = o3d.camera.PinholeCameraIntrinsic(20,20,20,20,9.5,9.5)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((20,20,3),dtype=np.uint8)),
                o3d.geometry.Image(np.full((20,20),2000,dtype=np.uint16)),
                depth_scale=1000,depth_trunc=5,convert_rgb_to_intensity=False)
            registrations = []
            for distance in (.02,.04,.06):
                transform = np.eye(4);transform[0,3] = distance
                registrations.append((True,transform,np.eye(6),{'reason':'validated'}))
            if recent:
                step=np.eye(4);step[0,3]=.02
                good=(True,step,np.eye(6),{'reason':'validated'})
                registrations=[good,(False,np.eye(4),np.eye(6),{'reason':'insufficient_features'}),good,good]
            viewer = Mock();viewer.poll_events.return_value = True
            pairs = [(index,Path('rgb'),Path('depth')) for index in (0,3,6,9)]+[None]
            with patch.object(mapper,'_wait_for_intrinsic',return_value=intrinsic), \
                 patch.object(mapper,'_create_visualizer',return_value=viewer), \
                 patch.object(mapper,'_find_next_pair',side_effect=pairs), \
                 patch.object(mapper,'_load_rgbd',return_value=rgbd), \
                 patch.object(module,'make_frame',return_value=SimpleNamespace(rgbd=rgbd,intrinsic=intrinsic)), \
                 patch.object(module,'register_pair',side_effect=registrations):
                self.assertEqual(mapper.run(),0)
            poses = [np.array(line.split()[1:],float).reshape(4,4)
                     for line in mapper.trajectory_path.read_text().splitlines()]
            np.testing.assert_allclose([pose[0,3] for pose in poses],[0,-.02,-.04,-.06],atol=1e-8)

    def test_retained_reference_poses_are_not_accumulated_as_step_motion(self):
        self._check_reference_poses()

    def test_recent_reference_fallback_uses_recent_pose(self):
        self._check_reference_poses(recent=True)


if __name__ == "__main__":
    unittest.main()
