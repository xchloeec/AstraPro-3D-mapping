"""Geometric regression checks; no camera required."""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d

from processing.validated_registration import (
    robust_rigid, transform_points, rotation_degrees, make_frame, register_pair,
)
from processing.room_trajectory import optimise_trajectory


class ValidatedRegistrationTests(unittest.TestCase):
    def test_motion_with_wrong_matches(self):
        rng = np.random.default_rng(7)
        source = rng.uniform(-1,1,(120,3)); source[:,2] += 3
        theta = .08
        truth = np.eye(4)
        truth[:3,:3] = [[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]
        truth[:3,3] = [.1,-.02,.04]
        target = transform_points(source,truth) + rng.normal(0,.002,source.shape)
        target[:35] = rng.uniform(-2,2,(35,3))
        estimated,inliers = robust_rigid(source,target)
        self.assertGreaterEqual(inliers.sum(),80)
        self.assertLess(np.linalg.norm(estimated[:3,3]-truth[:3,3]),.01)
        self.assertLess(rotation_degrees(estimated[:3,:3].T@truth[:3,:3]),.5)

    def test_collinear_geometry_is_rejected(self):
        line = np.array([[0.,0,0],[1,0,0],[2,0,0]])
        estimated,_ = robust_rigid(line,line)
        self.assertIsNone(estimated)

    @staticmethod
    def scene():
        color = np.random.default_rng(4).integers(0,256,(240,320,3),dtype=np.uint8)
        y,x = np.mgrid[:240,:320]
        depth = (2+.0008*x+.0005*y).astype(np.float32)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(320,240,285,285,159.5,119.5)
        return color,depth,intrinsic

    @staticmethod
    def frame(color,depth,intrinsic):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),o3d.geometry.Image(depth),
            depth_scale=1,depth_trunc=5,convert_rgb_to_intensity=False,
        )
        return make_frame(rgbd,intrinsic)

    def test_identity_and_wrong_depth_scale(self):
        color,depth,k = self.scene()
        source = self.frame(color,depth,k)
        ok,transform,_,_ = register_pair(source,source)
        self.assertTrue(ok)
        np.testing.assert_allclose(transform,np.eye(4),atol=1e-5)
        wrong = self.frame(color,depth*1.4,k)
        self.assertFalse(register_pair(source,wrong)[0])

    def test_textureless_frame_is_not_claimed_tracked(self):
        color,depth,k = self.scene()
        blank = self.frame(np.zeros_like(color),depth,k)
        self.assertFalse(register_pair(blank,blank)[0])

    def test_verified_revisit_graph_preserves_stationary_scene(self):
        color,depth,k = self.scene()
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset/'color').mkdir(); (dataset/'depth').mkdir()
            o3d.io.write_pinhole_camera_intrinsic(str(dataset/'intrinsic.json'),k)
            for index in range(31):
                cv2.imwrite(str(dataset/'color'/f'{index:05d}.jpg'),cv2.cvtColor(color,cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(dataset/'depth'/f'{index:05d}.png'),(depth*1000).astype(np.uint16))
            trajectory,report = optimise_trajectory(dataset,stride=1)
            self.assertEqual(report['validated_poses'],31)
            self.assertGreater(report['retained_revisit_constraints'],0)
            self.assertTrue(report['pose_graph_optimised'])
            rows = [json.loads(line) for line in (trajectory.parent/'tracking_quality.jsonl').read_text().splitlines()]
            self.assertEqual(rows[10]['reference_node'],0)
            self.assertEqual(rows[20]['reference_node'],15)
            for line in trajectory.read_text().splitlines():
                pose = np.array(line.split()[1:],float).reshape(4,4)
                np.testing.assert_allclose(pose,np.eye(4),atol=1e-5)

    def test_failed_old_keyframe_tries_recent_validated_frame(self):
        color,depth,k = self.scene()
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset/'color').mkdir();(dataset/'depth').mkdir()
            o3d.io.write_pinhole_camera_intrinsic(str(dataset/'intrinsic.json'),k)
            for index in range(3):
                cv2.imwrite(str(dataset/'color'/f'{index:05d}.jpg'),cv2.cvtColor(color,cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(dataset/'depth'/f'{index:05d}.png'),(depth*1000).astype(np.uint16))
            success = (True,np.eye(4),np.eye(6),{'reason':'validated'})
            failure = (False,np.eye(4),np.eye(6),{'reason':'insufficient_features'})
            with patch('processing.room_trajectory.register_pair',side_effect=[success,failure,success]):
                trajectory,report = optimise_trajectory(dataset,stride=1)
            self.assertEqual(report['validated_poses'],3)
            rows=[json.loads(line) for line in (trajectory.parent/'tracking_quality.jsonl').read_text().splitlines()]
            self.assertEqual(rows[2]['reference_node'],1)
            self.assertEqual(rows[2]['reason'],'validated_recent_reference')


if __name__ == '__main__':
    unittest.main()
