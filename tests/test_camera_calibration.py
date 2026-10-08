"""Check recovery and held-out projection against a known synthetic camera."""
import unittest
import cv2
import numpy as np
from processing.camera_calibration import fit_checkerboard


class CameraCalibrationTests(unittest.TestCase):
    def test_known_camera_and_held_out_views(self):
        rng = np.random.default_rng(17)
        objects = np.zeros((54,3),np.float32)
        objects[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*.018
        matrix = np.array([[570.,0,319.5],[0,575.,239.5],[0,0,1.]])
        distortion = np.array([-.12,.04,.002,-.001,0.])
        views = []
        while len(views) < 35:
            rotation = rng.uniform([-.65,-.65,-.4],[.65,.65,.4])
            translation = rng.uniform([-.18,-.14,.35],[.09,.08,.85])
            points,_ = cv2.projectPoints(objects,rotation,translation,matrix,distortion)
            if (points[:,:,0].min()>5 and points[:,:,0].max()<635 and
                    points[:,:,1].min()>5 and points[:,:,1].max()<475):
                views.append(points.astype(np.float32)+rng.normal(0,.08,points.shape).astype(np.float32))
        result = fit_checkerboard(views[:30],(640,480),18.)
        fitted = np.asarray(result['camera_matrix'])
        coefficients = np.asarray(result['distortion_coefficients'])
        self.assertLess(abs(fitted[0,0]-570),3)
        self.assertLess(abs(fitted[1,1]-575),3)
        self.assertLess(result['rms_pixels'],.2)
        for held_out in views[30:]:
            ok,rotation,translation = cv2.solvePnP(objects,held_out,fitted,coefficients)
            self.assertTrue(ok)
            projected,_ = cv2.projectPoints(objects,rotation,translation,fitted,coefficients)
            self.assertLess(np.sqrt(np.mean(np.sum((projected-held_out)**2,axis=2))),.25)

    def test_too_few_views_rejected(self):
        with self.assertRaises(ValueError):
            fit_checkerboard([], (640,480),18.)


if __name__ == '__main__': unittest.main()
