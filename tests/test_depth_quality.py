import unittest
import numpy as np
from processing.depth_quality import screen_depth


class DepthQualityTests(unittest.TestCase):
    def test_depth_step_does_not_create_intermediate_surface(self):
        depth = np.full((20,20),1.,dtype=np.float32)
        depth[:,10:] = 3.
        original = depth.copy()
        cleaned,stats = screen_depth(depth)
        self.assertTrue(np.all(cleaned[:,9:11] == 0))
        self.assertTrue(np.all(cleaned[:,:9] == 1))
        self.assertTrue(np.all(cleaned[:,11:] == 3))
        self.assertEqual(stats['excluded_pixels'],40)
        np.testing.assert_array_equal(depth,original)

    def test_smooth_plane_preserved_and_missing_depth_not_filled(self):
        depth = np.tile(np.linspace(1.,1.1,30,dtype=np.float32),(30,1))
        cleaned,_ = screen_depth(depth)
        np.testing.assert_array_equal(cleaned,depth)
        depth[10:20,10:20] = 0
        depth[0,0] = np.nan
        cleaned,_ = screen_depth(depth)
        self.assertTrue(np.isfinite(cleaned).all())
        self.assertTrue(np.all(cleaned[10:20,10:20] == 0))

    def test_isolated_measurement_is_not_supported(self):
        depth = np.zeros((20,20),dtype=np.float32)
        depth[10,10] = 2.
        cleaned,stats = screen_depth(depth)
        self.assertFalse(np.any(cleaned))
        self.assertEqual(stats['excluded_pixels'],1)


if __name__ == '__main__':
    unittest.main()
