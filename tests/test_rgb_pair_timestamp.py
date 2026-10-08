"""Transport delay must not replace the isolated camera's capture timestamp."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from camera.rgbd_adapter import RGBDFrameAdapter


class RGBPairTimestampTests(unittest.TestCase):
    def receive_once(self, timestamp=None):
        camera=SimpleNamespace(rgb_camera=SimpleNamespace())
        adapter=RGBDFrameAdapter(camera)
        def read():
            adapter._stop_colour_reader.set()
            return np.zeros((2,2,3),dtype=np.uint8)
        camera.rgb_camera.read=read
        if timestamp is not None:camera.rgb_camera.last_timestamp_ns=timestamp
        with patch('camera.rgbd_adapter.time.perf_counter_ns',return_value=999999):
            adapter._colour_reader_loop()
        return adapter._colour_buffer[0][0]

    def test_isolated_read_timestamp_preserved(self):
        self.assertEqual(self.receive_once(123456),123456)

    def test_in_process_camera_keeps_arrival_timestamp(self):
        self.assertEqual(self.receive_once(),999999)


if __name__=='__main__':unittest.main()
