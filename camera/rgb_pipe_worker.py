"""Standalone UVC process: deliberately never imports OpenNI or camera package."""
import sys
import struct
import time
import cv2


def main():
    index=int(sys.argv[1]);backend_name=sys.argv[2]
    backend=cv2.CAP_DSHOW if backend_name=='DSHOW' else cv2.CAP_MSMF
    capture=cv2.VideoCapture(index,backend)
    try:
        if not capture.isOpened():raise RuntimeError(f'Cannot open UVC camera {index} with {backend_name}')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        while True:
            ok,frame=capture.read()
            timestamp=time.perf_counter_ns()
            if not ok:raise RuntimeError('UVC frame read failed')
            if frame.shape!=(480,640,3):raise RuntimeError(f'Unexpected UVC resolution: {frame.shape}')
            sys.stdout.buffer.write(struct.pack('<QII',timestamp,640,480))
            sys.stdout.buffer.write(frame.tobytes())
            sys.stdout.buffer.flush()
    finally:capture.release()


if __name__=='__main__':main()
