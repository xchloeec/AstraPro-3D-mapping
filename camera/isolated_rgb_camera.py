"""Latest-frame UVC bridge keeping native VideoCapture outside OpenNI process."""
from pathlib import Path
import subprocess
import sys
import struct
import threading
import time
import numpy as np


class IsolatedRGBCamera:
    def __init__(self,camera_index=0,backend='DSHOW'):
        self.camera_index=camera_index
        self.active_backend_name=backend
        self.process=None;self.thread=None
        self.condition=threading.Condition()
        self.frame=None;self.sequence=0;self.consumed=0
        self.error=None;self.stopping=False
        self.last_timestamp_ns=None

    def start(self):
        command=[sys.executable,'-u',str(Path(__file__).with_name('rgb_pipe_worker.py')),
                 str(self.camera_index),self.active_backend_name]
        self.process=subprocess.Popen(command,stdout=subprocess.PIPE,bufsize=0,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.thread=threading.Thread(target=self._receive,daemon=True)
        self.thread.start()
        deadline=time.monotonic()+20
        with self.condition:
            while self.frame is None and self.error is None:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise RuntimeError('Isolated UVC startup timed out')
                self.condition.wait(remaining)
            if self.error:raise RuntimeError(str(self.error))
        return self

    def _exact(self,count):
        buffer=bytearray()
        while len(buffer)<count:
            chunk=self.process.stdout.read(count-len(buffer))
            if not chunk:
                code=self.process.wait(timeout=2)
                raise RuntimeError(f'Isolated UVC worker exited ({code}, 0x{code & 0xffffffff:08X})')
            buffer.extend(chunk)
        return buffer

    def _receive(self):
        try:
            while not self.stopping:
                timestamp,width,height=struct.unpack('<QII',self._exact(16))
                if (width,height)!=(640,480):raise RuntimeError('Invalid UVC frame header')
                frame=np.frombuffer(self._exact(width*height*3),dtype=np.uint8).reshape(height,width,3).copy()
                with self.condition:
                    self.frame=frame;self.timestamp=timestamp;self.sequence+=1
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                if not self.stopping:self.error=error
                self.condition.notify_all()

    def read(self):
        deadline=time.monotonic()+5
        with self.condition:
            while self.sequence<=self.consumed and self.error is None and not self.stopping:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise RuntimeError('Isolated UVC read timed out')
                self.condition.wait(remaining)
            if self.error:raise RuntimeError(str(self.error))
            if self.stopping:raise RuntimeError('Isolated UVC closed')
            self.consumed=self.sequence
            self.last_timestamp_ns=self.timestamp
            return self.frame.copy()

    def close(self):
        with self.condition:
            self.stopping=True;self.condition.notify_all()
        if self.process is not None:
            if self.process.poll() is None:self.process.terminate()
            self.process.wait(timeout=5)
        if self.thread is not None:self.thread.join(timeout=2)
        if self.process is not None and self.process.stdout is not None:self.process.stdout.close()
