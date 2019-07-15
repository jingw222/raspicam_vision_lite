import io
import os
import sys
import cv2
import threading
import logging
import picamera
import picamera.array
import numpy as np


WIDTH, HEIGHT = 1024, 768
FRAMERATE = 40

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)
logger = logging.getLogger(__name__)

class VideoStreamCV2(object):
    def __init__(self, **kwargs):
        self.cap = cv2.VideoCapture(0, **kwargs)
        
        # Sets camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FRAMERATE)
        
        # Makes sure video streaming is thread-safe
        self.lock = threading.Lock()
        
        logger.info('OpenCV VideoCapture created.')

        
    def __del__(self):
        self.cap.release()
        logger.info('OpenCV VideoCapture released.')
        
        
    def __iter__(self):
        return self
        
        
    def __next__(self):
        with self.lock:        
            while self.cap.isOpened():
                # Gets frames from camera
                ret, frame = self.cap.read()
                return frame

        
class VideoStreamPiCam(object):
    def __init__(self, **kwargs):
        
        # Initiates a picamera object
        self.camera = picamera.PiCamera(resolution=(WIDTH, HEIGHT), framerate=FRAMERATE, **kwargs)
        logger.info('PiCamera specs: resolution={} framerate={}'.format(self.camera.resolution, self.camera.framerate))
        
        self.stream = picamera.array.PiRGBArray(self.camera)
        
        # Starts preview and prepares to capture images continuously to stream
        self.camera.start_preview()
        logger.info('PiCamera created.')
        
        self.cap = self.camera.capture_continuous(self.stream, format='bgr', use_video_port=True)
        logger.info('Streamer created.')
        
        # Makes sure video streaming is thread-safe
        self.lock = threading.Lock()
        

    def __del__(self):
        self.cap.close()
        self.stream.close()
        logger.info('Streamer closed.')

        self.camera.stop_preview() 
        self.camera.close()
        logger.info('PiCamera closed.')
        
        
    def __iter__(self):
        return self
    
    
    def __next__(self):
        with self.lock:
            while not self.camera.closed:
                self.stream.truncate()
                self.stream.seek(0)
                # Gets a frame as an array from camera
                return next(self.cap).array
    
    
class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = threading.Condition()

        
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

    
    def flush(self):
        logger.info('Buffer flushed.')
    
    
    def close(self):
        self.buffer.close()
        logger.info('Buffer closed.')
        

class VideoStreamCustom(object):
    def __init__(self, **kwargs):
        # Initiates a picamera object
        self.camera = picamera.PiCamera(resolution=(WIDTH, HEIGHT), framerate=FRAMERATE, **kwargs)
        logger.info('PiCamera specs: resolution={} framerate={}'.format(self.camera.resolution, self.camera.framerate))
        
        self.stream = StreamingOutput()
        self.camera.start_recording(self.stream, format='mjpeg')
        

    def __del__(self):
        self.stream.close()
        self.camera.stop_recording() 
        self.camera.close()
        logger.info('PiCamera closed.')
        
        
    def __iter__(self):
        return self
    
    
    def __next__(self):     
        with self.stream.condition:
            while not self.camera.closed:
                self.stream.condition.wait()
                buffer = self.stream.frame
                frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), -1)
                return frame

            