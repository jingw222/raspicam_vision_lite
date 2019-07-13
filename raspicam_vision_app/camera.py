import os
import sys
import cv2
import threading
import logging
import picamera
import picamera.array


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
        logger.info('OpenCV VideoCapture destructed.')
        
        
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
        # Sets camera properties
        self.resolution = (WIDTH, HEIGHT)
        self.framerate = FRAMERATE
        
        # Initiates a picamera object
        self.camera = picamera.PiCamera(resolution=self.resolution, framerate=self.framerate, **kwargs)
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
        logger.info('PiCamera destructed.')
        
        
    def __iter__(self):
        return self
    
    
    def __next__(self):
        with self.lock:
            while not self.camera.closed:
                self.stream.truncate()
                self.stream.seek(0)
                # Gets frames from camera
                frame = next(self.cap)
                return frame.array
    