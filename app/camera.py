import os
import cv2
import time
import numpy as np
import picamera
import picamera.array


WIDTH, HEIGHT = 1024, 768
FRAMERATE = 40

class VideoStreamCV2(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FRAMERATE)


    def __del__(self):
        self.cap.release()

        
    def get_frame(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame

        
class VideoStreamPiCam(object):
    def __init__(self):
        self.resolution = (WIDTH, HEIGHT)
        self.framerate = FRAMERATE
        self.camera = picamera.PiCamera(resolution=self.resolution, framerate=self.framerate)
        self.stream = picamera.array.PiRGBArray(self.camera)
        
        # Starts preview and prepares to capture images continuously to stream
        self.camera.start_preview()

        self.cap = self.camera.capture_continuous(self.stream, format='bgr', use_video_port=True)

        
    def __del__(self):
        self.stream.close()
        self.camera.stop_preview() 
        self.camera.close()
        
        
    def get_frame(self):
        self.stream.truncate()
        self.stream.seek(0)
        frame = next(self.cap)
        return frame.array
        

