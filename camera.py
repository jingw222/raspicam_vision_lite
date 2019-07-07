import os
import cv2
import time
import numpy as np
import picamera
import picamera.array


class VideoStreamCV2(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)


    def __del__(self):
        self.cap.release()

        
    def get_frame(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            # Encodes image into JPEG in order to correctly display the video stream.
            ret, buf = cv2.imencode('.jpg', frame)
            return frame, buf.tobytes()

        
class VideoStreamPiCam(object):
    def __init__(self):
        self.resolution = (1024, 768)
        self.framerate = 30
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
        next(self.cap)

        self.stream.truncate()
        self.stream.seek(0)
        
        # Reads image from stream.array
        frame = self.stream.array
        ret, buf = cv2.imencode('.jpg', frame)
        return frame, buf.tobytes()
        

