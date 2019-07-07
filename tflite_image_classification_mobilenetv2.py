import os
import time
import picamera
import picamera.array
import PIL
import cv2
import numpy as np
import tensorflow as tf


MODEL_DIR = 'mobilenet_v2_1.0_224_quant'
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenet_v2_1.0_224_quant.tflite')
LABEL_PATH = os.path.join(MODEL_DIR, 'labels_mobilenet_quant_v2_224.txt')


with open(LABEL_PATH, 'r', newline='\n') as f:
    labels = f.readlines()
    labels = [item.rstrip('\n') for item in labels]

interpreter = tf.lite.Interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_WIDTH, IMG_HEIGHT = input_details[0]['shape'][1:3]


with picamera.PiCamera(resolution=(1024, 768), framerate=30) as camera:
    camera.start_preview()
    time.sleep(2)
    try:
        with picamera.array.PiRGBArray(camera) as stream:
            for _ in camera.capture_continuous(stream, format='bgr', use_video_port=True):
                # At this point the image is available as stream.array
                stream.truncate()
                stream.seek(0)
                image = stream.array
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                image = np.expand_dims(image, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], image)
                interpreter.invoke()
                res = interpreter.get_tensor(output_details[0]['index'])
                
                label_index = np.argmax(res)
                print('Label index: {0}\nLabel name: {1}'.format(label_index, labels[label_index]))
                print('-'*20)
                
    finally:
        camera.stop_preview()


