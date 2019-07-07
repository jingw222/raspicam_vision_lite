import os
import time
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
from camera import VideoStreamCV2, VideoStreamPiCam
from tflite_model import TFLiteInterpreter


app = Flask(__name__)

MODEL_DIR = 'mobilenet_v2_1.0_224_quant'
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenet_v2_1.0_224_quant.tflite')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels_mobilenet_quant_v2_224.txt')
tflitemodel = TFLiteInterpreter(MODEL_PATH, LABELS_PATH)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame, buffer = camera.get_frame()
        label_index, label = tflitemodel.inference(frame)
        print('Label index: {0}\nLabel name: {1}'.format(label_index, label))
        print('-'*20)
        
        yield (b'--buffer\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n\r\n')

        
@app.route('/videostream')
def videostream():
    return Response(gen(VideoStreamPiCam()),
                    mimetype='multipart/x-mixed-replace; boundary=buffer')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

