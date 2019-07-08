import os
import time
import cv2
import numpy as np
import multiprocessing as mp
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

    def inference(frame_in_queue, label_out_queue):
        while True:
            if not frame_in_queue.empty():
                frame = frame_in_queue.get()
                result = tflitemodel.inference(frame)
                for item in result:
                    print('Index: {}, Label: {}, Score: {}'.format(*item))
                print('-'*30)

                label_out_queue.put(result)

    
    frame_in_queue = mp.Queue(maxsize=1)
    label_out_queue = mp.Queue(maxsize=1)
    
    p = mp.Process(target=inference, args=(frame_in_queue, label_out_queue,), daemon=True)
    p.start()
        
    while True:
        frame, buffer = camera.get_frame()
        
        # Set the current frame onto frame_in_queue, if the input queue is empty
        if frame_in_queue.empty():
            frame_in_queue.put(frame)
            
        # Fetch the results from label_out_queue, if the output queue is not empty
        if not label_out_queue.empty():
            result = label_out_queue.get()
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n\r\n' + )

        
@app.route('/videostream')
def videostream():
    return Response(gen(VideoStreamPiCam()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

