import os
from flask import Flask, render_template, Response
from .camera import VideoStreamCV2, VideoStreamPiCam
from .interpreter import TFLiteInterpreter
from .stream import gen


basepath = os.path.dirname(__file__)

def create_app():
    app = Flask(__name__)
    
    camera = VideoStreamPiCam()

    @app.route('/')
    def index():
        return render_template('index.html')


    @app.route('/videostream/<model_version>', methods=['GET'])
    def videostream(model_version):
        model_path = os.path.join(basepath, 'models', model_version, '{}.tflite'.format(model_version))
        label_path = os.path.join(basepath, 'models', model_version, 'labels.txt')
        model = TFLiteInterpreter(model_path, label_path)
        return Response(gen(camera, model),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return app