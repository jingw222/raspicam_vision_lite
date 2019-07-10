import os
from flask import Flask, Response, render_template, request, session, redirect, url_for
from .camera import VideoStreamCV2, VideoStreamPiCam
from .interpreter import TFLiteInterpreter
from .stream import gen


basepath = os.path.dirname(__file__)

def create_app():
    app = Flask(__name__)
    
    camera = VideoStreamPiCam()

    @app.route('/', methods=['GET', 'POST'])
    def index():
        candidates = [d for d in next(os.walk(os.path.join(basepath, 'models')))[1] if not d.startswith('.')]
        target = request.form.get("candidates")
        print('selected target: {}'.format(target))
        return render_template('index.html', candidates=candidates, target=target)


    @app.route('/videostream/<target>', methods=['GET', 'POST'])
    def videostream(target):
        model_path = os.path.join(basepath, 'models', target, '{}.tflite'.format(target))
        label_path = os.path.join(basepath, 'models', target, 'labels.txt')
        model = TFLiteInterpreter(model_path, label_path)
        return Response(gen(camera, model),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return app