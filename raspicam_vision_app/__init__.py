import os
import sys
import logging
from flask import Flask, Response, render_template, request, session, redirect, url_for
from .camera import VideoStreamCV2, VideoStreamPiCam
from .interpreter import TFLiteInterpreter
from .stream import gen
from config import config


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)

logger = logging.getLogger(__name__)

basepath = os.path.dirname(__file__)

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Builds a camera instance
    camera = VideoStreamPiCam()

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if session.get('candidates') is None:
            model_dir = os.path.join(basepath, 'models')
            candidates = [d for d in next(os.walk(model_dir))[1] if not d.startswith('.')]
            logger.info('Fetched model candidates from directory {}'.format(model_dir))
            session['candidates'] = candidates

        if request.method == 'GET':
            logger.info('Model served: {}'.format(session.get('target')))
            
        if request.method == 'POST':
            logger.info('Request from User-Agent: {}'.format(request.headers.get('User-Agent')))
            target = request.form.get("target")
            if session.get('target') is not None and session.get('target')==target:
                logger.info('Submitted model not changed')
                return '', 204
            else:
                session['target'] = target
                logger.info('Submitted model: {}'.format(target))
        return render_template('index.html', candidates=session.get('candidates'), target=session.get('target'))
    
    
    @app.route('/videostream/<target>', methods=['GET', 'POST'])
    def videostream(target):
        model = TFLiteInterpreter(target)
        return Response(gen(camera, model),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


    def shutdown_server():
        shutdown = request.environ.get('werkzeug.server.shutdown')
        if shutdown is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        shutdown()

        
    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        if request.method == 'POST':
            shutdown_server()
            return 'Server shutting down.'    
    
    
    return app