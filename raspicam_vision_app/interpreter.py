import os
import sys
import logging
import time
import cv2
import numpy as np
import tensorflow as tf


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)
logger = logging.getLogger(__name__)

basepath = os.path.dirname(__file__)

def timeit(func):
    def timed(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = '{:0.1f} ms'.format(1000 * (time.perf_counter() - start_time))
        return result, elapsed_time
    return timed


class TFLiteInterpreter(object):
    def __init__(self, target):
        self.target = target
        self.model_path = os.path.join(basepath, 'models', target, '{}.tflite'.format(target))
        self.label_path = os.path.join(basepath, 'models', target, 'labels.txt')
        
        self.interpreter = tf.lite.Interpreter(self.model_path)
        logger.info('Loaded model from file {}'.format(self.model_path))
        
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.IMG_WIDTH, self.IMG_HEIGHT = self.input_details[0].get('shape')[1:3]
        self.DTYPE = self.input_details[0].get('dtype')
        self.QUANT = self.input_details[0].get('quantization')
        logger.debug('Model interpreter details:\n input_details: {}\n output_details: {}'.format(self.input_details, self.output_details))
        
        def load_labels(path):
            with open(path, 'r', newline='\n') as f:
                labels = f.readlines()
                labels = [item.rstrip('\n') for item in labels]
            return labels        
        
        self.labels = load_labels(self.label_path)
        logger.info('Loaded label from file {}'.format(self.label_path))
        
        
    def pre_process(self, x):
        x = cv2.resize(x, (self.IMG_WIDTH, self.IMG_HEIGHT))
        x = np.expand_dims(x, axis=0)
        return x
    
    
    @timeit
    def inference(self, x):
        x = self.pre_process(x)
        self.interpreter.set_tensor(self.input_details[0]['index'], x.astype(self.DTYPE))
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return preds