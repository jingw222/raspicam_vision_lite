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
        self.MODEL_PATH = os.path.join(basepath, 'models', target, '{}.tflite'.format(target))
        self.LABEL_PATH = os.path.join(basepath, 'models', target, 'labels.txt')
        
        self.interpreter = tf.lite.Interpreter(self.MODEL_PATH)
        logger.info('Loaded model from file {}'.format(self.MODEL_PATH))
        
        self.interpreter.allocate_tensors()
        
        self.INPUT_DETAILS = self.interpreter.get_input_details()
        self.OUTPUT_DETAILS = self.interpreter.get_output_details()
        self.INPUT_WIDTH, self.INPUT_HEIGHT = self.INPUT_DETAILS[0].get('shape')[1:3]
        self.INPUT_DTYPE = self.INPUT_DETAILS[0].get('dtype')
        self.INPUT_QUANT = self.INPUT_DETAILS[0].get('quantization')
        logger.debug('Model interpreter details:\n input_details: {}\n output_details: {}'.format(self.INPUT_DETAILS, self.OUTPUT_DETAILS))
        
        def load_labels(path):
            with open(path, 'r', newline='\n') as f:
                labels = f.readlines()
                labels = [item.rstrip('\n') for item in labels]
            return labels        
        
        self.labels = load_labels(self.LABEL_PATH)
        logger.info('Loaded label from file {}'.format(self.LABEL_PATH))

        
    def crop_square(self, x):
        h, w, _ = x.shape
        w_new = h
        startw = w//2-(w_new//2)
        return x[:, startw:startw+w_new]
        

    def resize(self, x):
        return cv2.resize(x, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
    
        
    def pre_process(self, x):
        # x = self.crop_square(x)
        x = self.resize(x)
        return np.expand_dims(x, axis=0)
    
    
    @timeit
    def inference(self, x):
        x = self.pre_process(x)
        self.interpreter.set_tensor(self.INPUT_DETAILS[0]['index'], x.astype(self.INPUT_DTYPE))
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.OUTPUT_DETAILS[0]['index'])[0]
        return preds