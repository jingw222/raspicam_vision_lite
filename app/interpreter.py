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

def timeit(func):
    def timed(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = '{:0.1f} ms'.format(1000 * (time.perf_counter() - start_time))
        return result, elapsed_time
    return timed


class TFLiteInterpreter(object):
    def __init__(self, model_path, labels_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.IMG_WIDTH, self.IMG_HEIGHT = self.input_details[0].get('shape')[1:3]
        self.DTYPE = self.input_details[0].get('dtype')
        self.QUANT = self.input_details[0].get('quantization')
        
        logger.info('Loaded model from file {}'.format(model_path))
        logger.info('Loaded label from file {}'.format(labels_path))
        logger.info('Model interpreter details:\n input_details: {}\n output_details: {}'.format(self.input_details, self.output_details))
        
        def load_labels(path):
            with open(path, 'r', newline='\n') as f:
                labels = f.readlines()
                labels = [item.rstrip('\n') for item in labels]
            return labels        
        
        self.labels = load_labels(labels_path)
        
        
    def pre_process(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.IMG_WIDTH, self.IMG_HEIGHT))
        x = np.expand_dims(x, axis=0)
        return x
    
    
    @timeit
    def inference(self, image):
        image = self.pre_process(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image.astype(self.DTYPE))
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return preds