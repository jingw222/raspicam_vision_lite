import cv2
import numpy as np
import tensorflow as tf


class TFLiteInterpreter(object):
    def __init__(self, model_path, labels_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.IMG_WIDTH, self.IMG_HEIGHT = self.input_details[0]['shape'][1:3]
        self.DTYPE = self.input_details[0]['dtype']
        
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
    
    
    def inference(self, image, top=5):
        image = self.pre_process(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image.astype(self.DTYPE))
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        top_indices = preds.argsort()[-top:][::-1]
        result = [(self.labels[i], preds[i]) for i in top_indices] # (labels, scores)
        result.sort(key=lambda x: x[1], reverse=True)
        return result