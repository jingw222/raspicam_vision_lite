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
        
        def load_labels(path):
            with open(path, 'r', newline='\n') as f:
                labels = f.readlines()
                labels = [item.rstrip('\n') for item in labels]
            return labels        
        
        self.labels = load_labels(labels_path)
        
        
    def pre_process(self, image):
        image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        image = np.expand_dims(image, axis=0)
        return image
    
    def inference(self, image):
        image = self.pre_process(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        label_index = np.argmax(output)
        label = self.labels[label_index]
        return label_index, label