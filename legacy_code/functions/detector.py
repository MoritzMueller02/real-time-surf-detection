"""
 Detector of Already PreTrained - Model
        
"""

from ultralytics import YOLO
    
class YoloDetector():
    
    """
        Input:
            - Model Path
            - Confindence for a Prediction to count
        Functions:
            - Detect - returns the model
    """
    
    def __init__(self, model_path, conf):
        self.model = YOLO(model_path)
        self.conf = conf
        
    def detect(self, frame):
        return self.model(frame, conf=self.conf, verbose = False)
        
        