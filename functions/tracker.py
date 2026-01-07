"""
    Tracks the surfer or whatever target class
    
"""

from ultralytics import YOLO

class ObjectTracker():
    def __init__(self, model_path, conf = 0.25):
        
        self.model = YOLO(model_path)
        self.conf = conf
    
    def track(self, frame):
        return self.model.track(frame, persist = True, conf = self.conf, verbose = False)
        
        
