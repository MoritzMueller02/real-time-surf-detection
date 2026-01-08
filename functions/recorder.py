
import cv2
from datetime import datetime
import os
import time

class Recorder():
    
    """
    Records a video if a wave over a certain confidence level has been tracked
    
    """
    def __init__(self, target_class, width, height, conf_threshold=0.2 ,out_dir="videos", fps=25, record_seconds = 100):
        self.target_class = target_class 
        self.width = width
        self.height = height
        self.fps = fps
        self.out_dir = out_dir
        self.conf_threshold = conf_threshold
        self.record_seconds = record_seconds

        self.writer = None
        self.recording = False
        self.record_start_time = None
        

        os.makedirs(self.out_dir, exist_ok=True)
    
    def update(self, frame, results):
        now = time.time()

        if not self.recording:
            if self._wave_detected(results):
                self._start_recording(frame)
        else:
            elapsed = now - self.record_start_time
            if elapsed >= self.record_seconds:
                self.stop()

        if self.recording:
            self.writer.write(frame)
            
    
    def _wave_detected(self, results):
        
        r = results[0]
        
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            conf = float(box.conf[0])
        
            if label == self.target_class and conf >= self.conf_threshold:
                return True

        return False
        
        
    def _start_recording(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir, f"wave_{timestamp}.mp4")
        
        self.height, self.width = frame.shape[0], frame.shape[1] 

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        self.record_start_time = time.time()
        self.recording = True
        print(f"[REC] Started recording: {path}")

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.recording = False
            print("[REC] Stopped recording")