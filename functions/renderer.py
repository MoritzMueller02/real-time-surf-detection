"""

Renders the Video into a Videoplayer and visualizes the 
tracked entities

"""

import cv2

class Renderer():
    def __init__(self, window_name = "Surf Tracker"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)
        
    def draw_boxes(self, frame, results, class_names):
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = class_names[cls]
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
    def draw_counter(self, frame, counts):
        cv2.putText(
            frame,
            f"Surfers detected: {counts['Surfers']}, Waves detected: {counts['Wave']}, Non-Surfers Detected {counts['Non-Surfers']}", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    def show(self,frame):
        cv2.imshow(self.window_name, frame)
    
                