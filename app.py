from dotenv import load_dotenv
import os
import cv2

from functions.camera import VideoStream
from functions.tracker import ObjectTracker
from functions.counter import SurferCounter
from functions.renderer import Renderer
from functions.recorder import Recorder

# Get that Video
load_dotenv()
url = os.getenv("URL")
print(url)


stream = VideoStream(url)
tracker = ObjectTracker("models/yolo8n_opt.pt", conf= 0.25)
renderer = Renderer()
recorder = Recorder(
    target_class="wave",
    width=1920,
    height=1080,
    conf_threshold=0.2
)
counters = {
        "Surfers": SurferCounter("Surfer"),
        "Non-Surfers": SurferCounter("Non Surfer"),
        "Wave": SurferCounter("wave")    
    }


while True:
    
    frame = stream.read()
    #print(frame.shape[1])
    
    if frame is None:
        stream.reconnect()
        continue
    
    results = tracker.track(frame)
    counts = {name: counter.update(results) for name, counter in counters.items()}
    recorder.update(frame, results)
    renderer.draw_boxes(frame, results, tracker.model.names)
    renderer.draw_counter(frame, counts)
    renderer.show(frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

stream.release()
cv2.destroyAllWindows()




