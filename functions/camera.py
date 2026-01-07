"""
Docstring for app.camera

This class initiates the contact to the MEO Beachcam  Cameras, where i can acess
the data through the .m8gx links in the network tab
"""

import dotenv
import os
import time
import cv2

class VideoStream():
    
    """
    Functions:
        - read: connects to url
        - reconnect: if no image is returned, it automatically reconnects
        - release: release / terminates the streaming
    
    
    
    """
    def __init__(self, url, retry_delay = 2):
        
        self.url = url
        self.retry_delay = retry_delay
        self.cam = cv2.VideoCapture(self.url)

    def read(self):
        ret, frame = self.cam.read()
        
        if not ret:
            print("Stream interrupted — retrying...")
            self.reconnect()
            return None
        return frame
            
    
    def reconnect(self):
        self.cam.release()
        time.sleep(self.retry_delay)
        self.cam = cv2.VideoCapture(self.url)
    
    def release(self):
        self.cam.release()
        
        
        