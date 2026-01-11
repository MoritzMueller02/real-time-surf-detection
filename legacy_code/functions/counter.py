"""

Counts a target class

Class = ["Surfer", "Non-Surfer", "Wave"]
{0: 'Non Surfer', 1: 'Surfer', 2: 'wave'}

"""

class SurferCounter():
    
    def __init__(self, target_label):
        self.target_label = target_label
        self.count = 0
        
    def update(self, results):
        self.count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                labels = r.names[cls_id]
                
                if labels == self.target_label:
                    self.count += 1
                    
        return self.count
                
                
                
                
        