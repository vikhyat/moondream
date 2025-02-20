import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

class DeepSORTTracker:
    def __init__(self, max_age=5):
        """Initialize DeepSORT tracker."""
        self.max_age = max_age
        self.tracker = self._create_tracker()
        
    def _create_tracker(self):
        """Create a new instance of DeepSort tracker."""
        return DeepSort(
            max_age=self.max_age,
            embedder='mobilenet',  # Using default MobileNetV2 embedder
            today=datetime.now().date()  # For track naming and daily ID reset
        )
        
    def reset(self):
        """Reset the tracker state by creating a new instance."""
        print("Resetting DeepSORT tracker...")
        self.tracker = self._create_tracker()
        
    def update(self, frame, detections):
        """Update tracking with new detections.
        
        Args:
            frame: Current video frame (numpy array)
            detections: List of (box, keyword) tuples where box is [x1, y1, x2, y2] normalized
            
        Returns:
            List of (box, keyword, track_id) tuples
        """
        if not detections:
            return []
            
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to absolute and format detections
        detection_list = []
        for box, keyword in detections:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            w = x2 - x1
            h = y2 - y1
            
            # Format: ([left,top,w,h], confidence, detection_class)
            detection_list.append(([x1, y1, w, h], 1.0, keyword))
            
        # Update tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Convert back to normalized coordinates with track IDs
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            ltrb = track.to_ltrb()  # Get [left,top,right,bottom] format
            x1, y1, x2, y2 = ltrb
            
            # Normalize coordinates
            x1 = max(0.0, min(1.0, x1 / width))
            y1 = max(0.0, min(1.0, y1 / height))
            x2 = max(0.0, min(1.0, x2 / width))
            y2 = max(0.0, min(1.0, y2 / height))
            
            tracked_objects.append(([x1, y1, x2, y2], track.det_class, track.track_id))
            
        return tracked_objects 