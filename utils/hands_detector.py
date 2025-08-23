import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

def detect_hands_only_enhanced(image: np.ndarray, confidence: float = 0.5, debug: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Enhanced hand detection that returns both the visualized image and hand data.
    
    Args:
        image: Input image as numpy array  
        confidence: Detection confidence threshold
        debug: Show debug information
    
    Returns:
        Tuple of (image_with_hand_boxes, hand_detection_data)
    """
    model = YOLO("models/yolo11x-pose.pt")
    results = model(image, conf=confidence, verbose=False)
    
    hand_boxes = []
    image_with_boxes = image.copy()
    
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy
            
            for person_idx, person_keypoints in enumerate(keypoints):
                kpts = person_keypoints.cpu().numpy()
                
                # Check left and right wrists (indices 9, 10)
                hand_indices = [9, 10]
                hand_names = ['LEFT_HAND', 'RIGHT_HAND']
                
                for hand_idx, hand_name in zip(hand_indices, hand_names):
                    if hand_idx < len(kpts):
                        x, y = kpts[hand_idx]
                        
                        if x > 0 and y > 0:  # Valid keypoint
                            box_size = 40
                            hand_data = {
                                'bbox': [int(x - box_size), int(y - box_size), 
                                       int(x + box_size), int(y + box_size)],
                                'name': hand_name,
                                'center': (int(x), int(y)),
                                'person_id': person_idx
                            }
                            hand_boxes.append(hand_data)
                            
                            # Draw hand bounding box
                            cv2.rectangle(image_with_boxes, 
                                        (int(x - box_size), int(y - box_size)),
                                        (int(x + box_size), int(y + box_size)), 
                                        (0, 255, 0), 2)
                            cv2.putText(image_with_boxes, hand_name, 
                                       (int(x - box_size), int(y - box_size) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if debug:
                                print(f"👐 Hand detected: {hand_name}")
                                print(f"   📍 Center: ({int(x)}, {int(y)})")
                                print(f"   📦 Bounding box: ({int(x - box_size)}, {int(y - box_size)}) to ({int(x + box_size)}, {int(y + box_size)})")
    
    if debug:
        print(f"👐 Total hands detected: {len(hand_boxes)}")
    
    return image_with_boxes, hand_boxes