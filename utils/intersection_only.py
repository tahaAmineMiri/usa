import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from .phone_hand_intersections import calculate_phone_hand_intersections

def detect_intersections_only(image: np.ndarray, phone_confidence: float = 0.2, 
                             hand_confidence: float = 0.5, 
                             min_intersection_area: int = 100,
                             min_overlap_ratio: float = 0.1,
                             debug: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Simplified function that only detects and labels intersections.
    Original image with only intersection boxes and labels overlaid.
    
    Args:
        image: Input image
        phone_confidence: Phone detection confidence threshold
        hand_confidence: Hand detection confidence threshold  
        min_intersection_area: Minimum intersection area for valid usage
        min_overlap_ratio: Minimum overlap ratio for valid usage
        debug: Show debug information
    
    Returns:
        Tuple of (image_with_intersection_labels, list_of_intersections)
    """
    # Get detection data (but don't use the visualization images)
    model_yolo = YOLO("models/yolo11x.pt")
    model_pose = YOLO("models/yolo11x-pose.pt")
    
    # Extract phone data
    phone_results = model_yolo(image, conf=phone_confidence, verbose=False)
    phone_boxes = []
    
    for result in phone_results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model_yolo.names[cls]
                
                if cls in [65, 67]:  # Phone classes
                    phone_boxes.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class': class_name
                    })
    
    # Extract hand data
    hand_results = model_pose(image, conf=hand_confidence, verbose=False)
    hand_boxes = []
    
    for result in hand_results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy
            
            for person_keypoints in keypoints:
                kpts = person_keypoints.cpu().numpy()
                
                hand_indices = [9, 10]
                hand_names = ['LEFT_HAND', 'RIGHT_HAND']
                
                for hand_idx, hand_name in zip(hand_indices, hand_names):
                    if hand_idx < len(kpts):
                        x, y = kpts[hand_idx]
                        
                        if x > 0 and y > 0:
                            box_size = 40
                            hand_boxes.append({
                                'bbox': [int(x - box_size), int(y - box_size), 
                                       int(x + box_size), int(y + box_size)],
                                'name': hand_name,
                                'center': (int(x), int(y))
                            })
    
    # Calculate intersections
    intersection_analysis = calculate_phone_hand_intersections(
        phone_boxes, hand_boxes, min_intersection_area, min_overlap_ratio
    )
    
    # Create clean image with only intersection indicators
    labeled_image = image.copy()
    valid_intersections = intersection_analysis['valid_intersections_only']
    
    # Draw simple intersection boxes without labels
    for i, intersection in enumerate(valid_intersections):
        intersection_box = intersection['intersection_details']['intersection_box']
        
        if intersection_box:
            x1, y1, x2, y2 = intersection_box
            # Draw bright red intersection box (thinner line)
            cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Add simple status message in top-right corner
    if len(valid_intersections) > 0:
        # Get image dimensions
        img_height, img_width = labeled_image.shape[:2]
        
        # Status text
        status_text = "Phone being used"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(status_text, font, font_scale, thickness)[0]
        
        # Position in top-right corner with some padding
        padding = 20
        text_x = img_width - text_size[0] - padding
        text_y = text_size[1] + padding
        
        # Draw background rectangle
        bg_padding = 10
        cv2.rectangle(labeled_image, 
                     (text_x - bg_padding, text_y - text_size[1] - bg_padding), 
                     (text_x + text_size[0] + bg_padding, text_y + bg_padding), 
                     (0, 0, 255), -1)
        
        # Draw text
        cv2.putText(labeled_image, status_text, 
                   (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    if debug:
        print(f"🎯 Intersection-only detection complete:")
        print(f"   📱 Phones found: {len(phone_boxes)}")
        print(f"   👐 Hands found: {len(hand_boxes)}")
        print(f"   ✅ Valid intersections: {len(valid_intersections)}")
    
    return labeled_image, valid_intersections