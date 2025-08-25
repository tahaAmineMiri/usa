import cv2
import numpy as np
from typing import List, Dict, Tuple
from .phone_hand_intersections import calculate_phone_hand_intersections
from .phone_detector import detect_phone_in_image_enhanced
from .hands_detector import detect_hands_only_enhanced

def detect_intersections_only(image: np.ndarray, phone_confidence: float = 0.2, 
                             hand_confidence: float = 0.5, 
                             min_intersection_area: int = 100,
                             min_overlap_ratio: float = 0.1,
                             debug: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Simplified function that only detects and labels intersections using constrained detection.
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
    if debug:
        print("🎯 Detecting phone-hand intersections...")
    
    # Stage 1: Detect phones using constrained detection
    _, phone_boxes = detect_phone_in_image_enhanced(image, phone_confidence, debug)
    
    # Stage 2: Detect hands using constrained detection  
    _, hand_boxes = detect_hands_only_enhanced(image, hand_confidence, debug)
    
    # Stage 3: Calculate intersections
    intersection_analysis = calculate_phone_hand_intersections(
        phone_boxes, hand_boxes, min_intersection_area, min_overlap_ratio
    )
    
    # Stage 4: Create clean image with only intersection indicators
    labeled_image = image.copy()
    valid_intersections = intersection_analysis['valid_intersections_only']
    
    # Draw intersection boxes only
    for i, intersection in enumerate(valid_intersections):
        intersection_box = intersection['intersection_details']['intersection_box']
        
        if intersection_box:
            x1, y1, x2, y2 = intersection_box
            # Draw bright red intersection box
            cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add intersection label with person association
            person_id = intersection.get('phone_data', {}).get('person_id', 'Unknown')
            label = f"Usage #{i+1}"
            if person_id != 'Unknown':
                label += f" (Person {person_id + 1})"
            
            # Add label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Position label above intersection box
            label_x = max(5, x1)
            label_y = max(text_size[1] + 10, y1 - 10)
            
            # Draw background rectangle for label
            cv2.rectangle(labeled_image, 
                         (label_x - 5, label_y - text_size[1] - 5), 
                         (label_x + text_size[0] + 5, label_y + 5), 
                         (0, 0, 255), -1)
            
            # Draw label text
            cv2.putText(labeled_image, label, 
                       (label_x, label_y), font, font_scale, (255, 255, 255), thickness)
    
    # Add overall status message in top-right corner
    if len(valid_intersections) > 0:
        # Get image dimensions
        img_height, img_width = labeled_image.shape[:2]
        
        # Status text with person count information
        unique_persons = set()
        for intersection in valid_intersections:
            phone_id = intersection.get('phone_id', 0)
            if phone_id < len(phone_boxes):
                person_id = phone_boxes[phone_id].get('person_id')
                if person_id is not None:
                    unique_persons.add(person_id)
        
        if len(unique_persons) > 0:
            status_text = f"Phone usage: {len(unique_persons)} person(s)"
        else:
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
    
    # Prepare enhanced intersection data with person information
    enhanced_intersections = []
    for intersection in valid_intersections:
        enhanced_intersection = intersection.copy()
        
        # Add person information from phone data
        phone_id = intersection.get('phone_id', 0)
        if phone_id < len(phone_boxes):
            phone_data = phone_boxes[phone_id]
            enhanced_intersection['phone_data'] = phone_data
            enhanced_intersection['associated_person'] = phone_data.get('person_id', 'Unknown')
        
        # Add hand information
        hand_id = intersection.get('hand_id', 0)
        if hand_id < len(hand_boxes):
            hand_data = hand_boxes[hand_id]
            enhanced_intersection['hand_data'] = hand_data
        
        enhanced_intersections.append(enhanced_intersection)
    
    if debug:
        unique_persons_count = len(unique_persons) if 'unique_persons' in locals() else 0
        print(f"✅ Found {len(valid_intersections)} valid intersections across {unique_persons_count} person(s)")
    
    return labeled_image, enhanced_intersections