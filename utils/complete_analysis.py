import cv2
import numpy as np
from typing import Dict, Tuple
from .phone_detector import detect_phone_in_image_enhanced
from .hands_detector import detect_hands_only_enhanced
from .phone_hand_intersections import calculate_phone_hand_intersections

def analyze_phone_usage_complete(image: np.ndarray, phone_confidence: float = 0.2, 
                                hand_confidence: float = 0.5, 
                                min_intersection_area: int = 100,
                                min_overlap_ratio: float = 0.1,
                                debug: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Complete phone usage analysis using constrained detection within person bounding boxes.
    Shows all detection boxes plus intersection indicators.
    
    Args:
        image: Input image
        phone_confidence: Phone detection confidence threshold
        hand_confidence: Hand detection confidence threshold
        min_intersection_area: Minimum intersection area for valid usage
        min_overlap_ratio: Minimum overlap ratio for valid usage
        debug: Show debug information
    
    Returns:
        Tuple of (final_visualization_image, complete_analysis_results)
    """
    if debug:
        print("🚀 CONSTRAINED COMPLETE PHONE USAGE ANALYSIS")
        print("=" * 60)
    
    # Step 1: Detect phones using constrained detection
    if debug:
        print("📱 Step 1: Detecting phones within person bounding boxes...")
    phone_image, phone_boxes = detect_phone_in_image_enhanced(image, phone_confidence, debug)
    
    # Step 2: Detect hands using constrained detection
    if debug:
        print(f"\n👐 Step 2: Detecting hands within person bounding boxes...")
    hand_image, hand_boxes = detect_hands_only_enhanced(image, hand_confidence, debug)
    
    # Step 3: Calculate intersections
    if debug:
        print(f"\n🎯 Step 3: Analyzing intersections...")
    intersection_analysis = calculate_phone_hand_intersections(
        phone_boxes, hand_boxes, min_intersection_area, min_overlap_ratio
    )
    
    # Step 4: Create comprehensive visualization
    final_image = image.copy()
    
    # Draw person bounding boxes in blue (light)
    persons_drawn = set()
    for phone in phone_boxes:
        if 'person_bbox' in phone and phone['person_id'] not in persons_drawn:
            px1, py1, px2, py2 = phone['person_bbox']
            cv2.rectangle(final_image, (px1, py1), (px2, py2), (255, 100, 100), 1)
            cv2.putText(final_image, f"Person {phone['person_id'] + 1}", 
                       (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            persons_drawn.add(phone['person_id'])
    
    for hand in hand_boxes:
        if 'person_bbox' in hand and hand['person_id'] not in persons_drawn:
            px1, py1, px2, py2 = hand['person_bbox']
            cv2.rectangle(final_image, (px1, py1), (px2, py2), (255, 100, 100), 1)
            cv2.putText(final_image, f"Person {hand['person_id'] + 1}", 
                       (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            persons_drawn.add(hand['person_id'])
    
    # Draw phone bounding boxes in green
    for i, phone in enumerate(phone_boxes):
        x1, y1, x2, y2 = phone['bbox']
        cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Enhanced phone label with person association
        person_id = phone.get('person_id', 'Unknown')
        phone_label = f"{phone['class']} ({phone['confidence']:.2f})"
        if person_id != 'Unknown':
            phone_label += f" P{person_id + 1}"
        
        cv2.putText(final_image, phone_label, 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw hand bounding boxes in cyan
    for i, hand in enumerate(hand_boxes):
        x1, y1, x2, y2 = hand['bbox']
        cv2.rectangle(final_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Enhanced hand label with person association
        person_id = hand.get('person_id', 'Unknown')
        hand_label = hand['name']
        if person_id != 'Unknown':
            hand_label += f" P{person_id + 1}"
        
        cv2.putText(final_image, hand_label, 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw intersection areas in bright red
    valid_intersections = intersection_analysis['valid_intersections_only']
    for i, intersection in enumerate(valid_intersections):
        intersection_box = intersection['intersection_details']['intersection_box']
        
        if intersection_box:
            x1, y1, x2, y2 = intersection_box
            # Thick red border for intersections
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Intersection label
            intersection_label = f"USAGE #{i+1}"
            cv2.putText(final_image, intersection_label, 
                       (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add comprehensive status information in top-right corner
    if len(valid_intersections) > 0:
        # Get image dimensions
        img_height, img_width = final_image.shape[:2]
        
        # Count unique persons with phone usage
        unique_persons = set()
        for intersection in valid_intersections:
            phone_id = intersection.get('phone_id', 0)
            if phone_id < len(phone_boxes):
                person_id = phone_boxes[phone_id].get('person_id')
                if person_id is not None:
                    unique_persons.add(person_id)
        
        # Multi-line status text
        status_lines = [
            f"Phone usage detected",
            f"Persons: {len(unique_persons)}",
            f"Intersections: {len(valid_intersections)}"
        ]
        
        # Calculate position for multi-line text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25
        padding = 15
        
        # Get max text width
        max_width = 0
        for line in status_lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])
        
        # Position in top-right corner
        text_x = img_width - max_width - padding
        start_y = padding + 20
        
        # Draw background rectangle for all lines
        bg_height = len(status_lines) * line_height + 10
        cv2.rectangle(final_image, 
                     (text_x - 10, start_y - 20), 
                     (text_x + max_width + 10, start_y + bg_height - 10), 
                     (0, 0, 255), -1)
        
        # Draw each line
        for i, line in enumerate(status_lines):
            y_pos = start_y + i * line_height
            cv2.putText(final_image, line, 
                       (text_x, y_pos), font, font_scale, (255, 255, 255), thickness)
    
    # Compile comprehensive analysis results
    complete_analysis = {
        'persons_detected': len(persons_drawn),
        'phones': phone_boxes,
        'hands': hand_boxes,
        'intersections': intersection_analysis,
        'usage_detected': intersection_analysis['phone_usage_detected'],
        'persons_with_usage': len(set([phone_boxes[intersection.get('phone_id', 0)].get('person_id') 
                                     for intersection in valid_intersections 
                                     if intersection.get('phone_id', 0) < len(phone_boxes)])),
        'summary': f"Analyzed {len(persons_drawn)} person(s): {len(phone_boxes)} phones, {len(hand_boxes)} hands. Found {intersection_analysis['valid_intersections']} valid intersections.",
        'detection_method': 'constrained_person_based'
    }
    
    if debug:
        print(f"\n" + "=" * 60)
        print(f"🎯 CONSTRAINED ANALYSIS COMPLETE:")
        print(f"👥 Persons detected: {len(persons_drawn)}")
        print(f"📱 Phones detected: {len(phone_boxes)}")
        print(f"👐 Hands detected: {len(hand_boxes)}")
        print(f"🔄 Total intersections: {intersection_analysis['total_intersections']}")
        print(f"✅ Valid intersections: {intersection_analysis['valid_intersections']}")
        print(f"🚨 Usage detected: {'YES' if intersection_analysis['phone_usage_detected'] else 'NO'}")
        print(f"📊 Persons with usage: {complete_analysis['persons_with_usage']}")
        print("=" * 60)
    
    return final_image, complete_analysis