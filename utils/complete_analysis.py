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
    Complete phone usage analysis combining detection and intersection calculation.
    Only shows visual indicators where intersections exist.
    
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
    print("🚀 COMPLETE PHONE USAGE ANALYSIS")
    print("=" * 50)
    
    # Step 1: Detect phones
    print("📱 Step 1: Detecting phones...")
    phone_image, phone_boxes = detect_phone_in_image_enhanced(image, phone_confidence, debug)
    
    # Step 2: Detect hands  
    print("\n👐 Step 2: Detecting hands...")
    hand_image, hand_boxes = detect_hands_only_enhanced(image, hand_confidence, debug)
    
    # Step 3: Calculate intersections
    print("\n🎯 Step 3: Analyzing intersections...")
    intersection_analysis = calculate_phone_hand_intersections(
        phone_boxes, hand_boxes, min_intersection_area, min_overlap_ratio
    )
    
    # Step 4: Create final visualization - simplified display
    final_image = image.copy()
    
    # Draw simple intersection boxes without detailed labels
    for i, intersection in enumerate(intersection_analysis['valid_intersections_only']):
        intersection_box = intersection['intersection_details']['intersection_box']
        
        if intersection_box:
            x1, y1, x2, y2 = intersection_box
            # Draw intersection area in bright red (thinner line)
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Add simple status text in top-right corner only if intersections exist
    if intersection_analysis['phone_usage_detected']:
        # Get image dimensions
        img_height, img_width = final_image.shape[:2]
        
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
        cv2.rectangle(final_image, 
                     (text_x - bg_padding, text_y - text_size[1] - bg_padding), 
                     (text_x + text_size[0] + bg_padding, text_y + bg_padding), 
                     (0, 0, 255), -1)
        
        # Draw text
        cv2.putText(final_image, status_text, 
                   (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Combine all results
    complete_analysis = {
        'phones': phone_boxes,
        'hands': hand_boxes,
        'intersections': intersection_analysis,
        'usage_detected': intersection_analysis['phone_usage_detected'],
        'summary': f"Analyzed {len(phone_boxes)} phones and {len(hand_boxes)} hands. Found {intersection_analysis['valid_intersections']} valid intersections."
    }
    
    print("\n" + "=" * 50)
    print(f"🎯 ANALYSIS COMPLETE:")
    print(f"📱 Phones detected: {len(phone_boxes)}")
    print(f"👐 Hands detected: {len(hand_boxes)}")
    print(f"🔄 Total intersections: {intersection_analysis['total_intersections']}")
    print(f"✅ Valid intersections: {intersection_analysis['valid_intersections']}")
    print(f"🚨 Usage detected: {'YES' if intersection_analysis['phone_usage_detected'] else 'NO'}")
    print("=" * 50)
    
    return final_image, complete_analysis