import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Any

def calculate_box_intersection(box1: List[int], box2: List[int]) -> Dict[str, Any]:
    """
    Calculate detailed intersection information between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] - First bounding box coordinates
        box2: [x1, y1, x2, y2] - Second bounding box coordinates
    
    Returns:
        Dictionary containing intersection details:
        - intersects: Boolean indicating if boxes intersect
        - intersection_area: Area of intersection in pixels
        - intersection_box: [x1, y1, x2, y2] of intersection region (None if no intersection)
        - box1_area: Area of first box
        - box2_area: Area of second box  
        - overlap_ratio_box1: Intersection area / box1 area
        - overlap_ratio_box2: Intersection area / box2 area
        - iou: Intersection over Union ratio
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's an intersection
    intersects = x2_i > x1_i and y2_i > y1_i
    
    if intersects:
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        intersection_box = [x1_i, y1_i, x2_i, y2_i]
    else:
        intersection_area = 0
        intersection_box = None
    
    # Calculate box areas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate overlap ratios
    overlap_ratio_box1 = intersection_area / box1_area if box1_area > 0 else 0
    overlap_ratio_box2 = intersection_area / box2_area if box2_area > 0 else 0
    
    # Calculate IoU (Intersection over Union)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return {
        'intersects': intersects,
        'intersection_area': intersection_area,
        'intersection_box': intersection_box,
        'box1_area': box1_area,
        'box2_area': box2_area,
        'overlap_ratio_box1': overlap_ratio_box1,
        'overlap_ratio_box2': overlap_ratio_box2,
        'iou': iou
    }

def calculate_phone_hand_intersections(phone_boxes: List[Dict], hand_boxes: List[Dict], 
                                     min_intersection_area: int = 100,
                                     min_overlap_ratio: float = 0.1) -> Dict[str, Any]:
    """
    Calculate intersections between phone bounding boxes and hand bounding boxes.
    
    Args:
        phone_boxes: List of phone detection dictionaries with 'bbox', 'confidence', 'class'
        hand_boxes: List of hand detection dictionaries with 'bbox', 'name', 'center'
        min_intersection_area: Minimum intersection area to consider as valid usage
        min_overlap_ratio: Minimum overlap ratio to consider as valid usage
    
    Returns:
        Dictionary containing detailed intersection analysis
    """
    intersections = []
    usage_detected = False
    
    print(f"\n🎯 CALCULATING INTERSECTIONS")
    print(f"📱 Phones to analyze: {len(phone_boxes)}")
    print(f"👐 Hands to analyze: {len(hand_boxes)}")
    print("-" * 40)
    
    for phone_idx, phone in enumerate(phone_boxes):
        phone_bbox = phone['bbox']
        
        for hand_idx, hand in enumerate(hand_boxes):
            hand_bbox = hand['bbox']
            
            # Calculate detailed intersection
            intersection_info = calculate_box_intersection(phone_bbox, hand_bbox)
            
            if intersection_info['intersects']:
                intersection_data = {
                    'phone_id': phone_idx,
                    'hand_id': hand_idx,
                    'phone_class': phone['class'],
                    'phone_confidence': phone['confidence'],
                    'hand_name': hand['name'],
                    'phone_bbox': phone_bbox,
                    'hand_bbox': hand_bbox,
                    'intersection_details': intersection_info,
                    'meets_usage_criteria': (
                        intersection_info['intersection_area'] >= min_intersection_area and
                        max(intersection_info['overlap_ratio_box1'], 
                            intersection_info['overlap_ratio_box2']) >= min_overlap_ratio
                    )
                }
                
                intersections.append(intersection_data)
                
                print(f"📍 INTERSECTION FOUND:")
                print(f"   📱 Phone #{phone_idx}: {phone['class']} (conf: {phone['confidence']:.3f})")
                print(f"   👐 Hand #{hand_idx}: {hand['name']}")
                print(f"   📏 Intersection area: {intersection_info['intersection_area']} pixels")
                print(f"   📊 Phone overlap: {intersection_info['overlap_ratio_box1']:.3f}")
                print(f"   📊 Hand overlap: {intersection_info['overlap_ratio_box2']:.3f}")
                print(f"   🎯 IoU: {intersection_info['iou']:.3f}")
                print(f"   ✅ Meets criteria: {intersection_data['meets_usage_criteria']}")
                print("-" * 40)
                
                if intersection_data['meets_usage_criteria']:
                    usage_detected = True
    
    valid_intersections = [i for i in intersections if i['meets_usage_criteria']]
    
    analysis_result = {
        'total_intersections': len(intersections),
        'valid_intersections': len(valid_intersections),
        'phone_usage_detected': usage_detected,
        'intersections': intersections,
        'valid_intersections_only': valid_intersections,
        'phones_analyzed': len(phone_boxes),
        'hands_analyzed': len(hand_boxes),
        'criteria': {
            'min_intersection_area': min_intersection_area,
            'min_overlap_ratio': min_overlap_ratio
        }
    }
    
    print(f"🎯 INTERSECTION ANALYSIS COMPLETE:")
    print(f"   Total intersections found: {len(intersections)}")
    print(f"   Valid intersections (meeting criteria): {len(valid_intersections)}")
    print(f"   Phone usage detected: {'YES' if usage_detected else 'NO'}")
    
    return analysis_result

def detect_phone_in_image_enhanced(image: np.ndarray, confidence: float = 0.5, debug: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Enhanced phone detection that returns both the visualized image and phone data.
    
    Args:
        image: Input image as numpy array
        confidence: Detection confidence threshold
        debug: Show debug information
    
    Returns:
        Tuple of (image_with_phone_boxes, phone_detection_data)
    """
    model = YOLO("models/yolo11x.pt")
    results = model(image, conf=confidence, verbose=False)
    
    phone_boxes = []
    image_with_boxes = image.copy()
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                
                # Check for phone classes (65: remote, 67: cell phone)
                if cls in [65, 67]:
                    phone_data = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class': class_name,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    }
                    phone_boxes.append(phone_data)
                    
                    # Draw phone bounding box
                    cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(image_with_boxes, f"{class_name} ({conf:.2f})", 
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    if debug:
                        print(f"📱 Phone detected: {class_name} (confidence: {conf:.3f})")
                        print(f"   📍 Location: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
    
    if debug:
        print(f"📱 Total phones detected: {len(phone_boxes)}")
    
    return image_with_boxes, phone_boxes

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
    
    # Step 4: Create final visualization - ONLY show intersections
    final_image = image.copy()
    
    # Only draw intersection indicators for valid intersections
    for i, intersection in enumerate(intersection_analysis['valid_intersections_only']):
        intersection_box = intersection['intersection_details']['intersection_box']
        
        if intersection_box:
            x1, y1, x2, y2 = intersection_box
            
            # Draw intersection area in bright red
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add intersection label
            label = f"INTERSECTION #{i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Position label above intersection box
            label_x = max(0, x1)
            label_y = max(label_size[1] + 5, y1 - 5)
            
            # Draw background rectangle for label
            cv2.rectangle(final_image, 
                         (label_x, label_y - label_size[1] - 5), 
                         (label_x + label_size[0], label_y + 5), 
                         (0, 0, 255), -1)
            
            # Draw label text
            cv2.putText(final_image, label, 
                       (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add intersection details as smaller text
            details_text = f"Area: {intersection['intersection_details']['intersection_area']}px"
            cv2.putText(final_image, details_text, 
                       (label_x, label_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add overall status text only if intersections exist
    if intersection_analysis['phone_usage_detected']:
        status_text = f"PHONE USAGE: {intersection_analysis['valid_intersections']} INTERSECTION(S)"
        cv2.putText(final_image, status_text, 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
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
    
    # Create clean image with only intersection labels
    labeled_image = image.copy()
    valid_intersections = intersection_analysis['valid_intersections_only']
    
    for i, intersection in enumerate(valid_intersections):
        intersection_box = intersection['intersection_details']['intersection_box']
        
        if intersection_box:
            x1, y1, x2, y2 = intersection_box
            
            # Draw bright red intersection box
            cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Create intersection label
            label = f"PHONE-HAND INTERSECTION"
            area_text = f"{intersection['intersection_details']['intersection_area']}px"
            
            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            area_size = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            label_x = max(5, x1)
            label_y = max(25, y1 - 10)
            
            # Draw label background
            bg_width = max(label_size[0], area_size[0]) + 10
            bg_height = label_size[1] + area_size[1] + 15
            
            cv2.rectangle(labeled_image, 
                         (label_x - 5, label_y - bg_height), 
                         (label_x + bg_width, label_y + 5), 
                         (0, 0, 255), -1)
            
            # Draw text
            cv2.putText(labeled_image, label, 
                       (label_x, label_y - area_size[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(labeled_image, area_text, 
                       (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if debug:
        print(f"🎯 Intersection-only detection complete:")
        print(f"   📱 Phones found: {len(phone_boxes)}")
        print(f"   👐 Hands found: {len(hand_boxes)}")
        print(f"   ✅ Valid intersections: {len(valid_intersections)}")
    
    return labeled_image, valid_intersections