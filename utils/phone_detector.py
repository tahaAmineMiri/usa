import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

def detect_persons_first(image: np.ndarray, confidence: float = 0.3, debug: bool = True) -> List[Dict]:
    """
    First stage: Detect all persons in the image.
    """
    model = YOLO("models/yolo11x.pt")
    
    # Find person class ID (usually 0)
    person_class_id = None
    for idx, name in model.names.items():
        if name.lower() == 'person':
            person_class_id = idx
            break
    
    if person_class_id is None:
        if debug:
            print("❌ Person class not found in model!")
        return []
    
    results = model(image, conf=confidence, classes=[person_class_id], verbose=False)
    
    persons = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                person_data = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'area': (x2 - x1) * (y2 - y1),
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                }
                persons.append(person_data)
    
    if debug:
        print(f"👥 Persons detected: {len(persons)}")
    
    return persons

def detect_phone_in_image_enhanced(image: np.ndarray, confidence: float = 0.2, debug: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Constrained phone detection that only looks within person bounding boxes.
    """
    # Stage 1: Detect persons
    persons = detect_persons_first(image, 0.3, debug)
    
    if not persons:
        if debug:
            print("❌ No persons detected - no phone detection performed")
        return image.copy(), []
    
    # Stage 2: Detect phones within each person's area
    model = YOLO("models/yolo11x.pt")
    phone_boxes = []
    image_with_boxes = image.copy()
    h, w = image.shape[:2]
    
    for person_idx, person in enumerate(persons):
        person_bbox = person['bbox']
        x1, y1, x2, y2 = person_bbox
        
        # Create expanded crop for detection
        margin = 40
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(w, x2 + margin)
        crop_y2 = min(h, y2 + margin)
        
        person_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if person_crop.size == 0:
            continue
        
        # Multiple detection attempts with different parameters
        detection_configs = [
            (confidence, 0.3, "Standard"),
            (confidence * 0.5, 0.3, "Lower confidence"),
            (confidence * 0.25, 0.2, "Very low confidence")
        ]
        
        person_phone_candidates = []
        
        for conf, iou, desc in detection_configs:
            try:
                results = model(person_crop, conf=conf, iou=iou, classes=[65, 67], verbose=False)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            crop_x1_box, crop_y1_box, crop_x2_box, crop_y2_box = box.xyxy[0].cpu().numpy()
                            conf_score = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = model.names[cls]
                            
                            # Convert coordinates back to original image space
                            orig_x1 = crop_x1_box + crop_x1
                            orig_y1 = crop_y1_box + crop_y1
                            orig_x2 = crop_x2_box + crop_x1
                            orig_y2 = crop_y2_box + crop_y1
                            
                            phone_candidate = {
                                'bbox': [int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)],
                                'confidence': conf_score,
                                'class': class_name,
                                'center': (int((orig_x1 + orig_x2) / 2), int((orig_y1 + orig_y2) / 2)),
                                'person_id': person_idx,
                                'person_bbox': person_bbox,
                                'detection_config': desc
                            }
                            
                            person_phone_candidates.append(phone_candidate)
                
            except Exception as e:
                continue  # Silently continue on detection failures
        
        # Apply spatial constraints - phones must overlap with person bounding box
        valid_phones_for_person = []
        
        for candidate in person_phone_candidates:
            phone_bbox = candidate['bbox']
            ph_x1, ph_y1, ph_x2, ph_y2 = phone_bbox
            
            # Check overlap with person bounding box
            overlap_x1 = max(x1, ph_x1)
            overlap_y1 = max(y1, ph_y1)
            overlap_x2 = min(x2, ph_x2)
            overlap_y2 = min(y2, ph_y2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                phone_area = (ph_x2 - ph_x1) * (ph_y2 - ph_y1)
                overlap_ratio = overlap_area / phone_area if phone_area > 0 else 0
                
                # Require at least 30% of phone to be within person box
                if overlap_ratio >= 0.3:
                    candidate['overlap_ratio'] = overlap_ratio
                    valid_phones_for_person.append(candidate)
        
        # Remove duplicates using IoU
        unique_phones = remove_duplicate_detections(valid_phones_for_person, debug=False)
        
        # Limit to 1 phone per person (take highest confidence)
        if len(unique_phones) > 1:
            unique_phones.sort(key=lambda x: x['confidence'], reverse=True)
            unique_phones = unique_phones[:1]
        
        phone_boxes.extend(unique_phones)
    
    # Draw visualization
    for person in persons:
        x1, y1, x2, y2 = person['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for persons
    
    for phone in phone_boxes:
        x1, y1, x2, y2 = phone['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for phones
        cv2.putText(image_with_boxes, f"{phone['class']} ({phone['confidence']:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if debug:
        print(f"📱 Phones detected: {len(phone_boxes)} across {len(persons)} person(s)")
    
    return image_with_boxes, phone_boxes

def remove_duplicate_detections(detections: List[Dict], iou_threshold: float = 0.5, debug: bool = False) -> List[Dict]:
    """Remove duplicate detections using IoU threshold."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    for detection in detections:
        is_duplicate = False
        
        for kept_detection in keep:
            iou = calculate_iou(detection['bbox'], kept_detection['bbox'])
            
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(detection)
    
    return keep

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x1_i >= x2_i or y1_i >= y2_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union