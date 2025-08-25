import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

def detect_hands_only_enhanced(image: np.ndarray, confidence: float = 0.5, debug: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Constrained hand detection that only looks within person bounding boxes.
    """
    # Import the person detection function from phone_detector
    from .phone_detector import detect_persons_first
    
    # Stage 1: Detect persons
    if debug:
        print("🏃 Stage 1: Detecting persons for hand detection...")
    persons = detect_persons_first(image, 0.3, debug)
    
    if not persons:
        if debug:
            print("❌ No persons detected - no hand detection performed")
        return image.copy(), []
    
    # Stage 2: Detect hands within each person's area
    if debug:
        print(f"\n👐 Stage 2: Detecting hands within {len(persons)} person area(s)...")
    
    model_pose = YOLO("models/yolo11x-pose.pt")
    hand_boxes = []
    image_with_boxes = image.copy()
    h, w = image.shape[:2]
    
    for person_idx, person in enumerate(persons):
        person_bbox = person['bbox']
        x1, y1, x2, y2 = person_bbox
        
        # Create expanded crop for detection
        margin = 60  # Slightly larger margin for hands as they can extend outside body
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(w, x2 + margin)
        crop_y2 = min(h, y2 + margin)
        
        person_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if person_crop.size == 0:
            continue
        
        if debug:
            print(f"   👤 Processing person {person_idx + 1} for hands...")
            print(f"      📦 Person bbox: ({x1},{y1})-({x2},{y2})")
            print(f"      ✂️ Crop region: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
        
        # Multiple detection attempts with different parameters
        detection_configs = [
            (confidence, "Standard confidence"),
            (confidence * 0.7, "Lower confidence"),
            (confidence * 0.5, "Very low confidence")
        ]
        
        person_hand_candidates = []
        
        for conf, desc in detection_configs:
            try:
                results = model_pose(person_crop, conf=conf, verbose=False)
                
                for result in results:
                    if result.keypoints is not None:
                        keypoints = result.keypoints.xy
                        
                        for kp_person_idx, person_keypoints in enumerate(keypoints):
                            kpts = person_keypoints.cpu().numpy()
                            
                            # Check left and right wrists (indices 9, 10)
                            hand_indices = [9, 10]
                            hand_names = ['LEFT_HAND', 'RIGHT_HAND']
                            
                            for hand_idx, hand_name in zip(hand_indices, hand_names):
                                if hand_idx < len(kpts):
                                    crop_x, crop_y = kpts[hand_idx]
                                    
                                    if crop_x > 0 and crop_y > 0:  # Valid keypoint in crop
                                        # Convert coordinates back to original image space
                                        orig_x = crop_x + crop_x1
                                        orig_y = crop_y + crop_y1
                                        
                                        box_size = 40
                                        hand_candidate = {
                                            'bbox': [int(orig_x - box_size), int(orig_y - box_size), 
                                                   int(orig_x + box_size), int(orig_y + box_size)],
                                            'name': hand_name,
                                            'center': (int(orig_x), int(orig_y)),
                                            'person_id': person_idx,
                                            'person_bbox': person_bbox,
                                            'detection_config': desc,
                                            'keypoint_confidence': 1.0  # Pose keypoints don't have individual confidence
                                        }
                                        
                                        person_hand_candidates.append(hand_candidate)
                                        
                                        if debug:
                                            print(f"      👐 Hand candidate: {hand_name} at ({int(orig_x)},{int(orig_y)}) via {desc}")
                
            except Exception as e:
                if debug:
                    print(f"      ❌ {desc} detection failed: {str(e)[:50]}")
        
        # Apply spatial constraints - hands should be reasonably close to person
        valid_hands_for_person = []
        
        for candidate in person_hand_candidates:
            hand_center = candidate['center']
            hx, hy = hand_center
            
            # Check if hand center is within reasonable distance of person bounding box
            # Allow hands to be slightly outside the person box (extended arms)
            extended_margin = 80
            extended_x1 = x1 - extended_margin
            extended_y1 = y1 - extended_margin
            extended_x2 = x2 + extended_margin
            extended_y2 = y2 + extended_margin
            
            if extended_x1 <= hx <= extended_x2 and extended_y1 <= hy <= extended_y2:
                # Calculate distance from hand to person center for additional validation
                person_center_x = (x1 + x2) // 2
                person_center_y = (y1 + y2) // 2
                distance_to_person = ((hx - person_center_x) ** 2 + (hy - person_center_y) ** 2) ** 0.5
                
                # Max reasonable distance is roughly the diagonal of the person box
                person_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                max_hand_distance = person_diagonal * 1.5  # Allow some extension
                
                if distance_to_person <= max_hand_distance:
                    candidate['distance_to_person'] = distance_to_person
                    valid_hands_for_person.append(candidate)
                    
                    if debug:
                        print(f"      ✅ Hand kept: {candidate['name']} at distance {distance_to_person:.0f} (max: {max_hand_distance:.0f})")
                else:
                    if debug:
                        print(f"      ❌ Hand rejected: {candidate['name']} too far ({distance_to_person:.0f} > {max_hand_distance:.0f})")
            else:
                if debug:
                    print(f"      ❌ Hand rejected: {candidate['name']} outside extended person area")
        
        # Remove duplicate hands (same hand detected multiple times)
        unique_hands = remove_duplicate_hands(valid_hands_for_person, debug=debug)
        
        # Limit to 2 hands per person (one left, one right)
        final_hands = limit_hands_per_person(unique_hands, debug=debug)
        
        hand_boxes.extend(final_hands)
        
        if debug:
            print(f"      ✅ Final hands for person {person_idx + 1}: {len(final_hands)}")
    
    # Draw visualization
    for person in persons:
        x1, y1, x2, y2 = person['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for persons
    
    for hand in hand_boxes:
        x1, y1, x2, y2 = hand['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for hands
        cv2.putText(image_with_boxes, hand['name'], 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if debug:
        print(f"👐 Total constrained hands detected: {len(hand_boxes)}")
    
    return image_with_boxes, hand_boxes

def remove_duplicate_hands(hands: List[Dict], distance_threshold: float = 30.0, debug: bool = False) -> List[Dict]:
    """Remove duplicate hand detections based on center distance."""
    if len(hands) <= 1:
        return hands
    
    keep = []
    
    for hand in hands:
        is_duplicate = False
        hx, hy = hand['center']
        
        for kept_hand in keep:
            kx, ky = kept_hand['center']
            distance = ((hx - kx) ** 2 + (hy - ky) ** 2) ** 0.5
            
            # If hands are very close and same type, consider duplicate
            if distance < distance_threshold and hand['name'] == kept_hand['name']:
                is_duplicate = True
                if debug:
                    print(f"         🔄 Duplicate hand removed: {hand['name']} (distance: {distance:.1f})")
                break
        
        if not is_duplicate:
            keep.append(hand)
    
    return keep

def limit_hands_per_person(hands: List[Dict], debug: bool = False) -> List[Dict]:
    """Limit to maximum 1 left hand and 1 right hand per person."""
    left_hands = [h for h in hands if h['name'] == 'LEFT_HAND']
    right_hands = [h for h in hands if h['name'] == 'RIGHT_HAND']
    
    final_hands = []
    
    # Keep best left hand (closest to person center if multiple)
    if left_hands:
        if len(left_hands) > 1:
            left_hands.sort(key=lambda x: x.get('distance_to_person', float('inf')))
            if debug:
                print(f"         📏 Keeping closest left hand (distance: {left_hands[0].get('distance_to_person', 0):.0f})")
        final_hands.append(left_hands[0])
    
    # Keep best right hand (closest to person center if multiple)
    if right_hands:
        if len(right_hands) > 1:
            right_hands.sort(key=lambda x: x.get('distance_to_person', float('inf')))
            if debug:
                print(f"         📏 Keeping closest right hand (distance: {right_hands[0].get('distance_to_person', 0):.0f})")
        final_hands.append(right_hands[0])
    
    return final_hands