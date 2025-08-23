from typing import List, Dict, Any
from .box_intersection import calculate_box_intersection

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