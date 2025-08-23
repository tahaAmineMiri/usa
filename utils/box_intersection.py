from typing import List, Dict, Any

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