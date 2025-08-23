import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

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