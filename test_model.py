# Simple Test Script - Custom Model Only

import cv2
from ultralytics import YOLO

# Load your custom trained model (fixed path)
model = YOLO("./cellphone_dataset_yolov8/runs/detect/yolov8s_optimized/weights/best.pt")

# Load and test on your image
image_path = "./images/remote.png"  # Change this to your image path

try:
    results = model(image_path, conf=0.25)
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Display results
for r in results:
    # Print detections
    if len(r.boxes) > 0:
        print(f"✅ Found {len(r.boxes)} phones:")
        for i, box in enumerate(r.boxes):
            conf = box.conf[0].item()
            print(f"   Phone {i+1}: {conf:.3f} confidence")
    else:
        print("❌ No phones detected")
    
    # Show image with detections
    img = r.plot()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Custom Phone Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()