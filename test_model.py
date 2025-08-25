# Constrained Two-Stage Phone Detection with Spatial and Logical Limits
import cv2
from ultralytics import YOLO
import numpy as np

class ConstrainedPhoneDetector:
    def __init__(self, model_path="./models/yolo12x.pt"):
        self.model = YOLO(model_path)
        print(f"🤖 Loaded model: {model_path}")
        
        # Find person and phone class IDs
        self.person_class_id = None
        self.phone_class_id = None
        
        for idx, name in self.model.names.items():
            if name.lower() == 'person':
                self.person_class_id = idx
            elif 'phone' in name.lower() or 'cell' in name.lower():
                self.phone_class_id = idx
        
        print(f"👥 Person class ID: {self.person_class_id}")
        print(f"📱 Phone class ID: {self.phone_class_id} ({self.model.names.get(self.phone_class_id, 'Not found')})")
    
    def detect_phones_constrained(self, image_path, max_phones_per_person=1, debug=True):
        """Two-stage detection with spatial and logical constraints"""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Could not load image")
            return None
        
        h, w = img.shape[:2]
        print(f"\n📸 Image size: {w} x {h}")
        
        # STAGE 1: Detect persons
        print(f"\n🏃 STAGE 1: Detecting persons...")
        person_results = self.detect_persons(img)
        
        if not person_results:
            print("❌ No persons detected!")
            return None
        
        print(f"✅ Found {len(person_results)} person(s)")
        
        # STAGE 2: Detect phones with constraints
        print(f"\n📱 STAGE 2: Detecting phones with constraints...")
        constrained_phone_detections = []
        
        for i, person_box in enumerate(person_results):
            print(f"\n👤 Processing person {i+1}...")
            
            # Get all phone candidates for this person
            phone_candidates = self.get_phone_candidates_for_person(img, person_box, person_id=i+1, debug=debug)
            
            # Apply constraints
            valid_phones = self.apply_constraints(
                phone_candidates, 
                person_box, 
                max_phones_per_person, 
                person_id=i+1, 
                debug=debug
            )
            
            constrained_phone_detections.extend(valid_phones)
        
        # Final result
        final_result = {
            'persons': person_results,
            'phones': constrained_phone_detections,
            'image_shape': (h, w)
        }
        
        if debug:
            self.visualize_constrained_results(img, final_result, image_path)
        
        return final_result
    
    def detect_persons(self, img):
        """Stage 1: Detect all persons in the image"""
        results = self.model(img, 
                           conf=0.3,           
                           iou=0.5,            
                           classes=[self.person_class_id],  
                           verbose=False)
        
        persons = []
        for r in results:
            for box in r.boxes:
                confidence = box.conf[0].item()
                coords = box.xyxy[0].tolist()
                
                persons.append({
                    'confidence': confidence,
                    'coords': coords,
                    'area': (coords[2] - coords[0]) * (coords[3] - coords[1])
                })
                
                print(f"   👤 Person detected: conf={confidence:.3f}")
        
        return persons
    
    def get_phone_candidates_for_person(self, img, person_box, person_id, debug=True):
        """Get all potential phone detections for a person"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in person_box['coords']]
        
        # Create expanded crop for detection (but will filter results later)
        margin = 40
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)  
        crop_x2 = min(w, x2 + margin)
        crop_y2 = min(h, y2 + margin)
        
        person_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if person_crop.size == 0:
            return []
        
        # Multiple detection attempts with very low confidence
        detection_configs = [
            (0.01, 0.3, "Ultra-low confidence"),
            (0.02, 0.3, "Very low confidence"),
            (0.005, 0.2, "Extremely low + strict NMS"),
            (0.01, 0.4, "Ultra-low + loose NMS"),
        ]
        
        all_candidates = []
        
        for conf, iou, desc in detection_configs:
            try:
                results = self.model(person_crop,
                                   conf=conf,
                                   iou=iou,
                                   imgsz=640,
                                   classes=[self.phone_class_id],
                                   verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        confidence = box.conf[0].item()
                        crop_coords = box.xyxy[0].tolist()
                        
                        # Convert to original image coordinates
                        original_coords = [
                            crop_coords[0] + crop_x1,
                            crop_coords[1] + crop_y1,
                            crop_coords[2] + crop_x1,
                            crop_coords[3] + crop_y1
                        ]
                        
                        candidate = {
                            'confidence': confidence,
                            'coords': original_coords,
                            'area': (original_coords[2] - original_coords[0]) * (original_coords[3] - original_coords[1]),
                            'person_id': person_id,
                            'detection_config': desc,
                            'person_box': person_box['coords']
                        }
                        
                        all_candidates.append(candidate)
                        
            except Exception as e:
                if debug:
                    print(f"      {desc} failed: {str(e)[:30]}")
        
        if debug:
            print(f"   🔍 Found {len(all_candidates)} phone candidates before filtering")
        
        return all_candidates
    
    def apply_constraints(self, candidates, person_box, max_phones_per_person, person_id, debug=True):
        """Apply spatial and logical constraints to phone candidates"""
        if not candidates:
            return []
        
        person_coords = person_box['coords']
        px1, py1, px2, py2 = person_coords
        
        # CONSTRAINT 1: Spatial filtering - phones must overlap with person bounding box
        print(f"   🎯 Applying spatial constraints...")
        spatially_valid = []
        
        for candidate in candidates:
            phone_coords = candidate['coords']
            ph_x1, ph_y1, ph_x2, ph_y2 = phone_coords
            
            # Check if phone overlaps with person bounding box
            overlap_x1 = max(px1, ph_x1)
            overlap_y1 = max(py1, ph_y1)
            overlap_x2 = min(px2, ph_x2)
            overlap_y2 = min(py2, ph_y2)
            
            # Calculate overlap area
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                phone_area = candidate['area']
                overlap_ratio = overlap_area / phone_area if phone_area > 0 else 0
                
                # Require at least 30% of phone to be within person box
                if overlap_ratio >= 0.3:
                    candidate['overlap_ratio'] = overlap_ratio
                    spatially_valid.append(candidate)
                    if debug:
                        print(f"      ✅ Phone kept: {overlap_ratio:.2%} overlap, conf={candidate['confidence']:.4f}")
                else:
                    if debug:
                        print(f"      ❌ Phone rejected: {overlap_ratio:.2%} overlap (< 30%)")
            else:
                if debug:
                    print(f"      ❌ Phone rejected: no overlap with person")
        
        print(f"   📊 After spatial filtering: {len(spatially_valid)}/{len(candidates)} phones")
        
        if not spatially_valid:
            return []
        
        # CONSTRAINT 2: Remove duplicate detections (same phone detected multiple times)
        print(f"   🔄 Removing duplicates...")
        unique_phones = self.remove_duplicate_phones(spatially_valid, debug=debug)
        print(f"   📊 After duplicate removal: {len(unique_phones)}/{len(spatially_valid)} phones")
        
        # CONSTRAINT 3: Limit number of phones per person
        print(f"   📏 Applying limit of {max_phones_per_person} phone per person...")
        if len(unique_phones) > max_phones_per_person:
            # Sort by confidence and keep top N
            unique_phones.sort(key=lambda x: x['confidence'], reverse=True)
            limited_phones = unique_phones[:max_phones_per_person]
            
            if debug:
                print(f"      📋 Keeping top {max_phones_per_person} phone by confidence:")
                for i, phone in enumerate(limited_phones):
                    print(f"         {i+1}. Confidence: {phone['confidence']:.4f}")
                print(f"      🗑️ Discarded {len(unique_phones) - max_phones_per_person} lower-confidence detections")
        else:
            limited_phones = unique_phones
        
        print(f"   ✅ Final phones for person {person_id}: {len(limited_phones)}")
        
        return limited_phones
    
    def remove_duplicate_phones(self, phones, iou_threshold=0.5, debug=True):
        """Remove duplicate phone detections using IoU"""
        if len(phones) <= 1:
            return phones
        
        # Sort by confidence (highest first)
        phones.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        for i, phone in enumerate(phones):
            is_duplicate = False
            
            for kept_phone in keep:
                iou = self.calculate_iou(phone['coords'], kept_phone['coords'])
                
                if iou > iou_threshold:
                    is_duplicate = True
                    if debug:
                        print(f"      🔄 Duplicate removed: IoU={iou:.3f} with existing detection")
                    break
            
            if not is_duplicate:
                keep.append(phone)
                if debug:
                    print(f"      ✅ Phone kept: conf={phone['confidence']:.4f}")
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
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
    
    def visualize_constrained_results(self, img, results, image_path):
        """Visualize results with constraint information"""
        vis_img = img.copy()
        
        # Draw persons in blue
        for i, person in enumerate(results['persons']):
            x1, y1, x2, y2 = [int(c) for c in person['coords']]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_img, f"Person {i+1}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw constrained phones in bright green
        for i, phone in enumerate(results['phones']):
            x1, y1, x2, y2 = [int(c) for c in phone['coords']]
            
            # Triple border for constrained phones
            cv2.rectangle(vis_img, (x1-4, y1-4), (x2+4, y2+4), (0, 255, 0), 5)
            cv2.rectangle(vis_img, (x1-1, y1-1), (x2+1, y2+1), (0, 255, 0), 2)
            
            # Add detailed phone label
            cv2.putText(vis_img, f"PHONE {phone['confidence']:.3f}", 
                       (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add person association
            cv2.putText(vis_img, f"Person {phone['person_id']}", 
                       (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            
            # Add overlap info if available
            if 'overlap_ratio' in phone:
                cv2.putText(vis_img, f"{phone['overlap_ratio']:.0%} in person", 
                           (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        
        # Display and save
        cv2.imshow("Constrained Phone Detection", vis_img)
        
        output_path = f"constrained_detection_{len(results['persons'])}p_{len(results['phones'])}ph.jpg"
        cv2.imwrite(output_path, vis_img)
        
        print(f"\n💾 Result saved as: {output_path}")
        print(f"📊 Final Summary: {len(results['persons'])} persons, {len(results['phones'])} phones")
        
        # Print detailed summary
        person_phone_count = {}
        for phone in results['phones']:
            pid = phone['person_id']
            person_phone_count[pid] = person_phone_count.get(pid, 0) + 1
        
        print(f"\n📋 Phones per person (max 1 each):")
        for pid, count in person_phone_count.items():
            print(f"   Person {pid}: {count} phone")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Initialize detector
    detector = ConstrainedPhoneDetector("./models/yolo11x.pt")
    
    # Test image path
    image_path = "./images/usephone2.png"  # Replace with your image path
    
    print("🎯 CONSTRAINED TWO-STAGE PHONE DETECTION")
    print("=" * 60)
    print("Constraints:")
    print("  • Phones must overlap ≥30% with person bounding box")
    print("  • Maximum 1 phone per person")
    print("  • Duplicate removal using IoU")
    
    # Run constrained detection
    results = detector.detect_phones_constrained(
        image_path, 
        max_phones_per_person=1,  # Only 1 phone per person
        debug=True
    )
    
    if results:
        print(f"\n🎉 CONSTRAINED RESULTS:")
        print(f"   👥 Persons detected: {len(results['persons'])}")
        print(f"   📱 Valid phones: {len(results['phones'])}")
        
        # Show breakdown by person
        person_phones = {}
        for phone in results['phones']:
            pid = phone['person_id']
            if pid not in person_phones:
                person_phones[pid] = []
            person_phones[pid].append(phone)
        
        for pid, phones in person_phones.items():
            print(f"\n   Person {pid}: {len(phones)} phone")
            for i, phone in enumerate(phones):
                overlap = phone.get('overlap_ratio', 0) * 100
                print(f"     Phone: conf={phone['confidence']:.4f}, {overlap:.0f}% in person box")
    else:
        print(f"\n❌ No results obtained")

if __name__ == "__main__":
    main()