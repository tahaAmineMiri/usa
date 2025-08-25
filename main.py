import cv2
from utils.intersection_only import detect_intersections_only
from utils.complete_analysis import analyze_phone_usage_complete
from utils.phone_hand_intersections import calculate_phone_hand_intersections
from utils.box_intersection import calculate_box_intersection

def main():
    # Load your image
    image = cv2.imread("./images/Notusephone.png")
    
    if image is None:
        print("❌ Error: Could not load image. Please check the file path.")
        return
    
    print("🚀 CONSTRAINED Phone-Hand Intersection Detection System")
    print("=" * 70)
    print("🎯 NEW: Detection now happens within person bounding boxes only!")
    print("   • Step 1: Detect persons in image")
    print("   • Step 2: Detect phones only within each person's area")
    print("   • Step 3: Detect hands only within each person's area") 
    print("   • Step 4: Calculate intersections between associated phones/hands")
    print("=" * 70)
    
    # Method 1: Constrained Intersection-Only Detection (Recommended)
    print("🎯 METHOD 1: Constrained Intersection-Only Labeling")
    print("-" * 50)
    
    intersection_image, intersections = detect_intersections_only(
        image, 
        phone_confidence=0.2,
        hand_confidence=0.5,
        min_intersection_area=100,
        min_overlap_ratio=0.1,
        debug=True
    )
    
    # Display results with enhanced information
    if intersections:
        print(f"\n✅ INTERSECTIONS DETECTED: {len(intersections)}")
        
        # Group by person
        person_usage = {}
        for i, intersection in enumerate(intersections):
            person_id = intersection.get('associated_person', 'Unknown')
            if person_id not in person_usage:
                person_usage[person_id] = []
            person_usage[person_id].append(intersection)
        
        for person_id, person_intersections in person_usage.items():
            print(f"\n   👤 Person {person_id + 1 if person_id != 'Unknown' else 'Unknown'}:")
            for i, intersection in enumerate(person_intersections):
                details = intersection['intersection_details']
                print(f"      Intersection #{i+1}:")
                print(f"      📱 Phone: {intersection['phone_class']} (conf: {intersection['phone_confidence']:.3f})")
                print(f"      👐 Hand: {intersection['hand_name']}")
                print(f"      📏 Area: {details['intersection_area']} pixels")
                print(f"      📊 Coverage: Phone {details['overlap_ratio_box1']:.1%}, Hand {details['overlap_ratio_box2']:.1%}")
        
        print(f"\n📊 Summary: {len(person_usage)} person(s) using phones")
    else:
        print("\n❌ NO INTERSECTIONS DETECTED")
    
    # Method 2: Constrained Complete Analysis (with all detection boxes)
    print(f"\n" + "=" * 70)
    print("🔧 METHOD 2: Constrained Complete Analysis (All Boxes + Intersections)")
    print("-" * 50)
    
    complete_image, complete_analysis = analyze_phone_usage_complete(
        image, 
        phone_confidence=0.2,
        hand_confidence=0.5,
        min_intersection_area=100,
        min_overlap_ratio=0.1,
        debug=False  # Reduce console output for this method
    )
    
    print(f"📊 Complete Analysis Summary: {complete_analysis['summary']}")
    print(f"👥 Persons detected: {complete_analysis['persons_detected']}")
    print(f"📱 Phones detected: {len(complete_analysis['phones'])}")
    print(f"👐 Hands detected: {len(complete_analysis['hands'])}")
    print(f"🎯 Persons with usage: {complete_analysis['persons_with_usage']}")
    
    # Method 3: Parameter sensitivity testing
    print(f"\n" + "=" * 70)
    print("🧪 METHOD 3: Parameter Sensitivity Testing")
    print("-" * 50)
    
    # Test with different parameter combinations
    test_configs = [
        {
            'name': 'Very Strict',
            'phone_conf': 0.3, 'hand_conf': 0.6,
            'min_area': 150, 'min_overlap': 0.15
        },
        {
            'name': 'Moderate',
            'phone_conf': 0.2, 'hand_conf': 0.5,
            'min_area': 100, 'min_overlap': 0.1
        },
        {
            'name': 'Relaxed',
            'phone_conf': 0.1, 'hand_conf': 0.3,
            'min_area': 50, 'min_overlap': 0.05
        }
    ]
    
    test_results = []
    for config in test_configs:
        test_image, test_intersections = detect_intersections_only(
            image,
            phone_confidence=config['phone_conf'],
            hand_confidence=config['hand_conf'],
            min_intersection_area=config['min_area'],
            min_overlap_ratio=config['min_overlap'],
            debug=False
        )
        
        test_results.append((config['name'], test_image, test_intersections))
        print(f"   {config['name']:>12}: {len(test_intersections)} intersections")
    
    # Display results
    print(f"\n" + "=" * 70)
    print("🖼️  DISPLAYING RESULTS")
    print("-" * 50)
    
    # Show intersection-only result (main focus)
    cv2.imshow("Constrained Intersection-Only Detection", intersection_image)
    print("📍 Main window: Constrained Intersection-Only Detection")
    print("   Press 'c' for complete analysis, 't' for parameter tests, 'q' to quit, any other key to continue...")
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q'):
        cv2.destroyAllWindows()
        return
    elif key == ord('c'):
        # Show complete analysis
        cv2.destroyAllWindows()
        cv2.imshow("Constrained Complete Analysis", complete_image)
        print("📍 Showing constrained complete analysis with all detection boxes...")
        print("   Press 't' for parameter tests, any other key to continue...")
        
        key2 = cv2.waitKey(0) & 0xFF
        if key2 == ord('t'):
            # Show parameter test results
            for name, test_img, test_intersections in test_results:
                cv2.destroyAllWindows()
                cv2.imshow(f"Parameter Test: {name} ({len(test_intersections)} intersections)", test_img)
                print(f"📍 Showing {name} parameters: {len(test_intersections)} intersections found")
                cv2.waitKey(0)
    elif key == ord('t'):
        # Show parameter test results directly
        for name, test_img, test_intersections in test_results:
            cv2.destroyAllWindows()
            cv2.imshow(f"Parameter Test: {name} ({len(test_intersections)} intersections)", test_img)
            print(f"📍 Showing {name} parameters: {len(test_intersections)} intersections found")
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # Optional: Save results with enhanced reporting
    save_results = input("\n💾 Save constrained intersection results? (y/n): ").lower().strip()
    if save_results == 'y':
        output_filename = "constrained_intersection_result.jpg"
        cv2.imwrite(output_filename, intersection_image)
        print(f"✅ Constrained intersection result saved to {output_filename}")
        
        # Save enhanced intersection report
        with open("constrained_intersection_report.txt", "w") as f:
            f.write("CONSTRAINED PHONE-HAND INTERSECTION DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write("DETECTION METHOD: Person-Constrained Pipeline\n")
            f.write("1. Detect persons in full image\n")
            f.write("2. Detect phones only within person bounding boxes\n")
            f.write("3. Detect hands only within person bounding boxes\n")
            f.write("4. Calculate intersections between associated detections\n\n")
            
            f.write(f"DETECTION PARAMETERS:\n")
            f.write(f"  Phone confidence: 0.2\n")
            f.write(f"  Hand confidence: 0.5\n")
            f.write(f"  Min intersection area: 100 pixels\n")
            f.write(f"  Min overlap ratio: 0.1\n\n")
            
            f.write(f"RESULTS SUMMARY:\n")
            f.write(f"  Total intersections found: {len(intersections)}\n")
            f.write(f"  Persons with phone usage: {len(person_usage) if intersections else 0}\n\n")
            
            if intersections:
                f.write("DETAILED INTERSECTION DATA:\n")
                for person_id, person_intersections in person_usage.items():
                    f.write(f"\nPerson {person_id + 1 if person_id != 'Unknown' else 'Unknown'}:\n")
                    for i, intersection in enumerate(person_intersections):
                        f.write(f"  Intersection #{i+1}:\n")
                        f.write(f"    Phone: {intersection['phone_class']} (confidence: {intersection['phone_confidence']:.3f})\n")
                        f.write(f"    Hand: {intersection['hand_name']}\n")
                        f.write(f"    Phone bbox: {intersection['phone_bbox']}\n")
                        f.write(f"    Hand bbox: {intersection['hand_bbox']}\n")
                        
                        details = intersection['intersection_details']
                        f.write(f"    Intersection area: {details['intersection_area']} pixels\n")
                        f.write(f"    Phone coverage: {details['overlap_ratio_box1']:.1%}\n")
                        f.write(f"    Hand coverage: {details['overlap_ratio_box2']:.1%}\n")
                        f.write(f"    IoU score: {details['iou']:.3f}\n")
                        f.write(f"    Intersection coordinates: {details['intersection_box']}\n")
                        
                        # Add person association info
                        if 'phone_data' in intersection:
                            phone_person_id = intersection['phone_data'].get('person_id', 'Unknown')
                            f.write(f"    Associated with person: {phone_person_id + 1 if phone_person_id != 'Unknown' else 'Unknown'}\n")
            else:
                f.write("No intersections detected using constrained detection.\n")
            
            f.write(f"\nADVANTAGES OF CONSTRAINED DETECTION:\n")
            f.write(f"- Reduces false positives by associating detections with specific persons\n")
            f.write(f"- Improves detection accuracy through focused search areas\n")
            f.write(f"- Enables person-specific phone usage tracking\n")
            f.write(f"- Handles multiple persons in the same image effectively\n")
            f.write(f"- Reduces computational overhead by limiting search space\n")
        
        print("✅ Enhanced constrained intersection report saved to constrained_intersection_report.txt")

if __name__ == "__main__":
    main()