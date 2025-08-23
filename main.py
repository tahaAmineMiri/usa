import cv2
from utils.intersection_only import detect_intersections_only
from utils.complete_analysis import analyze_phone_usage_complete
from utils.phone_hand_intersections import calculate_phone_hand_intersections
from utils.box_intersection import calculate_box_intersection

def main():
    # Load your image
    image = cv2.imread("./images/usephone.png")
    
    if image is None:
        print("❌ Error: Could not load image. Please check the file path.")
        return
    
    print("🚀 Phone-Hand Intersection Detection System")
    print("=" * 60)
    
    # Method 1: Intersection-Only Detection (Recommended)
    print("🎯 METHOD 1: Intersection-Only Labeling")
    print("-" * 40)
    
    intersection_image, intersections = detect_intersections_only(
        image, 
        phone_confidence=0.2,
        hand_confidence=0.5,
        min_intersection_area=100,
        min_overlap_ratio=0.1,
        debug=True
    )
    
    # Display results
    if intersections:
        print(f"\n✅ INTERSECTIONS DETECTED: {len(intersections)}")
        for i, intersection in enumerate(intersections):
            details = intersection['intersection_details']
            print(f"\n   Intersection #{i+1}:")
            print(f"   📱 Phone: {intersection['phone_class']} (conf: {intersection['phone_confidence']:.3f})")
            print(f"   👐 Hand: {intersection['hand_name']}")
            print(f"   📏 Area: {details['intersection_area']} pixels")
            print(f"   📊 Coverage: Phone {details['overlap_ratio_box1']:.1%}, Hand {details['overlap_ratio_box2']:.1%}")
    else:
        print("\n❌ NO INTERSECTIONS DETECTED")
    
    # Method 2: Complete Analysis (with all detection boxes)
    print(f"\n" + "=" * 60)
    print("🔧 METHOD 2: Complete Analysis (All Boxes + Intersections)")
    print("-" * 40)
    
    complete_image, complete_analysis = analyze_phone_usage_complete(
        image, 
        phone_confidence=0.2,
        hand_confidence=0.5,
        min_intersection_area=100,
        min_overlap_ratio=0.1,
        debug=False  # Reduce console output for this method
    )
    
    print(f"📊 Complete Analysis Summary: {complete_analysis['summary']}")
    
    # Method 3: Custom intersection testing
    print(f"\n" + "=" * 60)
    print("🧪 METHOD 3: Custom Intersection Testing")
    print("-" * 40)
    
    # Test with different parameters
    test_image, test_intersections = detect_intersections_only(
        image, 
        phone_confidence=0.1,        # Lower confidence
        hand_confidence=0.3,         # Lower confidence  
        min_intersection_area=50,    # Smaller area requirement
        min_overlap_ratio=0.05,      # Lower overlap requirement
        debug=False
    )
    
    print(f"🧪 With relaxed parameters:")
    print(f"   Intersections found: {len(test_intersections)}")
    
    # Display results
    print(f"\n" + "=" * 60)
    print("🖼️  DISPLAYING RESULTS")
    print("-" * 40)
    
    # Show intersection-only result (main focus)
    cv2.imshow("Intersection-Only Detection", intersection_image)
    print("📍 Main window: Intersection-Only Detection")
    print("   Press 'n' for next view, 'q' to quit, any other key to continue...")
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q'):
        cv2.destroyAllWindows()
        return
    elif key == ord('n'):
        # Show complete analysis
        cv2.destroyAllWindows()
        cv2.imshow("Complete Analysis", complete_image)
        print("📍 Showing complete analysis with all detection boxes...")
        cv2.waitKey(0)
        
        # Show test results
        cv2.destroyAllWindows()
        cv2.imshow("Relaxed Parameters Test", test_image)
        print("📍 Showing results with relaxed detection parameters...")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # Optional: Save results
    save_results = input("\n💾 Save intersection-only results? (y/n): ").lower().strip()
    if save_results == 'y':
        output_filename = "intersection_only_result.jpg"
        cv2.imwrite(output_filename, intersection_image)
        print(f"✅ Intersection-only result saved to {output_filename}")
        
        # Save intersection report
        with open("intersection_report.txt", "w") as f:
            f.write("PHONE-HAND INTERSECTION DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total intersections found: {len(intersections)}\n")
            
            if intersections:
                f.write("\nDETAILED INTERSECTION DATA:\n")
                for i, intersection in enumerate(intersections):
                    f.write(f"\nIntersection #{i+1}:\n")
                    f.write(f"  Phone: {intersection['phone_class']} (confidence: {intersection['phone_confidence']:.3f})\n")
                    f.write(f"  Hand: {intersection['hand_name']}\n")
                    f.write(f"  Phone bbox: {intersection['phone_bbox']}\n")
                    f.write(f"  Hand bbox: {intersection['hand_bbox']}\n")
                    
                    details = intersection['intersection_details']
                    f.write(f"  Intersection area: {details['intersection_area']} pixels\n")
                    f.write(f"  Phone coverage: {details['overlap_ratio_box1']:.1%}\n")
                    f.write(f"  Hand coverage: {details['overlap_ratio_box2']:.1%}\n")
                    f.write(f"  IoU score: {details['iou']:.3f}\n")
                    f.write(f"  Intersection coordinates: {details['intersection_box']}\n")
            else:
                f.write("\nNo intersections detected.\n")
        
        print("✅ Intersection report saved to intersection_report.txt")

if __name__ == "__main__":
    main()