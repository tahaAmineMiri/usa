import cv2
import os
from typing import Dict, Optional
from utils.intersection_only import detect_intersections_only

def process_video_intersections(video_path: str, 
                               output_path: Optional[str] = None,
                               save_video: bool = True,
                               show_progress: bool = True) -> Dict:
    """
    Process a video for phone-hand intersections frame by frame.
    Simplified version with fixed optimal settings.
    """
    
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"{width}x{height} @ {fps}fps, {duration:.1f}s ({total_frames} frames)")
    
    # Prepare output video writer if needed
    video_writer = None
    if save_video:
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"processed_{base_name}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Warning: Could not create output video")
            video_writer = None
            save_video = False
    
    # Initialize counters
    frame_count = 0
    frames_with_usage = 0
    total_intersections = 0
    
    # Process video frame by frame
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Show progress occasionally
            if show_progress and frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processed {frame_count} frames ({progress:.0f}%)")
            
            # Process frame with fixed optimal settings
            processed_frame, intersections = detect_intersections_only(
                frame, 
                phone_confidence=0.2,
                hand_confidence=0.5, 
                min_intersection_area=100,
                min_overlap_ratio=0.1, 
                debug=False
            )
            
            # Track statistics
            if len(intersections) > 0:
                frames_with_usage += 1
                total_intersections += len(intersections)
            
            # Write processed frame to output video
            if save_video and video_writer:
                video_writer.write(processed_frame)
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
    
    # Calculate results
    usage_percentage = (frames_with_usage / frame_count) * 100 if frame_count > 0 else 0
    
    # Print simple summary
    print(f"Analysis complete:")
    print(f"   Usage detected in {frames_with_usage}/{frame_count} frames ({usage_percentage:.1f}%)")
    print(f"   Total intersections: {total_intersections}")
    
    if save_video and video_writer is not None:
        print(f"   Processed video saved: {os.path.basename(output_path)}")
    
    return {
        'frames_with_usage': frames_with_usage,
        'total_frames': frame_count,
        'usage_percentage': usage_percentage,
        'total_intersections': total_intersections,
        'output_path': output_path if save_video else None
    }