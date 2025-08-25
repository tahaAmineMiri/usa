import cv2
import os
from utils.video_processor import process_video_intersections

def main():
    """Simplified main function - direct video processing"""
    print("🎥 SINGLE VIDEO PROCESSING")
    print("=" * 50)
    
    # List available videos
    video_dir = "./videos/"
    if not os.path.exists(video_dir):
        print(f"❌ Videos directory not found: {video_dir}")
        print("Please create the videos/ directory and add your video files.")
        return
    
    video_files = [f for f in os.listdir(video_dir) 
                   if f.lower().endswith('.mp4')]
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"📁 Available videos in {video_dir}:")
    for i, video_file in enumerate(video_files):
        print(f"   {i+1}. {video_file}")
    
    # Get user selection
    try:
        choice = int(input(f"\nSelect video (1-{len(video_files)}): ")) - 1
        if choice < 0 or choice >= len(video_files):
            print("❌ Invalid selection")
            return
        
        selected_video = os.path.join(video_dir, video_files[choice])
        print(f"🎬 Selected: {video_files[choice]}")
        
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Only ask about saving the processed video
    save_video = input("Save processed video? (y/n) [default: y]: ").lower().strip()
    save_video = save_video != 'n'
    
    # Process the video with fixed defaults
    print(f"\n🚀 Starting video processing...")
    try:
        result = process_video_intersections(
            video_path=selected_video,
            output_path=None,  # Auto-generate
            save_video=save_video,
            show_progress=True
        )
        
        print(f"\n✅ Video processing completed successfully!")
        
        # Show quick results
        if result['frames_with_usage'] > 0:
            print(f"🎯 Phone usage detected in {result['usage_percentage']:.1f}% of frames")
            print(f"📱 Total intersections found: {result['total_intersections']}")
        else:
            print("📱 No phone usage detected in this video")
            
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")

if __name__ == "__main__":
    main()