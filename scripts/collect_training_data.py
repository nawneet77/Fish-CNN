"""
Collect Training Data for Fine-Tuning
Records video from camera and extracts frames for annotation
"""

import cv2
import os
import argparse
from pathlib import Path
from datetime import datetime


def record_video(camera_id=0, duration=300, output_dir="data/raw"):
    """
    Record video from camera

    Args:
        camera_id: Camera device ID
        duration: Recording duration in seconds
        output_dir: Output directory for video
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f"training_video_{timestamp}.mp4")

    # Open camera
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Recording Settings:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration} seconds")
    print(f"  Output: {video_path}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"\nRecording started... Press 'q' to stop early")
    print("Tips:")
    print("  • Ensure fish are well-lit and visible")
    print("  • Capture fish in different positions and behaviors")
    print("  • Avoid glare and reflections")
    print("  • Keep camera steady")

    frame_count = 0
    max_frames = duration * fps

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame")
                break

            # Write frame
            out.write(frame)

            # Show preview
            display_frame = frame.copy()
            elapsed = frame_count / fps
            remaining = duration - elapsed

            # Add recording indicator
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(
                display_frame,
                f"REC {int(elapsed)}s / {duration}s",
                (50, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            cv2.imshow('Recording - Press Q to stop', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRecording stopped by user")
                break

            frame_count += 1

            # Print progress every 5 seconds
            if frame_count % (fps * 5) == 0:
                print(f"  Progress: {int(elapsed)}s / {duration}s ({frame_count} frames)")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\n✓ Recording complete!")
    print(f"  Frames recorded: {frame_count}")
    print(f"  Duration: {frame_count / fps:.1f} seconds")
    print(f"  Saved to: {video_path}")

    return video_path


def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from video

    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        frame_rate: Extract 1 frame per N seconds
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_rate)

    print(f"\nExtracting frames...")
    print(f"  Video FPS: {fps}")
    print(f"  Extracting 1 frame every {frame_rate} second(s)")
    print(f"  Frame interval: {frame_interval} frames")

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save frame
            frame_filename = os.path.join(
                output_dir,
                f"frame_{extracted_count:05d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

            if extracted_count % 10 == 0:
                print(f"  Extracted {extracted_count} frames...")

        frame_count += 1

    cap.release()

    print(f"\n✓ Frame extraction complete!")
    print(f"  Total frames in video: {frame_count}")
    print(f"  Frames extracted: {extracted_count}")
    print(f"  Saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review frames in {output_dir}")
    print(f"  2. Delete any blurry or unusable frames")
    print(f"  3. Annotate fish using Roboflow or LabelImg")
    print(f"  4. Follow docs/FINE_TUNING_GUIDE.md for training")


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data for fish detection model"
    )
    parser.add_argument(
        '--mode',
        choices=['record', 'extract', 'both'],
        default='both',
        help='Mode: record video, extract frames, or both'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Recording duration in seconds (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Video file path (for extract mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/frames',
        help='Output directory for frames'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Extract 1 frame per N seconds (default: 1.0)'
    )

    args = parser.parse_args()

    print("="*70)
    print("FISH TRAINING DATA COLLECTION")
    print("="*70)

    if args.mode in ['record', 'both']:
        # Record video
        video_path = record_video(
            camera_id=args.camera,
            duration=args.duration,
            output_dir="data/raw"
        )

        if video_path and args.mode == 'both':
            # Extract frames from recorded video
            extract_frames(video_path, args.output, args.fps)

    elif args.mode == 'extract':
        if not args.video:
            print("Error: --video is required for extract mode")
            return

        # Extract frames from existing video
        extract_frames(args.video, args.output, args.fps)

    print("\n" + "="*70)
    print("Collection complete! See docs/FINE_TUNING_GUIDE.md for next steps.")
    print("="*70)


if __name__ == "__main__":
    main()
