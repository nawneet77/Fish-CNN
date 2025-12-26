"""
Quick Demo Script - Test Detection Before Fish Tank
Shows what the system detects (spoiler: probably you, not fish!)
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.fish_detector import FishDetector
import torch


def main():
    print("="*70)
    print("FISH DETECTION DEMO - What Does the Generic Model See?")
    print("="*70)

    print("\n⚠️  IMPORTANT: This uses a GENERIC YOLO model")
    print("   It's trained on people, cars, cats - NOT fish!")
    print("   It will probably detect YOU as something 😂")
    print("\n   For actual fish detection, you MUST fine-tune on your tank!")
    print("   See: docs/FINE_TUNING_GUIDE.md\n")

    # Detect device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print(f"Using CPU")

    # Initialize detector
    print("\nInitializing detector...")
    detector = FishDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5,
        device=device
    )

    print("\nFiltered classes (won't detect these):")
    print("  - person, bicycle, car, motorcycle, bus, truck")
    print("\nWhat might be detected:")
    print("  - Anything else the COCO model recognizes")
    print("  - Results will be labeled as 'fish' regardless!")

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n❌ Cannot open camera")
        return

    print("\n" + "="*70)
    print("LIVE DEMO - Press 'q' to quit")
    print("="*70)
    print("\nWhat you'll see:")
    print("  - Green boxes around detected objects")
    print("  - Class name + confidence score")
    print("  - Everything labeled as 'fish' (even if it's not!)")
    print("\nTry:")
    print("  - Moving your hand → might detect it")
    print("  - Showing objects → see what it detects")
    print("  - Pointing at fish tank → probably won't detect actual fish well")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            detections = detector.detect(frame)

            # Draw results
            vis_frame = detector.visualize_detections(
                frame,
                detections,
                show_confidence=True
            )

            # Add info
            info_text = f"Frame: {frame_count} | Detected: {len(detections)} objects"
            cv2.putText(
                vis_frame,
                info_text,
                (10, vis_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Show warning
            cv2.putText(
                vis_frame,
                "GENERIC MODEL - NOT TRAINED ON FISH!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.imshow('Detection Demo - Press Q to quit', vis_frame)

            # Print detections
            if len(detections) > 0 and frame_count % 30 == 0:
                print(f"\nFrame {frame_count}: Detected {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det.class_name} (conf: {det.confidence:.2f})")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)

    print("\n📊 Summary:")
    print(f"  Frames processed: {frame_count}")

    print("\n💡 Next Steps:")
    print("  1. Point camera at your actual fish tank")
    print("  2. Record 5-10 minutes: python scripts/collect_training_data.py")
    print("  3. Annotate fish on Roboflow.com (free)")
    print("  4. Train custom model: python scripts/train_fish_detector.py")
    print("  5. Get accurate fish detection!")

    print("\n📖 Full guide: docs/FINE_TUNING_GUIDE.md")


if __name__ == "__main__":
    main()
