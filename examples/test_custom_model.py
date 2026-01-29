"""
Quick Test Script for Custom Trained Model
Tests your fish detection model before full health monitoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
from src.detection.fish_detector import FishDetector


def detect_device():
    """Detect best device (MPS for Apple Silicon, CPU for Intel)"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    print("="*70)
    print("CUSTOM MODEL TEST - Quick Validation")
    print("="*70)

    # Detect device
    device = detect_device()
    print(f"\n Device detected: {device.upper()}")
    if device == 'mps':
        print("  ✓ Using Apple Silicon GPU acceleration")
    else:
        print("  Using CPU")

    # Initialize detector with custom model
    print("\nLoading your custom trained model...")
    try:
        detector = FishDetector(
            model_path="models/custom_trained/best.pt",
            confidence_threshold=0.5,
            device=device
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that models/custom_trained/best.pt exists")
        print("  2. Verify it's a valid YOLOv8 model file")
        return

    print(f"\nModel Configuration:")
    print(f"  Device: {detector.device}")
    print(f"  Confidence threshold: {detector.confidence_threshold}")
    print(f"  Filter classes: {detector.filter_classes if detector.filter_classes else 'None (custom model)'}")

    # Test with camera
    print("\n" + "="*70)
    print("STARTING CAMERA TEST")
    print("="*70)
    print("\nOpening camera...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        print("\nTroubleshooting:")
        print("  1. Check camera permissions in System Settings")
        print("  2. Ensure no other app is using the camera")
        return

    print("✓ Camera opened successfully")
    print("\nTest Instructions:")
    print("  • Point camera at your fish tank")
    print("  • Watch for green boxes around detected fish")
    print("  • Press 'q' to quit test")
    print("\nRunning 10-second test...\n")

    frame_count = 0
    detection_count = 0
    confidence_scores = []

    try:
        while frame_count < 300:  # 10 seconds at 30 FPS
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # Detect fish
            detections = detector.detect(frame)
            detection_count += len(detections)

            # Collect confidence scores
            for det in detections:
                confidence_scores.append(det.confidence)

            # Visualize
            vis_frame = detector.visualize_detections(
                frame,
                detections,
                show_confidence=True
            )

            # Add test info
            cv2.putText(
                vis_frame,
                f"Test: Frame {frame_count}/300 | Detected: {len(detections)} fish",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow('Custom Model Test - Press Q to quit', vis_frame)

            # Print detections periodically
            if len(detections) > 0 and frame_count % 30 == 0:
                print(f"✓ Frame {frame_count}: Detected {len(detections)} fish")
                for det in detections:
                    print(f"    - {det.class_name} (confidence: {det.confidence:.3f})")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Test stopped by user")
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    print(f"\n📊 Detection Statistics:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total detections: {detection_count}")

    if frame_count > 0:
        avg_detections = detection_count / frame_count
        print(f"  Average detections/frame: {avg_detections:.2f}")

    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        min_confidence = min(confidence_scores)
        max_confidence = max(confidence_scores)

        print(f"\n📈 Confidence Scores:")
        print(f"  Average: {avg_confidence:.3f}")
        print(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")

        if avg_confidence >= 0.7:
            print("  ✅ Excellent! High confidence detections")
        elif avg_confidence >= 0.5:
            print("  ✅ Good! Reasonable confidence")
        else:
            print("  ⚠️  Low confidence - consider lowering threshold")

    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)

    if detection_count == 0:
        print("\n❌ No fish detected")
        print("\nPossible reasons:")
        print("  1. Camera not pointed at fish tank")
        print("  2. Confidence threshold too high (try 0.3)")
        print("  3. Fish not visible in frame")
        print("  4. Model needs retraining with more data")

        print("\nNext steps:")
        print("  • Point camera at fish tank and run again")
        print("  • Try lower confidence: detector = FishDetector(confidence_threshold=0.3)")

    elif avg_detections < 0.5:
        print("\n⚠️  Few detections")
        print("\nSuggestions:")
        print("  • Lower confidence threshold to 0.3-0.4")
        print("  • Ensure fish are visible and well-lit")
        print("  • Check that camera is focused on fish")

    else:
        print("\n✅ Model is working well!")
        print("\nObservations:")
        if avg_detections >= 1.0:
            print("  ✓ Consistently detecting fish")
        if avg_confidence >= 0.7:
            print("  ✓ High confidence scores")
        if detection_count > frame_count * 0.8:
            print("  ✓ Fish detected in most frames")

        print("\n🎉 Your custom model is ready for full monitoring!")

        print("\nNext steps:")
        print("  1. Run full monitoring:")
        print("     python examples/mac_camera_monitoring.py")
        print("\n  2. (Optional) Calibrate health thresholds:")
        print("     See docs/FINE_TUNING_GUIDE.md - Step 7")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
