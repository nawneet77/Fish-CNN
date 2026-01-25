"""
Mac Camera Monitoring Example
Demonstrates real-time monitoring from Mac camera with proper device detection
"""

import sys
import platform
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import cv2
from src.health_monitor import FishHealthMonitor
from src.utils.logger import setup_logging


def check_mac_camera_permissions():
    """Check if camera is accessible on Mac"""
    print("Checking camera access...")

    cap = cv2.VideoCapture(0)
    is_open = cap.isOpened()
    cap.release()

    if not is_open:
        print("\n⚠️  CAMERA ACCESS DENIED")
        print("\nTo grant camera permissions on macOS:")
        print("1. Go to System Settings → Privacy & Security → Camera")
        print("2. Enable camera access for Terminal (or your IDE)")
        print("3. Restart Terminal/IDE and try again")
        print("\nAlternatively, run this command to reset permissions:")
        print("   tccutil reset Camera")
        print("\nThen restart and grant permissions when prompted.\n")
        return False

    print("✓ Camera access granted")
    return True


def detect_device():
    """Detect best device for Mac (MPS for Apple Silicon, CPU for Intel)"""
    system = platform.system()
    machine = platform.machine()

    print(f"\nSystem: {system}")
    print(f"Architecture: {machine}")

    if system != "Darwin":
        print("Warning: This script is optimized for macOS")
        return "cpu"

    # Check for Apple Silicon (M1/M2/M3)
    if machine == "arm64":
        print("Detected: Apple Silicon Mac")

        # Check if MPS is available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ MPS (Metal Performance Shaders) available")
            return "mps"
        else:
            print("⚠️  MPS not available - using CPU")
            print("Install PyTorch with MPS support:")
            print("   pip install --upgrade torch torchvision torchaudio")
            return "cpu"
    else:
        print("Detected: Intel Mac")
        print("Using CPU (no GPU acceleration on Intel Macs)")
        return "cpu"


def list_available_cameras():
    """List all available cameras on Mac"""
    print("\nScanning for cameras...")
    available_cameras = []

    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it works
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"  Camera {i}: Available ({width}x{height})")
                available_cameras.append(i)
            cap.release()

    if not available_cameras:
        print("  No cameras found")
    else:
        print(f"\nFound {len(available_cameras)} camera(s)")

    return available_cameras


def main():
    print("="*70)
    print("FISH HEALTH MONITORING - Mac Camera Edition")
    print("="*70)

    # Setup logging
    setup_logging(
        log_file="logs/mac_monitoring.log",
        level="INFO",
        console_output=True
    )

    # Check camera permissions (Mac specific)
    if not check_mac_camera_permissions():
        return

    # List available cameras
    cameras = list_available_cameras()
    if not cameras:
        print("\n❌ No cameras detected. Please check your camera connection.")
        return

    # Detect optimal device
    device = detect_device()

    # Select camera
    if len(cameras) == 1:
        camera_id = cameras[0]
        print(f"\nUsing camera {camera_id}")
    else:
        print(f"\nAvailable cameras: {cameras}")
        try:
            camera_input = input(f"Select camera ID [{cameras[0]}]: ").strip()
            camera_id = int(camera_input) if camera_input else cameras[0]
        except ValueError:
            camera_id = cameras[0]
            print(f"Using default camera {camera_id}")

    # Initialize the health monitoring system
    print(f"\nInitializing Fish Health Monitoring System...")
    print(f"Device: {device.upper()}")

    # Use smaller model for better Mac performance
    model_path = "yolov8n.pt"  # Nano model - fastest

    print("\n" + "⚠️ "*35)
    print("IMPORTANT: Generic Model Warning")
    print("⚠️ "*35)
    print("This uses a GENERIC YOLO model (trained on people/cars, NOT fish)!")
    print("It will filter out people, but may still detect random objects.")
    print("\nFor ACTUAL fish detection in your tank:")
    print("  1. Record your tank: python scripts/collect_training_data.py")
    print("  2. Annotate on Roboflow.com (free)")
    print("  3. Train: python scripts/train_fish_detector.py")
    print("\n📖 Full guide: docs/FINE_TUNING_GUIDE.md")
    print("="*70)

    monitor = FishHealthMonitor(
        detector_model_path=model_path,
        detection_confidence=0.6,  # Higher threshold for generic model
        enable_visualization=True,
        device=device,  # Use detected device (mps or cpu)
        analysis_interval=5,  # Analyze health every 5th frame (3-5x faster!)
        skip_expensive_analysis=True  # Skip eye detection and fin analysis for better FPS
    )

    print(f"\n{'='*70}")
    print("STARTING LIVE MONITORING")
    print('='*70)
    print("\nControls:")
    print("  'q' - Quit monitoring")
    print("  's' - Save screenshot")
    print("  'a' - Show active alerts")
    print("  'h' - Show health summary")
    print("\nPositioning tips:")
    print("  • Place camera 12-18 inches from tank")
    print("  • Minimize glare and reflections")
    print("  • Ensure even lighting")
    print("  • Center the tank in view")
    print("\nPress any key in the video window to start...")

    try:
        # Test camera first
        cap = cv2.VideoCapture(camera_id)
        ret, frame = cap.read()

        if not ret:
            print("\n❌ Failed to read from camera. Please check camera connection.")
            cap.release()
            return

        # Show preview
        cv2.imshow('Camera Preview - Press any key to start monitoring', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

        print("\nStarting monitoring in 3 seconds...")
        import time
        time.sleep(3)

        # Start live monitoring
        monitor.process_camera(
            camera_id=camera_id,
            display=True
        )

    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")

    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Print final summary
        print("\n" + "="*70)
        print("MONITORING SESSION SUMMARY")
        print("="*70)

        summary = monitor.get_health_summary()

        print(f"\n📊 Statistics:")
        print(f"  Total Fish Tracked: {summary['total_fish']}")

        if summary['total_fish'] > 0:
            print(f"  Average Health Score: {summary['avg_health_score']:.2f}")
            print(f"\n🏥 Health Distribution:")
            print(f"  Healthy (≥0.7): {summary['healthy']}")
            print(f"  Concern (0.5-0.7): {summary['concern']}")
            print(f"  Warning (0.3-0.5): {summary['warning']}")
            print(f"  Critical (<0.3): {summary['critical']}")

        alert_summary = summary['active_alerts']
        print(f"\n🚨 Alerts Generated:")
        print(f"  Total Active: {alert_summary['total_active']}")
        print(f"  Critical: {alert_summary['critical']}")
        print(f"  Warning: {alert_summary['warning']}")
        print(f"  Info: {alert_summary['info']}")

        # Show top alerts
        active_alerts = monitor.alert_system.get_active_alerts()
        if active_alerts:
            print(f"\n📋 Recent Alerts (showing last 3):")
            for alert in active_alerts[-3:]:
                print(f"\n  [{alert.alert_level.value.upper()}] {alert.title}")
                if alert.recommendations:
                    print(f"    → {alert.recommendations[0]}")

        print("\n" + "="*70)
        print("Thank you for using Fish Health Monitor!")
        print("="*70)


if __name__ == "__main__":
    main()
