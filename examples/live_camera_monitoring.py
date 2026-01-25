"""
Live Camera Monitoring Example
Demonstrates real-time monitoring from a webcam
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_monitor import FishHealthMonitor
from src.utils.logger import setup_logging


def main():
    # Setup logging
    setup_logging(
        log_file="logs/live_monitoring.log",
        level="INFO",
        console_output=True
    )

    # Initialize the health monitoring system
    print("Initializing Fish Health Monitoring System...")
    monitor = FishHealthMonitor(
        detector_model_path="yolov8n.pt",
        detection_confidence=0.5,
        enable_visualization=True,
        analysis_interval=5,  # Analyze health every 5th frame (3-5x faster!)
        skip_expensive_analysis=True  # Skip eye detection and fin analysis for better FPS
    )

    # Camera settings
    camera_id = 0  # Default camera (0), change if you have multiple cameras

    print(f"\nStarting live monitoring from camera {camera_id}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'a' - Show active alerts")
    print("\nMonitoring starting in 3 seconds...")

    import time
    time.sleep(3)

    try:
        # Start live monitoring
        monitor.process_camera(
            camera_id=camera_id,
            display=True
        )
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
    finally:
        # Print final summary
        print("\n" + "="*60)
        print("MONITORING SESSION SUMMARY")
        print("="*60)

        summary = monitor.get_health_summary()

        print(f"\nTotal Fish Tracked: {summary['total_fish']}")
        print(f"Average Health Score: {summary['avg_health_score']:.2f}")
        print(f"\nHealth Distribution:")
        print(f"  Healthy: {summary['healthy']}")
        print(f"  Concern: {summary['concern']}")
        print(f"  Warning: {summary['warning']}")
        print(f"  Critical: {summary['critical']}")

        alert_summary = summary['active_alerts']
        print(f"\nTotal Alerts Generated: {alert_summary['total_active']}")


if __name__ == "__main__":
    main()
