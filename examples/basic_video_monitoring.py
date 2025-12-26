"""
Basic Video Monitoring Example
Demonstrates how to process a video file and generate health reports
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
        log_file="logs/video_monitoring.log",
        level="INFO",
        console_output=True
    )

    # Initialize the health monitoring system
    print("Initializing Fish Health Monitoring System...")
    monitor = FishHealthMonitor(
        detector_model_path="yolov8n.pt",  # Will download if not present
        detection_confidence=0.5,
        enable_visualization=True
    )

    # Path to video file
    video_path = "data/raw/sample_aquarium.mp4"

    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        print("Please place a video file at the specified path.")
        return

    # Process the video
    print(f"\nProcessing video: {video_path}")
    print("Press 'q' to stop processing early")

    reports = monitor.process_video(
        video_path=video_path,
        output_path="data/processed/output_video.mp4",
        display=True,
        max_frames=None  # Process entire video
    )

    # Print summary
    print("\n" + "="*60)
    print("HEALTH MONITORING SUMMARY")
    print("="*60)

    summary = monitor.get_health_summary()

    print(f"\nTotal Fish Tracked: {summary['total_fish']}")
    print(f"Average Health Score: {summary['avg_health_score']:.2f}")
    print(f"\nHealth Distribution:")
    print(f"  Healthy: {summary['healthy']}")
    print(f"  Concern: {summary['concern']}")
    print(f"  Warning: {summary['warning']}")
    print(f"  Critical: {summary['critical']}")

    # Print alerts
    alert_summary = summary['active_alerts']
    print(f"\nActive Alerts: {alert_summary['total_active']}")
    print(f"  Critical: {alert_summary['critical']}")
    print(f"  Warning: {alert_summary['warning']}")
    print(f"  Info: {alert_summary['info']}")

    # Print detailed alerts
    active_alerts = monitor.alert_system.get_active_alerts()
    if active_alerts:
        print("\n" + "="*60)
        print("ACTIVE ALERTS")
        print("="*60)

        for alert in active_alerts[:10]:  # Show first 10
            print(f"\n[{alert.alert_level.value.upper()}] {alert.title}")
            print(f"Fish ID: {alert.fish_id}")
            print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Message: {alert.message}")
            if alert.recommendations:
                print("Recommendations:")
                for rec in alert.recommendations[:3]:
                    print(f"  - {rec}")

    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output saved to: data/processed/output_video.mp4")


if __name__ == "__main__":
    main()
