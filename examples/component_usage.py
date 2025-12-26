"""
Component Usage Example
Demonstrates how to use individual components separately
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.fish_detector import FishDetector
from src.tracking.fish_tracker import FishTracker
from src.health_assessment.visual_analyzer import VisualHealthAnalyzer
from src.behavior_analysis.behavior_analyzer import BehaviorAnalyzer
from src.alerts.alert_system import AlertSystem, AlertType, AlertLevel


def example_detection():
    """Example: Using the fish detector"""
    print("\n" + "="*60)
    print("FISH DETECTION EXAMPLE")
    print("="*60)

    # Initialize detector
    detector = FishDetector(model_path="yolov8n.pt", confidence_threshold=0.5)

    # Load sample image
    image_path = "data/raw/sample_frame.jpg"

    if Path(image_path).exists():
        frame = cv2.imread(image_path)

        # Detect fish
        detections = detector.detect(frame)

        print(f"\nDetected {len(detections)} fish")

        for i, det in enumerate(detections):
            print(f"\nFish {i+1}:")
            print(f"  Bounding Box: {det.bbox}")
            print(f"  Confidence: {det.confidence:.2f}")
            print(f"  Center: {det.center}")
            print(f"  Area: {det.area} pixels")

        # Visualize detections
        vis_frame = detector.visualize_detections(frame, detections)
        cv2.imwrite("data/processed/detections.jpg", vis_frame)
        print(f"\nVisualization saved to: data/processed/detections.jpg")
    else:
        print(f"Sample image not found: {image_path}")


def example_tracking():
    """Example: Using the fish tracker"""
    print("\n" + "="*60)
    print("FISH TRACKING EXAMPLE")
    print("="*60)

    # Initialize detector and tracker
    detector = FishDetector(model_path="yolov8n.pt")
    tracker = FishTracker(max_age=30, min_hits=3)

    # Create dummy video frames (in practice, read from video)
    print("\nTracking fish across frames...")

    # Simulate processing multiple frames
    for frame_idx in range(10):
        # In practice: frame = cv2.imread(f"frame_{frame_idx}.jpg")
        # For demo, we'll just show the concept

        # Detect fish in frame
        # detections = detector.detect(frame)

        # Update tracker
        # tracks = tracker.update(detections)

        print(f"Frame {frame_idx}: Tracking would update here")

    print("\nTracking complete!")
    active_tracks = tracker.get_active_tracks()
    print(f"Active tracks: {len(active_tracks)}")


def example_health_analysis():
    """Example: Using health analyzers"""
    print("\n" + "="*60)
    print("HEALTH ANALYSIS EXAMPLE")
    print("="*60)

    # Initialize analyzers
    visual_analyzer = VisualHealthAnalyzer()
    behavior_analyzer = BehaviorAnalyzer()

    # Create dummy data for demonstration
    print("\nAnalyzing visual health...")

    # In practice: analyze real fish ROI
    # visual_health = visual_analyzer.analyze(frame, bbox)

    print("Visual health metrics:")
    print("  - Color score")
    print("  - Texture score")
    print("  - Body condition score")
    print("  - Fin condition score")
    print("  - Eye clarity score")

    print("\nAnalyzing behavioral patterns...")

    # In practice: analyze real position history
    position_history = [(100 + i*5, 200 + i*2) for i in range(50)]
    bbox_history = [(90, 190, 110, 210) for _ in range(50)]

    behavior_metrics = behavior_analyzer.analyze(
        position_history,
        bbox_history,
        other_fish_positions=[(300, 200), (400, 250)]
    )

    print(f"\nBehavior metrics:")
    print(f"  Swimming speed: {behavior_metrics.swimming_speed:.2f} px/frame")
    print(f"  Pattern score: {behavior_metrics.swimming_pattern_score:.2f}")
    print(f"  Activity score: {behavior_metrics.activity_level_score:.2f}")
    print(f"  Isolation score: {behavior_metrics.isolation_score:.2f}")
    print(f"  Overall behavior: {behavior_metrics.overall_behavior_score:.2f}")
    print(f"  Status: {behavior_metrics.behavior_status.value}")


def example_alert_system():
    """Example: Using the alert system"""
    print("\n" + "="*60)
    print("ALERT SYSTEM EXAMPLE")
    print("="*60)

    # Initialize alert system
    alert_system = AlertSystem(db_path="data/example_alerts.db")

    # Generate sample alerts
    alert1 = alert_system.generate_alert(
        alert_type=AlertType.VISUAL_HEALTH,
        alert_level=AlertLevel.WARNING,
        title="Visual Health Concern - Fish #1",
        message="Fish shows signs of faded coloration",
        fish_id=1,
        metrics={'color_score': 0.45},
        recommendations=[
            "Check water quality parameters",
            "Review diet for nutritional deficiencies"
        ]
    )

    alert2 = alert_system.generate_alert(
        alert_type=AlertType.BEHAVIORAL_HEALTH,
        alert_level=AlertLevel.CRITICAL,
        title="Critical Behavioral Issue - Fish #2",
        message="Fish showing lethargy and bottom-sitting behavior",
        fish_id=2,
        metrics={'activity_score': 0.25},
        recommendations=[
            "URGENT: Check oxygen levels",
            "Test for ammonia and nitrite",
            "Consult aquatic veterinarian"
        ]
    )

    print(f"\nGenerated {len(alert_system.alert_history)} alerts")

    # Get alert summary
    summary = alert_system.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"  Total active: {summary['total_active']}")
    print(f"  Critical: {summary['critical']}")
    print(f"  Warning: {summary['warning']}")

    # Print alerts
    print("\nActive Alerts:")
    for alert in alert_system.get_active_alerts():
        print(f"\n[{alert.alert_level.value.upper()}] {alert.title}")
        print(f"  Message: {alert.message}")
        print(f"  Recommendations:")
        for rec in alert.recommendations:
            print(f"    - {rec}")


def main():
    print("\n" + "="*60)
    print("FISH HEALTH MONITORING - COMPONENT USAGE EXAMPLES")
    print("="*60)

    # Run examples
    example_detection()
    example_tracking()
    example_health_analysis()
    example_alert_system()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
