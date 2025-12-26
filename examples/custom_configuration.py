"""
Custom Configuration Example
Demonstrates how to use custom configuration settings
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_monitor import FishHealthMonitor
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging


def main():
    # Load configuration
    config = load_config("configs/default_config.yaml")

    # Setup logging from config
    setup_logging(
        log_file=config.logging.log_file,
        level=config.logging.level,
        console_output=config.logging.console_output
    )

    print("Configuration loaded:")
    print(f"  Detection model: {config.detection.model_path}")
    print(f"  Confidence threshold: {config.detection.confidence_threshold}")
    print(f"  Alert database: {config.alerts.database_path}")

    # Initialize monitor with config settings
    monitor = FishHealthMonitor(
        detector_model_path=config.detection.model_path,
        detection_confidence=config.detection.confidence_threshold,
        tracker_max_age=config.tracking.max_age,
        alert_db_path=config.alerts.database_path,
        device=config.detection.device,
        fps=config.video.fps,
        enable_visualization=config.video.enable_visualization
    )

    # Customize alert thresholds
    monitor.alert_system.thresholds = {
        'critical_health': config.alerts.thresholds.critical_health,
        'warning_health': config.alerts.thresholds.warning_health,
        'critical_behavior': config.alerts.thresholds.critical_behavior,
        'warning_behavior': config.alerts.thresholds.warning_behavior
    }

    # Customize behavioral analyzer
    monitor.behavior_analyzer.normal_speed_range = tuple(
        config.behavior.normal_speed_range
    )
    monitor.behavior_analyzer.normal_surface_ratio = config.behavior.normal_surface_ratio
    monitor.behavior_analyzer.normal_bottom_ratio = config.behavior.normal_bottom_ratio

    print("\nSystem configured and ready!")

    # Example: Process video with custom settings
    video_path = "data/raw/sample_aquarium.mp4"

    if Path(video_path).exists():
        print(f"\nProcessing video: {video_path}")
        monitor.process_video(
            video_path=video_path,
            output_path="data/processed/custom_output.mp4",
            display=True,
            max_frames=300  # Process first 300 frames
        )
    else:
        print(f"\nVideo not found: {video_path}")
        print("Skipping video processing demo")


if __name__ == "__main__":
    main()
