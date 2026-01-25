"""
Main Fish Health Monitoring System
Integrates detection, tracking, health assessment, and alerting
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .detection.fish_detector import FishDetector, Detection
from .tracking.fish_tracker import FishTracker, Track
from .health_assessment.visual_analyzer import VisualHealthAnalyzer, VisualHealthMetrics
from .behavior_analysis.behavior_analyzer import BehaviorAnalyzer, BehaviorMetrics
from .alerts.alert_system import AlertSystem, Alert


@dataclass
class FishHealthReport:
    """Complete health report for a single fish"""
    track_id: int
    timestamp: float
    visual_health: VisualHealthMetrics
    behavioral_health: BehaviorMetrics
    overall_health_score: float
    confidence: float


class FishHealthMonitor:
    """
    Real-time fish health monitoring system
    Integrates all components for comprehensive health assessment
    """

    def __init__(
        self,
        detector_model_path: str = "yolov8n.pt",
        detection_confidence: float = 0.5,
        tracker_max_age: int = 30,
        alert_db_path: str = "alerts.db",
        device: Optional[str] = None,
        fps: float = 30.0,
        enable_visualization: bool = True,
        analysis_interval: int = 5,
        skip_expensive_analysis: bool = True
    ):
        """
        Initialize fish health monitoring system

        Args:
            detector_model_path: Path to YOLO model
            detection_confidence: Minimum confidence for detections
            tracker_max_age: Maximum age for tracks
            alert_db_path: Path to alert database
            device: Device for models ('cuda', 'mps', 'cpu', or None for auto)
            fps: Video frame rate
            enable_visualization: Whether to generate visualization frames
            analysis_interval: Analyze health every N frames (1 = every frame, 5 = every 5th frame)
            skip_expensive_analysis: Skip expensive operations (eye detection, fin analysis)
        """
        # Auto-detect device if not specified
        if device is None:
            import torch
            import platform
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
            else:
                device = 'cpu'
        
        # Initialize components
        self.detector = FishDetector(
            model_path=detector_model_path,
            confidence_threshold=detection_confidence,
            device=device
        )

        self.tracker = FishTracker(
            max_age=tracker_max_age,
            min_hits=3
        )

        self.visual_analyzer = VisualHealthAnalyzer(
            use_deep_features=True,
            device=device,
            skip_expensive_analysis=skip_expensive_analysis
        )

        self.behavior_analyzer = BehaviorAnalyzer(
            history_length=100,
            fps=fps
        )

        self.alert_system = AlertSystem(
            db_path=alert_db_path
        )

        self.enable_visualization = enable_visualization
        self.fps = fps
        self.analysis_interval = analysis_interval
        self.skip_expensive_analysis = skip_expensive_analysis

        # Frame counter for analysis interval
        self.frame_count = 0

        # Health reports cache
        self.health_reports: Dict[int, FishHealthReport] = {}

        # Performance metrics
        self.processing_times = []

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, List[FishHealthReport], List[Alert]]:
        """
        Process a single video frame

        Args:
            frame: Input frame (BGR format)
            timestamp: Optional timestamp

        Returns:
            Tuple of (visualization_frame, health_reports, new_alerts)
        """
        start_time = time.time()

        if timestamp is None:
            timestamp = time.time()

        # Step 1: Detect fish
        detections = self.detector.detect(frame, timestamp)

        # Step 2: Update tracker
        active_tracks = self.tracker.update(detections)

        # Step 3: Analyze health for each tracked fish
        health_reports = []
        new_alerts = []

        # Increment frame counter
        self.frame_count += 1

        # Determine if we should do full analysis this frame
        # Always analyze on first frame or when interval matches
        should_analyze = (
            self.frame_count % self.analysis_interval == 0 or
            self.frame_count == 1
        )

        # Get positions of all fish for social analysis
        all_positions = [track.get_current_bbox() for track in active_tracks]
        all_centers = [(
            (bbox[0] + bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2
        ) for bbox in all_positions]

        for track in active_tracks:
            # Check if we should analyze this fish
            # Always analyze new tracks (not in cache) or when interval matches
            is_new_track = track.track_id not in self.health_reports
            
            if should_analyze or is_new_track:
                # Full health analysis
                current_bbox = track.get_current_bbox()
                visual_health = self.visual_analyzer.analyze(
                    frame,
                    current_bbox,
                    track_history=list(track.detections)
                )

                # Behavioral assessment
                position_history = track.get_position_history(n=100)
                bbox_history = [d.bbox for d in list(track.detections)[-100:]]

                # Get other fish positions (excluding current fish)
                other_positions = [pos for i, pos in enumerate(all_centers)
                                 if i != active_tracks.index(track)]

                behavioral_health = self.behavior_analyzer.analyze(
                    position_history,
                    bbox_history,
                    other_positions
                )

                # Calculate overall health score
                overall_health = (
                    0.5 * visual_health.overall_health_score +
                    0.5 * behavioral_health.overall_behavior_score
                )

                # Create health report
                report = FishHealthReport(
                    track_id=track.track_id,
                    timestamp=timestamp,
                    visual_health=visual_health,
                    behavioral_health=behavioral_health,
                    overall_health_score=overall_health,
                    confidence=track.hits / max(track.age, 1)
                )

                health_reports.append(report)
                self.health_reports[track.track_id] = report
            else:
                # Reuse previous analysis (much faster!)
                if track.track_id in self.health_reports:
                    report = self.health_reports[track.track_id]
                    # Update timestamp but keep health scores
                    report = FishHealthReport(
                        track_id=track.track_id,
                        timestamp=timestamp,
                        visual_health=report.visual_health,
                        behavioral_health=report.behavioral_health,
                        overall_health_score=report.overall_health_score,
                        confidence=track.hits / max(track.age, 1)
                    )
                    health_reports.append(report)
                else:
                    # First time seeing this track, must analyze
                    current_bbox = track.get_current_bbox()
                    visual_health = self.visual_analyzer.analyze(
                        frame,
                        current_bbox,
                        track_history=list(track.detections)
                    )
                    position_history = track.get_position_history(n=100)
                    bbox_history = [d.bbox for d in list(track.detections)[-100:]]
                    other_positions = [pos for i, pos in enumerate(all_centers)
                                     if i != active_tracks.index(track)]
                    behavioral_health = self.behavior_analyzer.analyze(
                        position_history,
                        bbox_history,
                        other_positions
                    )
                    overall_health = (
                        0.5 * visual_health.overall_health_score +
                        0.5 * behavioral_health.overall_behavior_score
                    )
                    report = FishHealthReport(
                        track_id=track.track_id,
                        timestamp=timestamp,
                        visual_health=visual_health,
                        behavioral_health=behavioral_health,
                        overall_health_score=overall_health,
                        confidence=track.hits / max(track.age, 1)
                    )
                    health_reports.append(report)
                    self.health_reports[track.track_id] = report

            # Check for alerts (only for well-established tracks)
            # Require at least 30 frames of tracking to avoid false alarms on startup
            MIN_FRAMES_FOR_ALERTS = 30

            if track.hits >= MIN_FRAMES_FOR_ALERTS:
                # Get the report for this specific track
                current_report = None
                for report in health_reports:
                    if report.track_id == track.track_id:
                        current_report = report
                        break
                
                # Fallback to cached report if not in current batch
                if current_report is None:
                    current_report = self.health_reports.get(track.track_id)
                
                if current_report:
                    visual_alert = self.alert_system.check_visual_health(
                        track.track_id,
                        current_report.visual_health
                    )
                    if visual_alert:
                        new_alerts.append(visual_alert)

                    behavioral_alert = self.alert_system.check_behavioral_health(
                        track.track_id,
                        current_report.behavioral_health
                    )
                    if behavioral_alert:
                        new_alerts.append(behavioral_alert)

        # Step 4: Generate visualization
        if self.enable_visualization:
            vis_frame = self._create_visualization(
                frame,
                active_tracks,
                health_reports
            )
        else:
            vis_frame = frame.copy()

        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        return vis_frame, health_reports, new_alerts

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None
    ) -> List[FishHealthReport]:
        """
        Process entire video file

        Args:
            video_path: Path to input video
            output_path: Optional path for output video
            display: Whether to display frames during processing
            max_frames: Maximum number of frames to process

        Returns:
            List of all health reports
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output video writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_reports = []
        frame_count = 0
        
        # Reset frame counter for new video
        self.frame_count = 0

        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        print(f"Analysis interval: Every {self.analysis_interval} frames")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_count / fps

                # Process frame
                vis_frame, reports, alerts = self.process_frame(frame, timestamp)

                all_reports.extend(reports)

                # Print alerts
                for alert in alerts:
                    print(f"\n[{alert.alert_level.value.upper()}] {alert.title}")
                    print(f"  {alert.message}")

                # Write output
                if writer:
                    writer.write(vis_frame)

                # Display
                if display:
                    cv2.imshow('Fish Health Monitor', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1

                if max_frames and frame_count >= max_frames:
                    break

                # Print progress
                if frame_count % 30 == 0:
                    avg_time = np.mean(self.processing_times)
                    print(f"Frame {frame_count}, "
                          f"Tracked: {len(reports)}, "
                          f"Processing: {avg_time*1000:.1f}ms/frame "
                          f"({1/avg_time:.1f} FPS)")

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        print(f"\nProcessing complete. Processed {frame_count} frames")
        print(f"Total fish tracked: {len(self.health_reports)}")
        print(f"Total alerts: {len(self.alert_system.alert_history)}")

        return all_reports

    def process_camera(
        self,
        camera_id: int = 0,
        display: bool = True
    ):
        """
        Process live camera feed

        Args:
            camera_id: Camera device ID
            display: Whether to display frames
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")

        print(f"Starting live monitoring from camera {camera_id}")
        print("Press 'q' to quit, 's' to screenshot, 'a' to show active alerts")
        print(f"Analysis interval: Every {self.analysis_interval} frames")

        frame_count = 0
        
        # Reset frame counter for new camera session
        self.frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                timestamp = time.time()

                # Process frame with error handling
                try:
                    vis_frame, reports, alerts = self.process_frame(frame, timestamp)
                except Exception as e:
                    print(f"\nWarning: Error processing frame {frame_count}: {e}")
                    # Show original frame on error
                    vis_frame = frame
                    reports = []
                    alerts = []

                # Print new alerts
                for alert in alerts:
                    print(f"\n[{alert.alert_level.value.upper()}] {alert.title}")
                    print(f"  {alert.message}")
                    print(f"  Recommendations: {', '.join(alert.recommendations[:2])}")

                # Display
                if display:
                    cv2.imshow('Fish Health Monitor - Live', vis_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('a'):
                    # Show active alerts
                    active = self.alert_system.get_active_alerts()
                    print(f"\nActive Alerts: {len(active)}")
                    for alert in active[-5:]:  # Show last 5
                        print(f"  - {alert.title}")

                frame_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"\nMonitoring ended. Processed {frame_count} frames")

    def _create_visualization(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        reports: List[FishHealthReport]
    ) -> np.ndarray:
        """Create visualization with health overlay"""
        vis_frame = frame.copy()

        # Create report lookup
        report_dict = {r.track_id: r for r in reports}

        for track in tracks:
            bbox = track.get_current_bbox()
            x1, y1, x2, y2 = bbox

            report = report_dict.get(track.track_id)

            if report:
                # Color based on health
                health_score = report.overall_health_score
                if health_score >= 0.7:
                    color = (0, 255, 0)  # Green - healthy
                    status = "Healthy"
                elif health_score >= 0.5:
                    color = (0, 255, 255)  # Yellow - mild concern
                    status = "Concern"
                elif health_score >= 0.3:
                    color = (0, 165, 255)  # Orange - moderate
                    status = "Warning"
                else:
                    color = (0, 0, 255)  # Red - severe
                    status = "Critical"

                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                # Draw track trail
                positions = track.get_position_history(n=30)
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(vis_frame, positions[i-1], positions[i], color, 1)

                # Draw info panel
                info_y = y1 - 10
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1

                # ID and status
                label = f"ID:{track.track_id} {status} ({health_score:.2f})"
                (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(vis_frame, (x1, info_y-h-5), (x1+w, info_y), color, -1)
                cv2.putText(vis_frame, label, (x1, info_y-3), font, font_scale, (0, 0, 0), thickness)

                # Additional health info
                info_y -= (h + 7)
                visual_status = report.visual_health.health_status.value
                behavior_status = report.behavioral_health.behavior_status.value

                info = f"V:{visual_status[:4]} B:{behavior_status[:4]}"
                (w, h), _ = cv2.getTextSize(info, font, font_scale, thickness)
                cv2.rectangle(vis_frame, (x1, info_y-h-5), (x1+w, info_y), (255, 255, 255), -1)
                cv2.putText(vis_frame, info, (x1, info_y-3), font, font_scale, (0, 0, 0), thickness)

        # Add statistics overlay
        self._add_stats_overlay(vis_frame, tracks, reports)

        return vis_frame

    def _add_stats_overlay(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        reports: List[FishHealthReport]
    ):
        """Add statistics overlay to frame"""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1

        y = 25
        spacing = 25

        # Fish count
        text = f"Fish Tracked: {len(tracks)}"
        cv2.putText(frame, text, (10, y), font, font_scale, color, thickness)
        y += spacing

        # Health summary
        if reports:
            healthy = sum(1 for r in reports if r.overall_health_score >= 0.7)
            concern = sum(1 for r in reports if 0.5 <= r.overall_health_score < 0.7)
            warning = sum(1 for r in reports if 0.3 <= r.overall_health_score < 0.5)
            critical = sum(1 for r in reports if r.overall_health_score < 0.3)

            text = f"Health: Healthy:{healthy} Concern:{concern} Warning:{warning} Critical:{critical}"
            cv2.putText(frame, text, (10, y), font, font_scale, color, thickness)
            y += spacing

        # Active alerts
        active_alerts = self.alert_system.get_active_alerts()
        alert_summary = self.alert_system.get_alert_summary()

        text = f"Active Alerts: {alert_summary['total_active']} "
        text += f"(Critical:{alert_summary['critical']} Warning:{alert_summary['warning']})"
        cv2.putText(frame, text, (10, y), font, font_scale, color, thickness)
        y += spacing

        # Processing FPS
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            fps = 1 / avg_time if avg_time > 0 else 0
            text = f"Processing: {avg_time*1000:.1f}ms/frame ({fps:.1f} FPS)"
            cv2.putText(frame, text, (10, y), font, font_scale, color, thickness)

    def get_health_summary(self) -> Dict:
        """Get summary of current health status"""
        summary = {
            'total_fish': len(self.health_reports),
            'healthy': 0,
            'concern': 0,
            'warning': 0,
            'critical': 0,
            'avg_health_score': 0.0,
            'active_alerts': self.alert_system.get_alert_summary()
        }

        if self.health_reports:
            scores = [r.overall_health_score for r in self.health_reports.values()]
            summary['avg_health_score'] = np.mean(scores)

            for report in self.health_reports.values():
                score = report.overall_health_score
                if score >= 0.7:
                    summary['healthy'] += 1
                elif score >= 0.5:
                    summary['concern'] += 1
                elif score >= 0.3:
                    summary['warning'] += 1
                else:
                    summary['critical'] += 1

        return summary

    def reset(self):
        """Reset the monitoring system"""
        self.detector.reset_frame_count()
        self.tracker.reset()
        self.health_reports.clear()
        self.processing_times.clear()
        self.frame_count = 0
