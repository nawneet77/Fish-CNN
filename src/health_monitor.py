"""
Main Fish Health Monitoring System
Integrates detection, tracking, health assessment, and alerting
"""

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

from .detection.fish_detector import FishDetector, Detection
from .tracking.fish_tracker import FishTracker, Track
from .health_assessment.visual_analyzer import VisualHealthAnalyzer, VisualHealthMetrics
from .behavior_analysis.behavior_analyzer import BehaviorAnalyzer, BehaviorMetrics
from .alerts.alert_system import AlertSystem, Alert


@dataclass
class FishBaseline:
    """Per-fish adaptive baseline using Welford's online algorithm.
    Learns each fish's normal behavior for Z-score anomaly detection."""
    # Running statistics per metric (mean, M2 for variance, count)
    _means: Dict[str, float] = field(default_factory=dict)
    _m2s: Dict[str, float] = field(default_factory=dict)
    _counts: Dict[str, int] = field(default_factory=dict)
    body_length_mean: float = 50.0
    sample_count: int = 0
    is_ready: bool = False

    METRICS = [
        'speed', 'activity', 'pattern', 'isolation',
        'surface', 'bottom', 'erratic', 'color', 'body_shape'
    ]
    MIN_SAMPLES = 150  # ~25 seconds at analysis_interval=5

    def update(self, values: Dict[str, float]):
        """Update baselines using Welford's online algorithm."""
        self.sample_count += 1
        for key, val in values.items():
            n = self._counts.get(key, 0) + 1
            self._counts[key] = n
            old_mean = self._means.get(key, 0.0)
            delta = val - old_mean
            new_mean = old_mean + delta / n
            delta2 = val - new_mean
            self._means[key] = new_mean
            self._m2s[key] = self._m2s.get(key, 0.0) + delta * delta2
        if self.sample_count >= self.MIN_SAMPLES:
            self.is_ready = True

    def get_mean(self, key: str) -> float:
        return self._means.get(key, 0.0)

    def get_std(self, key: str) -> float:
        n = self._counts.get(key, 0)
        if n < 2:
            return 1.0
        return math.sqrt(self._m2s.get(key, 0.0) / (n - 1))

    def z_score(self, key: str, value: float) -> float:
        """Compute Z-score: how many standard deviations from this fish's mean."""
        std = self.get_std(key)
        return abs(value - self.get_mean(key)) / max(std, 0.01)


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
        detection_iou_threshold: float = 0.45,
        tracker_max_age: int = 150,  # Increased from 30 to 150 for better persistence
        alert_db_path: str = "alerts.db",
        device: Optional[str] = None,
        fps: float = 30.0,
        enable_visualization: bool = True,
        analysis_interval: int = 5,
        skip_expensive_analysis: bool = True,
        enable_reid: bool = True  # Enable appearance-based re-identification
    ):
        """
        Initialize fish health monitoring system with enhanced tracking

        Args:
            detector_model_path: Path to YOLO model
            detection_confidence: Minimum confidence for detections
            detection_iou_threshold: IOU threshold for NMS (default: 0.45)
            tracker_max_age: Maximum frames to keep track alive without detection (default: 150 = 5 sec)
            alert_db_path: Path to alert database
            device: Device for models ('cuda', 'mps', 'cpu', or None for auto)
            fps: Video frame rate
            enable_visualization: Whether to generate visualization frames
            analysis_interval: Analyze health every N frames (1 = every frame, 5 = every 5th frame)
            skip_expensive_analysis: Skip expensive operations (eye detection, fin analysis)
            enable_reid: Enable appearance-based re-identification for better tracking persistence
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
            iou_threshold=detection_iou_threshold,
            device=device
        )

        self.tracker = FishTracker(
            max_age=tracker_max_age,
            min_hits=3,
            iou_threshold=0.2,
            appearance_weight=0.4,
            enable_reid=enable_reid,
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

        # Alert deduplication tracking
        # Format: {fish_id: {alert_type: last_alert_timestamp}}
        self.last_alert_times: Dict[int, Dict[str, float]] = {}
        self.alert_cooldown = 60.0  # Don't repeat same alert within 60 seconds

        # Baseline collection mode
        self.baseline_mode = True
        self.baseline_frames_needed = 300  # Reduced — per-fish baselines handle the rest
        self.baseline_data: Dict[int, List[float]] = {}
        self.baseline_scores: Dict[int, float] = {}

        # EMA smoothing for stable health scores
        self._score_history: Dict[int, deque] = {}
        self._smoothed_scores: Dict[int, float] = {}
        self._ema_alpha: float = 0.15  # Responds in ~7s instead of ~30s

        # Hysteresis state to prevent status flickering
        self._fish_status: Dict[int, str] = {}

        # Sustained alert counters — only alert after consecutive low scores
        self._consecutive_low: Dict[int, int] = {}
        self._sustained_alert_threshold: int = 4  # Multi-indicator consensus filters noise

        # Per-fish adaptive baselines for Z-score anomaly detection
        self._fish_baselines: Dict[int, FishBaseline] = {}
        self._concern_reasons: Dict[int, List[str]] = {}

    def _update_smoothed_score(self, fish_id: int, raw_score: float) -> float:
        """Apply exponential moving average to stabilize health scores."""
        if fish_id not in self._score_history:
            self._score_history[fish_id] = deque(maxlen=30)
            self._smoothed_scores[fish_id] = raw_score

        self._score_history[fish_id].append(raw_score)
        prev = self._smoothed_scores[fish_id]
        ema = self._ema_alpha * raw_score + (1 - self._ema_alpha) * prev
        self._smoothed_scores[fish_id] = ema
        return ema

    def _apply_hysteresis(self, fish_id: int, smoothed_score: float) -> str:
        """Apply dead-band hysteresis to prevent status flickering."""
        current = self._fish_status.get(fish_id, "Healthy")

        if current == "Healthy" and smoothed_score < 0.60:
            new_status = "Concern"
        elif current == "Concern" and smoothed_score >= 0.70:
            new_status = "Healthy"
        elif current == "Concern" and smoothed_score < 0.40:
            new_status = "Warning"
        elif current == "Warning" and smoothed_score >= 0.55:
            new_status = "Concern"
        elif current == "Warning" and smoothed_score < 0.25:
            new_status = "Critical"
        elif current == "Critical" and smoothed_score >= 0.35:
            new_status = "Warning"
        else:
            new_status = current

        self._fish_status[fish_id] = new_status
        return new_status

    def get_smoothed_scores(self) -> Dict[int, float]:
        """Get current smoothed health scores for all fish."""
        return dict(self._smoothed_scores)

    def get_fish_statuses(self) -> Dict[int, str]:
        """Get current hysteresis-based status for all fish."""
        return dict(self._fish_status)

    def get_score_history(self, fish_id: int) -> List[float]:
        """Get recent score history for a fish (up to 30 values)."""
        if fish_id in self._score_history:
            return list(self._score_history[fish_id])
        return []

    def _update_fish_baseline(self, fish_id: int, visual: VisualHealthMetrics,
                              behavioral: BehaviorMetrics, bbox: Tuple[int, int, int, int]):
        """Update per-fish adaptive baseline with current metrics."""
        if fish_id not in self._fish_baselines:
            self._fish_baselines[fish_id] = FishBaseline()

        bl = self._fish_baselines[fish_id]

        # Compute body length for speed normalization
        x1, y1, x2, y2 = bbox
        body_diag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if bl.sample_count == 0:
            bl.body_length_mean = body_diag
        else:
            bl.body_length_mean += (body_diag - bl.body_length_mean) / (bl.sample_count + 1)

        # Normalize speed by body length
        norm_speed = behavioral.swimming_speed / max(bl.body_length_mean, 1.0) * 100.0

        bl.update({
            'speed': norm_speed,
            'activity': behavioral.activity_level_score,
            'pattern': behavioral.swimming_pattern_score,
            'isolation': behavioral.isolation_score,
            'surface': behavioral.surface_time_ratio,
            'bottom': behavioral.bottom_time_ratio,
            'erratic': behavioral.erratic_movement_score,
            'color': visual.color_score,
            'body_shape': visual.body_condition_score,
        })

    def _compute_adaptive_health_score(self, fish_id: int, visual: VisualHealthMetrics,
                                        behavioral: BehaviorMetrics,
                                        bbox: Tuple[int, int, int, int]) -> Tuple[float, List[str]]:
        """Compute health score using per-fish baselines + Z-score anomaly detection.

        Before baseline is ready: 30% visual + 70% behavioral (raw scores).
        After baseline ready: Z-score anomaly detection with multi-indicator consensus.
        Returns (score, list_of_concern_reasons).
        """
        bl = self._fish_baselines.get(fish_id)

        # Before baseline is ready, use weighted raw scores (70% behavioral)
        if bl is None or not bl.is_ready:
            raw = 0.30 * visual.overall_health_score + 0.70 * behavioral.overall_behavior_score
            return raw, []

        # Compute Z-scores against this fish's OWN baseline
        norm_speed = behavioral.swimming_speed / max(bl.body_length_mean, 1.0) * 100.0

        z_scores = {
            'speed': bl.z_score('speed', norm_speed),
            'activity': bl.z_score('activity', behavioral.activity_level_score),
            'pattern': bl.z_score('pattern', behavioral.swimming_pattern_score),
            'isolation': bl.z_score('isolation', behavioral.isolation_score),
            'surface': bl.z_score('surface', behavioral.surface_time_ratio),
            'bottom': bl.z_score('bottom', behavioral.bottom_time_ratio),
            'erratic': bl.z_score('erratic', behavioral.erratic_movement_score),
            'color': bl.z_score('color', visual.color_score),
            'body_shape': bl.z_score('body_shape', visual.body_condition_score),
        }

        # Weights: behavioral metrics weighted more heavily
        weights = {
            'speed': 0.15, 'activity': 0.12, 'pattern': 0.12,
            'isolation': 0.10, 'surface': 0.08, 'bottom': 0.08,
            'erratic': 0.12, 'color': 0.12, 'body_shape': 0.11,
        }

        # Multi-indicator consensus: penalty based on how many indicators are concerning
        reasons = []
        total_penalty = 0.0
        total_weight = sum(weights.values())

        friendly_names = {
            'speed': 'Swimming speed', 'activity': 'Activity level',
            'pattern': 'Swimming pattern', 'isolation': 'Social isolation',
            'surface': 'Surface time', 'bottom': 'Bottom time',
            'erratic': 'Erratic movement', 'color': 'Color/appearance',
            'body_shape': 'Body shape',
        }

        for metric, z in z_scores.items():
            if z > 1.5:
                # Penalty scales with Z-score beyond 1.5
                penalty = (z - 1.5) * weights[metric]
                total_penalty += penalty

                if z > 2.0:
                    # Plain English descriptions
                    desc = {
                        'speed': 'Unusual swimming speed',
                        'activity': 'Abnormal activity level',
                        'pattern': 'Irregular swimming pattern',
                        'isolation': 'Isolating from other fish',
                        'surface': 'Too much time at surface',
                        'bottom': 'Sitting at bottom',
                        'erratic': 'Erratic/jerky movements',
                        'color': 'Color change detected',
                        'body_shape': 'Body shape looks different',
                    }
                    reasons.append(desc.get(metric, friendly_names[metric]))

        # Convert penalty to score (0-1, higher = healthier)
        score = max(0.0, 1.0 - total_penalty / total_weight)

        # Keep reasons in order of severity (already added highest z-scores first naturally)

        return score, reasons[:5]  # Top 5 reasons

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

        # Step 2: Update tracker with frame for appearance-based re-identification
        active_tracks = self.tracker.update(detections, frame)

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
                    other_positions,
                    frame_dimensions=(frame.shape[1], frame.shape[0])
                )

                # Update per-fish adaptive baseline
                self._update_fish_baseline(track.track_id, visual_health, behavioral_health, current_bbox)

                # Adaptive health score with Z-score anomaly detection
                raw_health, reasons = self._compute_adaptive_health_score(
                    track.track_id, visual_health, behavioral_health, current_bbox
                )
                self._concern_reasons[track.track_id] = reasons
                overall_health = self._update_smoothed_score(track.track_id, raw_health)
                self._apply_hysteresis(track.track_id, overall_health)

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
                    # Still feed last raw score into EMA so smoothing is continuous
                    current_bbox = track.get_current_bbox()
                    raw_health, _ = self._compute_adaptive_health_score(
                        track.track_id, report.visual_health, report.behavioral_health, current_bbox
                    )
                    overall_health = self._update_smoothed_score(track.track_id, raw_health)
                    self._apply_hysteresis(track.track_id, overall_health)
                    report = FishHealthReport(
                        track_id=track.track_id,
                        timestamp=timestamp,
                        visual_health=report.visual_health,
                        behavioral_health=report.behavioral_health,
                        overall_health_score=overall_health,
                        confidence=track.hits / max(track.age, 1)
                    )
                    health_reports.append(report)
                    self.health_reports[track.track_id] = report
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
                        other_positions,
                        frame_dimensions=(frame.shape[1], frame.shape[0])
                    )
                    self._update_fish_baseline(track.track_id, visual_health, behavioral_health, current_bbox)
                    raw_health, reasons = self._compute_adaptive_health_score(
                        track.track_id, visual_health, behavioral_health, current_bbox
                    )
                    self._concern_reasons[track.track_id] = reasons
                    overall_health = self._update_smoothed_score(track.track_id, raw_health)
                    self._apply_hysteresis(track.track_id, overall_health)
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

            # Get the report for this specific track
            current_report = None
            for report in health_reports:
                if report.track_id == track.track_id:
                    current_report = report
                    break

            # Fallback to cached report if not in current batch
            if current_report is None:
                current_report = self.health_reports.get(track.track_id)

            # Collect baseline data during initial monitoring period
            if self.baseline_mode and current_report and track.hits >= MIN_FRAMES_FOR_ALERTS:
                fish_id = track.track_id
                if fish_id not in self.baseline_data:
                    self.baseline_data[fish_id] = []

                self.baseline_data[fish_id].append(current_report.overall_health_score)

                # Check if we've collected enough baseline data
                if self.frame_count >= self.baseline_frames_needed:
                    # Calculate average baseline scores for each fish
                    for fid, scores in self.baseline_data.items():
                        if scores:
                            self.baseline_scores[fid] = sum(scores) / len(scores)

                    self.baseline_mode = False
                    print(f"\n{'='*70}")
                    print("BASELINE COLLECTION COMPLETE")
                    print(f"{'='*70}")
                    print(f"Collected baseline data for {len(self.baseline_scores)} fish")
                    for fid, score in self.baseline_scores.items():
                        print(f"  Fish #{fid}: Average baseline health = {score:.2f}")
                    print("\n🔔 Alert monitoring is now ACTIVE")
                    print(f"{'='*70}\n")

            # Generate alerts only after baseline collection and for well-established tracks
            if not self.baseline_mode and track.hits >= MIN_FRAMES_FOR_ALERTS and current_report:
                fish_id = track.track_id
                smoothed = self._smoothed_scores.get(fish_id, 1.0)

                # Track consecutive low scores — only alert after sustained issues
                if smoothed < self.alert_system.thresholds['warning_health']:
                    self._consecutive_low[fish_id] = self._consecutive_low.get(fish_id, 0) + 1
                else:
                    self._consecutive_low[fish_id] = 0

                # Only fire alerts after sustained threshold breach
                if self._consecutive_low.get(fish_id, 0) >= self._sustained_alert_threshold:
                    if fish_id not in self.last_alert_times:
                        self.last_alert_times[fish_id] = {}

                    alert_key = "adaptive_health"
                    last_alert = self.last_alert_times[fish_id].get(alert_key, 0)
                    if timestamp - last_alert > self.alert_cooldown:
                        reasons = self._concern_reasons.get(fish_id, [])
                        n_indicators = len(reasons)
                        reason_text = "; ".join(reasons[:3]) if reasons else "Multiple metrics below threshold"

                        from .alerts.alert_system import AlertLevel, AlertType
                        level = AlertLevel.CRITICAL if smoothed < 0.25 else AlertLevel.WARNING

                        title = f"Health Concern - Fish #{fish_id}"
                        if n_indicators > 0:
                            title += f" ({n_indicators} indicators)"
                        message = f"Fish #{fish_id}: {reason_text}"

                        alert = self.alert_system.generate_alert(
                            alert_type=AlertType.COMBINED_HEALTH,
                            alert_level=level,
                            title=title,
                            message=message,
                            fish_id=fish_id,
                            recommendations=["Monitor closely", "Check water quality"]
                        )
                        if alert:
                            new_alerts.append(alert)
                            self.last_alert_times[fish_id][alert_key] = timestamp

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
        """Create clean visualization — bounding boxes and trails only, no text overlays."""
        vis_frame = frame.copy()

        report_dict = {r.track_id: r for r in reports}

        for track in tracks:
            bbox = track.get_current_bbox()
            x1, y1, x2, y2 = bbox

            report = report_dict.get(track.track_id)

            if report:
                status = self._fish_status.get(track.track_id, "Healthy")
                if status == "Healthy":
                    color = (0, 255, 0)
                elif status == "Concern":
                    color = (0, 255, 255)
                elif status == "Warning":
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                # Draw fish ID label above box
                label = f"Fish:{track.track_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(vis_frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(vis_frame, label, (x1 + 3, y1 - 4), font, font_scale, (0, 0, 0), thickness)

                # Draw track trail
                positions = track.get_position_history(n=30)
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(vis_frame, positions[i-1], positions[i], color, 1)

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

        # Baseline collection status or Active alerts
        if self.baseline_mode:
            progress = min(100.0, (self.frame_count / self.baseline_frames_needed) * 100)
            text = f"BASELINE MODE: Collecting normal behavior data... {progress:.0f}%"
            cv2.putText(frame, text, (10, y), font, font_scale, (0, 255, 255), thickness)  # Yellow color
            y += spacing
            text = f"Alerts will activate after baseline collection (prevents false alarms)"
            cv2.putText(frame, text, (10, y), font, 0.4, (0, 255, 255), thickness)
            y += spacing
        else:
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
            scores = [self._smoothed_scores.get(fid, r.overall_health_score)
                      for fid, r in self.health_reports.items()]
            summary['avg_health_score'] = float(np.mean(scores))

            for fid in self.health_reports:
                status = self._fish_status.get(fid, "Healthy")
                if status == "Healthy":
                    summary['healthy'] += 1
                elif status == "Concern":
                    summary['concern'] += 1
                elif status == "Warning":
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
        self.last_alert_times.clear()
        self.baseline_data.clear()
        self.baseline_scores.clear()
        self.baseline_mode = True
        self._score_history.clear()
        self._smoothed_scores.clear()
        self._fish_status.clear()
        self._consecutive_low.clear()
        self._fish_baselines.clear()
        self._concern_reasons.clear()

    def skip_baseline_collection(self):
        """
        Skip baseline collection and start alerting immediately.
        Use this if you already know your fish's normal behavior.
        """
        self.baseline_mode = False
        print(f"\n{'='*70}")
        print("⚠️  BASELINE COLLECTION SKIPPED")
        print(f"{'='*70}")
        print("Alert monitoring is now ACTIVE (without baseline)")
        print("⚠️  WARNING: May generate false alerts without baseline data")
        print(f"{'='*70}\n")

    def get_baseline_status(self) -> Dict:
        """Get current baseline collection status"""
        if not self.baseline_mode:
            return {
                'collecting': False,
                'progress': 100.0,
                'fish_count': len(self.baseline_scores),
                'baseline_scores': self.baseline_scores.copy()
            }

        progress = min(100.0, (self.frame_count / self.baseline_frames_needed) * 100)
        return {
            'collecting': True,
            'progress': progress,
            'frames_collected': self.frame_count,
            'frames_needed': self.baseline_frames_needed,
            'fish_count': len(self.baseline_data)
        }