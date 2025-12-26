"""
Fish Tracking Module
Tracks individual fish across frames using SORT-like algorithm
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from ..detection.fish_detector import Detection


@dataclass
class Track:
    """Represents a tracked fish across multiple frames"""
    track_id: int
    detections: deque = field(default_factory=lambda: deque(maxlen=100))
    state: KalmanFilter = field(default_factory=lambda: None)
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    active: bool = True

    def __post_init__(self):
        if self.state is None:
            self.state = self._init_kalman_filter()

    def _init_kalman_filter(self) -> KalmanFilter:
        """Initialize Kalman filter for tracking"""
        # State: [x, y, w, h, vx, vy, vw, vh]
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement function
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # Measurement noise
        kf.R *= 10.0

        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        # Initial covariance
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0

        return kf

    def update(self, detection: Detection):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Convert detection to measurement
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        measurement = np.array([cx, cy, w, h])
        self.state.update(measurement)

        self.detections.append(detection)

    def predict(self) -> np.ndarray:
        """Predict next position"""
        self.state.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # Get predicted state
        x = self.state.x
        return x[:4]  # Return [cx, cy, w, h]

    def get_current_bbox(self) -> Tuple[int, int, int, int]:
        """Get current bounding box from Kalman state"""
        x = self.state.x
        cx, cy, w, h = x[0], x[1], x[2], x[3]

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        return (x1, y1, x2, y2)

    def get_position_history(self, n: int = 30) -> List[Tuple[int, int]]:
        """Get last n positions as (x, y) centers"""
        positions = []
        for det in list(self.detections)[-n:]:
            positions.append(det.center)
        return positions


class FishTracker:
    """
    Multi-fish tracker using Kalman filtering and Hungarian algorithm
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker

        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IOU for matching detection to track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections

        Args:
            detections: List of detections from current frame

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(
            detections, self.tracks
        )

        # Update matched tracks
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # Mark unmatched tracks as inactive if they're too old
        for trk_idx in unmatched_trks:
            track = self.tracks[trk_idx]
            if track.time_since_update > self.max_age:
                track.active = False

        # Remove very old inactive tracks
        self.tracks = [t for t in self.tracks if t.active or t.time_since_update < self.max_age * 2]

        # Return confirmed active tracks
        return [t for t in self.tracks if t.hit_streak >= self.min_hits and t.active]

    def _create_track(self, detection: Detection) -> Track:
        """Create new track from detection"""
        # Initialize Kalman filter state
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        track = Track(track_id=self.next_id)
        track.state.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        track.update(detection)

        self.tracks.append(track)
        self.next_id += 1

        return track

    def _match_detections_to_tracks(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using Hungarian algorithm

        Returns:
            matched: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(tracks)))

        # Compute IOU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))

        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                if not track.active:
                    continue
                track_bbox = track.get_current_bbox()
                iou_matrix[d, t] = self._compute_iou(det.bbox, track_bbox)

        # Use Hungarian algorithm for optimal assignment
        # Convert to cost matrix (maximize IOU = minimize -IOU)
        cost_matrix = -iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches by IOU threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute IOU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_active_tracks(self) -> List[Track]:
        """Get all active confirmed tracks"""
        return [t for t in self.tracks if t.hit_streak >= self.min_hits and t.active]

    def reset(self):
        """Reset tracker"""
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
