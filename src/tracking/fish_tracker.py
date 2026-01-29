"""
Fish Tracking Module
Tracks individual fish across frames using enhanced SORT algorithm with appearance features
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from ..detection.fish_detector import Detection


@dataclass
class Track:
    """Represents a tracked fish across multiple frames with appearance features"""
    track_id: int
    detections: deque = field(default_factory=lambda: deque(maxlen=100))
    state: KalmanFilter = field(default_factory=lambda: None)
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    active: bool = True
    appearance_features: deque = field(default_factory=lambda: deque(maxlen=10))  # Store last 10 features
    last_frame: Optional[np.ndarray] = None
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))

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

    def update(self, detection: Detection, frame: Optional[np.ndarray] = None):
        """Update track with new detection and extract appearance features"""
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
        self.confidence_history.append(detection.confidence)

        # Extract and store appearance features if frame provided
        if frame is not None:
            features = self._extract_appearance_features(frame, detection.bbox)
            if features is not None:
                self.appearance_features.append(features)
                self.last_frame = frame

    def _extract_appearance_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract appearance features from fish region for re-identification"""
        try:
            x1, y1, x2, y2 = bbox
            # Clamp to frame boundaries
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Extract fish region
            fish_region = frame[y1:y2, x1:x2]

            if fish_region.size == 0:
                return None

            # Resize to fixed size for consistent feature extraction
            fish_region = cv2.resize(fish_region, (64, 64))

            # Extract multiple feature types
            features = []

            # 1. Color histogram in HSV space (more robust to lighting)
            hsv = cv2.cvtColor(fish_region, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

            # Normalize histograms
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)

            features.extend(hist_h)
            features.extend(hist_s)
            features.extend(hist_v)

            # 2. Spatial color moments (mean color in grid)
            grid_size = 2
            cell_h, cell_w = fish_region.shape[0] // grid_size, fish_region.shape[1] // grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = fish_region[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    mean_color = cell.mean(axis=(0, 1))
                    features.extend(mean_color / 255.0)  # Normalize to [0, 1]

            return np.array(features, dtype=np.float32)

        except Exception:
            return None

    def get_appearance_feature(self) -> Optional[np.ndarray]:
        """Get average appearance feature from recent detections"""
        if len(self.appearance_features) == 0:
            return None

        # Average recent features for more robust matching
        return np.mean(list(self.appearance_features), axis=0)

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
    Enhanced multi-fish tracker using Kalman filtering, Hungarian algorithm, and appearance features
    """

    def __init__(
        self,
        max_age: int = 150,  # Increased from 30 to 150 (5 seconds at 30 FPS)
        min_hits: int = 3,
        iou_threshold: float = 0.2,  # Lowered from 0.3 for more flexible matching
        appearance_weight: float = 0.4,  # Weight for appearance similarity in matching
        enable_reid: bool = True  # Enable re-identification of lost tracks
    ):
        """
        Initialize enhanced tracker with appearance-based re-identification

        Args:
            max_age: Maximum frames to keep track alive without detection (increased for persistence)
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IOU for matching detection to track (lowered for flexibility)
            appearance_weight: Weight of appearance similarity vs IOU (0.0-1.0)
            enable_reid: Enable re-identification of recently lost tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.enable_reid = enable_reid

        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_count = 0
        self.lost_tracks: List[Track] = []  # Recently lost tracks for re-identification

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Update tracker with new detections and optional frame for appearance features

        Args:
            detections: List of detections from current frame
            frame: Optional frame for appearance feature extraction (BGR format)

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks using both IoU and appearance
        matched, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(
            detections, self.tracks, frame
        )

        # Update matched tracks with appearance features
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx], frame)

        # Try to re-identify unmatched detections with recently lost tracks
        if self.enable_reid and len(unmatched_dets) > 0 and len(self.lost_tracks) > 0 and frame is not None:
            reid_matched, still_unmatched = self._attempt_reidentification(
                detections, unmatched_dets, self.lost_tracks, frame
            )

            # Recover re-identified tracks
            for det_idx, lost_track in reid_matched:
                # Reactivate the track
                lost_track.active = True
                lost_track.time_since_update = 0
                lost_track.update(detections[det_idx], frame)
                self.tracks.append(lost_track)
                if det_idx in unmatched_dets:
                    unmatched_dets.remove(det_idx)

            # Update lost tracks list
            self.lost_tracks = [t for t in self.lost_tracks if t not in [m[1] for m in reid_matched]]

        # Create new tracks for still unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], frame)

        # Handle unmatched tracks
        for trk_idx in unmatched_trks:
            track = self.tracks[trk_idx]
            if track.time_since_update > self.max_age:
                track.active = False
                # Add to lost tracks for potential re-identification
                if self.enable_reid and track.hits >= self.min_hits:
                    if track not in self.lost_tracks:
                        self.lost_tracks.append(track)

        # Clean up lost tracks that are too old (keep for 2x max_age)
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update < self.max_age * 2]

        # Remove very old inactive tracks from main list
        self.tracks = [t for t in self.tracks if t.active or t.time_since_update < self.max_age]

        # Return confirmed active tracks
        return [t for t in self.tracks if t.hit_streak >= self.min_hits and t.active]

    def _create_track(self, detection: Detection, frame: Optional[np.ndarray] = None) -> Track:
        """Create new track from detection with appearance features"""
        # Initialize Kalman filter state
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        track = Track(track_id=self.next_id)
        track.state.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        track.update(detection, frame)

        self.tracks.append(track)
        self.next_id += 1

        return track

    def _match_detections_to_tracks(
        self,
        detections: List[Detection],
        tracks: List[Track],
        frame: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using Hungarian algorithm with IoU and appearance features

        Returns:
            matched: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(tracks)))

        # Compute combined similarity matrix (IoU + appearance)
        similarity_matrix = np.zeros((len(detections), len(tracks)))

        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                if not track.active:
                    continue

                track_bbox = track.get_current_bbox()
                iou_score = self._compute_iou(det.bbox, track_bbox)

                # Combine IoU with appearance similarity if available
                if frame is not None and self.appearance_weight > 0:
                    # Extract appearance feature for this detection
                    det_feature = track._extract_appearance_features(frame, det.bbox)
                    track_feature = track.get_appearance_feature()

                    if det_feature is not None and track_feature is not None:
                        # Compute cosine similarity
                        appearance_sim = self._cosine_similarity(det_feature, track_feature)
                        # Combine scores
                        similarity_matrix[d, t] = (
                            (1 - self.appearance_weight) * iou_score +
                            self.appearance_weight * appearance_sim
                        )
                    else:
                        similarity_matrix[d, t] = iou_score
                else:
                    similarity_matrix[d, t] = iou_score

        # Use Hungarian algorithm for optimal assignment
        # Convert to cost matrix (maximize similarity = minimize -similarity)
        cost_matrix = -similarity_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches by threshold
        # For combined score, we use a lower threshold since appearance helps
        effective_threshold = self.iou_threshold * (1 - self.appearance_weight * 0.5)

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))

        for r, c in zip(row_ind, col_ind):
            if similarity_matrix[r, c] >= effective_threshold:
                matched.append((r, c))
                unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    def _attempt_reidentification(
        self,
        detections: List[Detection],
        unmatched_det_indices: List[int],
        lost_tracks: List[Track],
        frame: np.ndarray
    ) -> Tuple[List[Tuple[int, Track]], List[int]]:
        """
        Attempt to re-identify lost tracks using appearance features

        Returns:
            reid_matched: List of (detection_idx, lost_track) pairs
            still_unmatched: List of detection indices that couldn't be re-identified
        """
        if len(unmatched_det_indices) == 0 or len(lost_tracks) == 0:
            return [], unmatched_det_indices

        # Compute appearance similarity between unmatched detections and lost tracks
        reid_matrix = np.zeros((len(unmatched_det_indices), len(lost_tracks)))

        for d_idx, det_idx in enumerate(unmatched_det_indices):
            det = detections[det_idx]
            det_feature = Track()._extract_appearance_features(frame, det.bbox)

            if det_feature is None:
                continue

            for t_idx, track in enumerate(lost_tracks):
                track_feature = track.get_appearance_feature()

                if track_feature is None:
                    continue

                # Use only appearance similarity for re-identification
                appearance_sim = self._cosine_similarity(det_feature, track_feature)

                # Also check if position is reasonable (not too far from last known position)
                track_bbox = track.get_current_bbox()
                iou_score = self._compute_iou(det.bbox, track_bbox)

                # If there's some spatial overlap OR strong appearance match, consider it
                if iou_score > 0.05 or appearance_sim > 0.75:
                    # Weighted combination, but favor appearance for re-ID
                    reid_matrix[d_idx, t_idx] = 0.7 * appearance_sim + 0.3 * iou_score

        # Use Hungarian algorithm for optimal re-identification
        if reid_matrix.max() > 0:
            cost_matrix = -reid_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            reid_matched = []
            still_unmatched = unmatched_det_indices.copy()

            # High threshold for re-identification to avoid false positives
            reid_threshold = 0.65

            for r, c in zip(row_ind, col_ind):
                if reid_matrix[r, c] >= reid_threshold:
                    det_idx = unmatched_det_indices[r]
                    track = lost_tracks[c]
                    reid_matched.append((det_idx, track))
                    if det_idx in still_unmatched:
                        still_unmatched.remove(det_idx)

            return reid_matched, still_unmatched

        return [], unmatched_det_indices

    @staticmethod
    def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors"""
        if feat1 is None or feat2 is None:
            return 0.0

        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

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
