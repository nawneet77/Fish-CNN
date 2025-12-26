"""
Behavioral Analysis Module
Analyzes fish swimming patterns and behavior to detect abnormalities
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum


class BehaviorStatus(Enum):
    """Behavior status categories"""
    NORMAL = "normal"
    SLIGHTLY_ABNORMAL = "slightly_abnormal"
    MODERATELY_ABNORMAL = "moderately_abnormal"
    SEVERELY_ABNORMAL = "severely_abnormal"
    UNKNOWN = "unknown"


@dataclass
class BehaviorMetrics:
    """Behavioral health metrics for a fish"""
    swimming_speed: float  # pixels per frame
    swimming_pattern_score: float  # 0-1, regularity of swimming
    activity_level_score: float  # 0-1, compared to normal
    isolation_score: float  # 0-1, 1=isolated, 0=social
    surface_time_ratio: float  # 0-1, time spent at surface
    bottom_time_ratio: float  # 0-1, time spent at bottom
    erratic_movement_score: float  # 0-1, 1=very erratic
    overall_behavior_score: float  # 0-1, overall health
    behavior_status: BehaviorStatus
    behavioral_symptoms: List[str]


class BehaviorAnalyzer:
    """
    Analyzes fish behavior patterns to detect health issues
    """

    def __init__(
        self,
        history_length: int = 100,
        fps: float = 30.0,
        tank_height: int = 480
    ):
        """
        Initialize behavior analyzer

        Args:
            history_length: Number of frames to keep in history
            fps: Video frame rate
            tank_height: Height of tank in pixels (for depth analysis)
        """
        self.history_length = history_length
        self.fps = fps
        self.tank_height = tank_height

        # Behavioral thresholds
        self.thresholds = {
            'normal': 0.7,
            'slight': 0.5,
            'moderate': 0.3
        }

        # Normal behavior baselines (can be calibrated)
        self.normal_speed_range = (1.0, 15.0)  # pixels per frame
        self.normal_surface_ratio = 0.2  # up to 20% at surface is normal
        self.normal_bottom_ratio = 0.3  # up to 30% at bottom is normal

    def analyze(
        self,
        position_history: List[Tuple[int, int]],
        bbox_history: List[Tuple[int, int, int, int]],
        other_fish_positions: Optional[List[Tuple[int, int]]] = None,
        time_window_seconds: float = 10.0
    ) -> BehaviorMetrics:
        """
        Analyze fish behavior based on tracking history

        Args:
            position_history: List of (x, y) center positions
            bbox_history: List of (x1, y1, x2, y2) bounding boxes
            other_fish_positions: Positions of other fish for social analysis
            time_window_seconds: Time window for analysis

        Returns:
            BehaviorMetrics object
        """
        if len(position_history) < 2:
            return self._get_default_metrics()

        # Calculate behavior metrics
        speed = self._calculate_swimming_speed(position_history)
        pattern_score = self._analyze_swimming_pattern(position_history)
        activity_score = self._analyze_activity_level(position_history, speed)
        isolation_score = self._analyze_isolation(
            position_history[-1] if position_history else (0, 0),
            other_fish_positions
        )
        surface_ratio = self._calculate_surface_time(bbox_history)
        bottom_ratio = self._calculate_bottom_time(bbox_history)
        erratic_score = self._analyze_erratic_movement(position_history)

        # Calculate overall behavior score
        overall_score = self._calculate_overall_score(
            speed,
            pattern_score,
            activity_score,
            isolation_score,
            surface_ratio,
            bottom_ratio,
            erratic_score
        )

        # Classify behavior and identify symptoms
        status, symptoms = self._classify_behavior(
            speed,
            pattern_score,
            activity_score,
            isolation_score,
            surface_ratio,
            bottom_ratio,
            erratic_score
        )

        return BehaviorMetrics(
            swimming_speed=speed,
            swimming_pattern_score=pattern_score,
            activity_level_score=activity_score,
            isolation_score=isolation_score,
            surface_time_ratio=surface_ratio,
            bottom_time_ratio=bottom_ratio,
            erratic_movement_score=erratic_score,
            overall_behavior_score=overall_score,
            behavior_status=status,
            behavioral_symptoms=symptoms
        )

    def _calculate_swimming_speed(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate average swimming speed in pixels per frame"""
        if len(positions) < 2:
            return 0.0

        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)

        return np.mean(speeds) if speeds else 0.0

    def _analyze_swimming_pattern(self, positions: List[Tuple[int, int]]) -> float:
        """
        Analyze regularity of swimming pattern

        Healthy fish typically have smooth, regular swimming patterns
        Sick fish may show jerky, irregular movements
        """
        if len(positions) < 5:
            return 0.5

        # Calculate directional changes
        directions = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            if dx != 0 or dy != 0:
                angle = np.arctan2(dy, dx)
                directions.append(angle)

        if len(directions) < 2:
            return 0.5

        # Calculate smoothness (smaller direction changes = smoother)
        direction_changes = []
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i-1])
            # Normalize to [0, pi]
            diff = min(diff, 2*np.pi - diff)
            direction_changes.append(diff)

        avg_change = np.mean(direction_changes)
        smoothness = 1.0 - min(avg_change / np.pi, 1.0)

        # Calculate path efficiency (straight line distance / actual path length)
        if len(positions) >= 2:
            straight_dist = np.sqrt(
                (positions[-1][0] - positions[0][0])**2 +
                (positions[-1][1] - positions[0][1])**2
            )
            actual_dist = sum([
                np.sqrt((positions[i][0] - positions[i-1][0])**2 +
                       (positions[i][1] - positions[i-1][1])**2)
                for i in range(1, len(positions))
            ])
            efficiency = straight_dist / actual_dist if actual_dist > 0 else 0.5
        else:
            efficiency = 0.5

        pattern_score = 0.6 * smoothness + 0.4 * efficiency

        return np.clip(pattern_score, 0.0, 1.0)

    def _analyze_activity_level(
        self,
        positions: List[Tuple[int, int]],
        speed: float
    ) -> float:
        """
        Analyze activity level

        Very low activity (lethargic) or very high activity (frantic) can indicate illness
        """
        # Normal activity is moderate speed
        min_speed, max_speed = self.normal_speed_range

        if speed < min_speed:
            # Lethargic
            activity_score = speed / min_speed
        elif speed > max_speed:
            # Hyperactive
            activity_score = max(0.0, 1.0 - (speed - max_speed) / max_speed)
        else:
            # Normal range
            activity_score = 1.0

        return np.clip(activity_score, 0.0, 1.0)

    def _analyze_isolation(
        self,
        fish_position: Tuple[int, int],
        other_positions: Optional[List[Tuple[int, int]]]
    ) -> float:
        """
        Analyze if fish is isolating from others

        Sick fish often isolate themselves
        Returns: 0 = social, 1 = isolated
        """
        if other_positions is None or len(other_positions) == 0:
            return 0.5  # Can't determine

        # Calculate distance to nearest neighbor
        distances = []
        for other_pos in other_positions:
            dist = np.sqrt(
                (fish_position[0] - other_pos[0])**2 +
                (fish_position[1] - other_pos[1])**2
            )
            distances.append(dist)

        min_distance = min(distances)

        # Normalize by tank size (assume 640x480)
        tank_diagonal = np.sqrt(640**2 + 480**2)
        normalized_distance = min_distance / tank_diagonal

        # Convert to isolation score
        # Close to others = 0, far from others = 1
        isolation_score = min(normalized_distance / 0.5, 1.0)

        return isolation_score

    def _calculate_surface_time(self, bbox_history: List[Tuple[int, int, int, int]]) -> float:
        """Calculate proportion of time spent at water surface"""
        if not bbox_history:
            return 0.0

        surface_threshold = self.tank_height * 0.2  # Top 20% of tank

        surface_frames = 0
        for bbox in bbox_history:
            _, y1, _, _ = bbox
            if y1 < surface_threshold:
                surface_frames += 1

        return surface_frames / len(bbox_history)

    def _calculate_bottom_time(self, bbox_history: List[Tuple[int, int, int, int]]) -> float:
        """Calculate proportion of time spent at tank bottom"""
        if not bbox_history:
            return 0.0

        bottom_threshold = self.tank_height * 0.8  # Bottom 20% of tank

        bottom_frames = 0
        for bbox in bbox_history:
            _, _, _, y2 = bbox
            if y2 > bottom_threshold:
                bottom_frames += 1

        return bottom_frames / len(bbox_history)

    def _analyze_erratic_movement(self, positions: List[Tuple[int, int]]) -> float:
        """
        Detect erratic, jerky movements

        Returns: 0 = smooth, 1 = very erratic
        """
        if len(positions) < 3:
            return 0.0

        # Calculate acceleration (change in velocity)
        velocities = []
        for i in range(1, len(positions)):
            vx = positions[i][0] - positions[i-1][0]
            vy = positions[i][1] - positions[i-1][1]
            velocities.append((vx, vy))

        if len(velocities) < 2:
            return 0.0

        accelerations = []
        for i in range(1, len(velocities)):
            ax = velocities[i][0] - velocities[i-1][0]
            ay = velocities[i][1] - velocities[i-1][1]
            accel_mag = np.sqrt(ax**2 + ay**2)
            accelerations.append(accel_mag)

        # High acceleration variance indicates erratic movement
        if accelerations:
            accel_std = np.std(accelerations)
            # Normalize (assuming max std of 10 is very erratic)
            erratic_score = min(accel_std / 10.0, 1.0)
        else:
            erratic_score = 0.0

        return erratic_score

    def _calculate_overall_score(
        self,
        speed: float,
        pattern: float,
        activity: float,
        isolation: float,
        surface: float,
        bottom: float,
        erratic: float
    ) -> float:
        """Calculate overall behavioral health score"""
        # Penalize for abnormal values
        surface_penalty = max(0.0, surface - self.normal_surface_ratio) / (1.0 - self.normal_surface_ratio)
        bottom_penalty = max(0.0, bottom - self.normal_bottom_ratio) / (1.0 - self.normal_bottom_ratio)

        overall = (
            0.20 * pattern +
            0.20 * activity +
            0.15 * (1.0 - isolation) +  # Low isolation is good
            0.15 * (1.0 - surface_penalty) +
            0.15 * (1.0 - bottom_penalty) +
            0.15 * (1.0 - erratic)  # Low erratic is good
        )

        return np.clip(overall, 0.0, 1.0)

    def _classify_behavior(
        self,
        speed: float,
        pattern: float,
        activity: float,
        isolation: float,
        surface: float,
        bottom: float,
        erratic: float
    ) -> Tuple[BehaviorStatus, List[str]]:
        """Classify behavior and identify specific symptoms"""
        symptoms = []

        # Check for specific behavioral symptoms
        if speed < self.normal_speed_range[0]:
            symptoms.append("lethargy")
        if speed > self.normal_speed_range[1]:
            symptoms.append("hyperactivity")
        if pattern < 0.4:
            symptoms.append("irregular_swimming")
        if isolation > 0.6:
            symptoms.append("social_isolation")
        if surface > self.normal_surface_ratio + 0.2:
            symptoms.append("excessive_surface_time")
        if bottom > self.normal_bottom_ratio + 0.2:
            symptoms.append("bottom_sitting")
        if erratic > 0.5:
            symptoms.append("erratic_movements")

        # Calculate overall score
        overall = self._calculate_overall_score(
            speed, pattern, activity, isolation, surface, bottom, erratic
        )

        # Classify status
        if overall >= self.thresholds['normal']:
            status = BehaviorStatus.NORMAL
        elif overall >= self.thresholds['slight']:
            status = BehaviorStatus.SLIGHTLY_ABNORMAL
        elif overall >= self.thresholds['moderate']:
            status = BehaviorStatus.MODERATELY_ABNORMAL
        else:
            status = BehaviorStatus.SEVERELY_ABNORMAL

        return status, symptoms

    def _get_default_metrics(self) -> BehaviorMetrics:
        """Return default metrics when analysis fails"""
        return BehaviorMetrics(
            swimming_speed=0.0,
            swimming_pattern_score=0.5,
            activity_level_score=0.5,
            isolation_score=0.5,
            surface_time_ratio=0.0,
            bottom_time_ratio=0.0,
            erratic_movement_score=0.0,
            overall_behavior_score=0.5,
            behavior_status=BehaviorStatus.UNKNOWN,
            behavioral_symptoms=["insufficient_data"]
        )
