"""
Visual Health Assessment Module
Analyzes visual symptoms of fish to assess health condition
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from torchvision import transforms


class HealthStatus(Enum):
    """Health status categories"""
    HEALTHY = "healthy"
    MILD_CONCERN = "mild_concern"
    MODERATE_CONCERN = "moderate_concern"
    SEVERE_CONCERN = "severe_concern"
    UNKNOWN = "unknown"


@dataclass
class VisualHealthMetrics:
    """Visual health metrics for a fish"""
    color_score: float  # 0-1, measures color vibrancy
    texture_score: float  # 0-1, measures skin/scale abnormalities
    body_condition_score: float  # 0-1, measures body shape/symmetry
    fin_condition_score: float  # 0-1, measures fin integrity
    eye_clarity_score: float  # 0-1, measures eye clarity
    overall_health_score: float  # 0-1, weighted combination
    health_status: HealthStatus
    symptoms: List[str]


class VisualHealthAnalyzer:
    """
    Analyzes visual appearance of fish to detect health issues
    """

    def __init__(
        self,
        use_deep_features: bool = True,
        device: Optional[str] = None,
        skip_expensive_analysis: bool = True
    ):
        """
        Initialize visual health analyzer

        Args:
            use_deep_features: Whether to use deep learning features
            device: Device to run models on
            skip_expensive_analysis: Skip expensive operations (eye detection, fin analysis)
        """
        self.use_deep_features = use_deep_features
        self.skip_expensive_analysis = skip_expensive_analysis

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Health score thresholds
        self.thresholds = {
            'healthy': 0.7,
            'mild': 0.5,
            'moderate': 0.3
        }

        # Initialize transforms for deep learning
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_history: Optional[List] = None
    ) -> VisualHealthMetrics:
        """
        Analyze visual health of a fish

        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_history: Optional history for temporal analysis

        Returns:
            VisualHealthMetrics object
        """
        # Extract fish region
        x1, y1, x2, y2 = bbox
        fish_roi = frame[y1:y2, x1:x2]

        if fish_roi.size == 0:
            return self._get_default_metrics()

        # Analyze different health aspects
        color_score = self._analyze_color(fish_roi)
        texture_score = self._analyze_texture(fish_roi)
        body_score = self._analyze_body_condition(fish_roi)
        
        # Skip expensive operations if requested (for performance)
        if self.skip_expensive_analysis:
            # Use default values for expensive analyses
            fin_score = 0.7  # Default healthy fin score
            eye_score = 0.7  # Default healthy eye score
            
            # Calculate overall health score (weighted average, adjusted weights)
            overall_score = (
                0.35 * color_score +
                0.30 * texture_score +
                0.35 * body_score
            )
        else:
            # Full analysis including expensive operations
            fin_score = self._analyze_fin_condition(fish_roi)
            eye_score = self._analyze_eye_clarity(fish_roi)
            
            # Calculate overall health score (weighted average)
            overall_score = (
                0.25 * color_score +
                0.20 * texture_score +
                0.20 * body_score +
                0.20 * fin_score +
                0.15 * eye_score
            )

        # Determine health status and symptoms
        health_status, symptoms = self._classify_health_status(
            overall_score,
            color_score,
            texture_score,
            body_score,
            fin_score,
            eye_score
        )

        return VisualHealthMetrics(
            color_score=color_score,
            texture_score=texture_score,
            body_condition_score=body_score,
            fin_condition_score=fin_score,
            eye_clarity_score=eye_score,
            overall_health_score=overall_score,
            health_status=health_status,
            symptoms=symptoms
        )

    def _analyze_color(self, roi: np.ndarray) -> float:
        """
        Analyze fish color vibrancy and uniformity

        Healthy fish typically have vibrant, uniform coloration
        Sick fish may show faded colors, dark spots, or discoloration
        """
        if roi.size == 0:
            return 0.5

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Analyze saturation (vibrancy)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation) / 255.0

        # Analyze value (brightness)
        value = hsv[:, :, 2]
        mean_value = np.mean(value) / 255.0

        # Check for uniformity (less variation is better)
        color_variance = np.std(roi) / 255.0
        uniformity_score = 1.0 - min(color_variance / 50.0, 1.0)

        # Combine metrics
        color_score = (
            0.4 * mean_saturation +
            0.3 * mean_value +
            0.3 * uniformity_score
        )

        return np.clip(color_score, 0.0, 1.0)

    def _analyze_texture(self, roi: np.ndarray) -> float:
        """
        Analyze fish skin/scale texture for abnormalities

        Healthy fish have smooth, regular scale patterns
        Sick fish may show lesions, fungus, or irregular textures
        """
        if roi.size == 0:
            return 0.5

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Edge detection to find texture patterns
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Local Binary Pattern-like analysis for texture
        # High variance in local regions may indicate skin problems
        kernel_size = min(5, gray.shape[0] // 4, gray.shape[1] // 4)
        if kernel_size >= 3:
            local_std = cv2.blur(
                (gray - cv2.blur(gray, (kernel_size, kernel_size))) ** 2,
                (kernel_size, kernel_size)
            )
            texture_uniformity = 1.0 - min(np.mean(np.sqrt(local_std)) / 50.0, 1.0)
        else:
            texture_uniformity = 0.7

        # Combine metrics
        # Moderate edge density is good, too high suggests irregularities
        optimal_edge_density = 0.1
        edge_score = 1.0 - abs(edge_density - optimal_edge_density) / optimal_edge_density

        texture_score = 0.6 * texture_uniformity + 0.4 * edge_score

        return np.clip(texture_score, 0.0, 1.0)

    def _analyze_body_condition(self, roi: np.ndarray) -> float:
        """
        Analyze body shape and condition

        Healthy fish have proper body proportions and symmetry
        Sick fish may show bloating, emaciation, or deformities
        """
        if roi.size == 0:
            return 0.5

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold to get fish silhouette
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.5

        # Get largest contour (should be fish body)
        largest_contour = max(contours, key=cv2.contourArea)

        # Analyze aspect ratio (width/height ratio)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / max(h, 1)

        # Most fish have aspect ratios between 1.5 and 4.0
        if 1.5 <= aspect_ratio <= 4.0:
            aspect_score = 1.0
        else:
            aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 2.5) / 2.5)

        # Analyze compactness (perimeter^2 / area)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        if area > 0:
            compactness = (perimeter ** 2) / (4 * np.pi * area)
            # Circle has compactness of 1.0, elongated fish ~2-4
            compactness_score = max(0.0, 1.0 - abs(compactness - 3.0) / 3.0)
        else:
            compactness_score = 0.5

        body_score = 0.6 * aspect_score + 0.4 * compactness_score

        return np.clip(body_score, 0.0, 1.0)

    def _analyze_fin_condition(self, roi: np.ndarray) -> float:
        """
        Analyze fin condition

        Healthy fish have intact, spread fins
        Sick fish may show clamped, torn, or rotting fins
        """
        if roi.size == 0:
            return 0.5

        # This is a simplified analysis
        # In production, you'd want a trained model to detect fins

        # Edge detection for fin detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        # Fins typically appear at periphery
        h, w = edges.shape
        periphery = np.zeros_like(edges)
        border_width = max(2, min(h, w) // 10)

        periphery[:border_width, :] = edges[:border_width, :]
        periphery[-border_width:, :] = edges[-border_width:, :]
        periphery[:, :border_width] = edges[:, :border_width]
        periphery[:, -border_width:] = edges[:, -border_width:]

        fin_edge_density = np.sum(periphery > 0) / periphery.size

        # Moderate fin edge density indicates healthy fins
        optimal_density = 0.15
        fin_score = 1.0 - min(abs(fin_edge_density - optimal_density) / optimal_density, 1.0)

        return np.clip(fin_score, 0.3, 1.0)  # Baseline of 0.3

    def _analyze_eye_clarity(self, roi: np.ndarray) -> float:
        """
        Analyze eye clarity

        Healthy fish have clear, bright eyes
        Sick fish may show cloudy or sunken eyes
        """
        if roi.size == 0:
            return 0.5

        # This is simplified - in production, use eye detection
        # For now, look for bright circular regions that might be eyes

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Eyes are typically bright spots
        _, bright_regions = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Detect circles (potential eyes)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=2,
            maxRadius=min(roi.shape[0], roi.shape[1]) // 8
        )

        if circles is not None:
            eye_score = min(1.0, len(circles[0]) / 2.0)  # Expect ~2 eyes
        else:
            eye_score = 0.5  # Neutral if can't detect

        return np.clip(eye_score, 0.3, 1.0)

    def _classify_health_status(
        self,
        overall: float,
        color: float,
        texture: float,
        body: float,
        fin: float,
        eye: float
    ) -> Tuple[HealthStatus, List[str]]:
        """Classify health status and identify symptoms"""
        symptoms = []

        # Check individual metrics for specific symptoms
        if color < 0.4:
            symptoms.append("faded_coloration")
        if texture < 0.4:
            symptoms.append("skin_lesions_or_fungus")
        if body < 0.4:
            symptoms.append("abnormal_body_shape")
        if fin < 0.4:
            symptoms.append("fin_damage")
        if eye < 0.4:
            symptoms.append("cloudy_eyes")

        # Overall classification
        if overall >= self.thresholds['healthy']:
            status = HealthStatus.HEALTHY
        elif overall >= self.thresholds['mild']:
            status = HealthStatus.MILD_CONCERN
        elif overall >= self.thresholds['moderate']:
            status = HealthStatus.MODERATE_CONCERN
        else:
            status = HealthStatus.SEVERE_CONCERN

        return status, symptoms

    def _get_default_metrics(self) -> VisualHealthMetrics:
        """Return default metrics when analysis fails"""
        return VisualHealthMetrics(
            color_score=0.5,
            texture_score=0.5,
            body_condition_score=0.5,
            fin_condition_score=0.5,
            eye_clarity_score=0.5,
            overall_health_score=0.5,
            health_status=HealthStatus.UNKNOWN,
            symptoms=["unable_to_analyze"]
        )
