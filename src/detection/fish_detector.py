"""
Fish Detection Module using YOLOv8
Detects and localizes multiple fish in video frames
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from ultralytics import YOLO
import torch


@dataclass
class Detection:
    """Represents a single fish detection"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    timestamp: float

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def area(self) -> int:
        """Calculate area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        """Get width of bounding box"""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Get height of bounding box"""
        return self.bbox[3] - self.bbox[1]


class FishDetector:
    """
    Fish detector using YOLOv8 for real-time multi-fish detection
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        filter_classes: Optional[List[int]] = None
    ):
        """
        Initialize fish detector

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            filter_classes: List of class IDs to EXCLUDE (e.g., [0] to filter out 'person')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Auto-select device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'  # Apple Silicon GPU
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Frame counter
        self.frame_count = 0

        # Class filtering for generic models
        # Default: filter out 'person' (class 0 in COCO) to avoid detecting humans as fish!
        if filter_classes is None and model_path in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
            # Using generic COCO model - filter out common non-fish classes
            self.filter_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
            print("⚠️  WARNING: Using generic YOLO model (trained on people/cars, NOT fish)")
            print("   Filtering out: person, bicycle, car, motorcycle, bus, truck")
            print("   For accurate fish detection, fine-tune on your tank!")
            print("   See: docs/FINE_TUNING_GUIDE.md")
        else:
            self.filter_classes = filter_classes or []

    def detect(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[Detection]:
        """
        Detect fish in a single frame

        Args:
            frame: Input image frame (BGR format)
            timestamp: Optional timestamp for the frame

        Returns:
            List of Detection objects
        """
        if timestamp is None:
            timestamp = self.frame_count / 30.0  # Assume 30 FPS

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]

                # Filter out unwanted classes (e.g., person, car, etc.)
                if cls_id in self.filter_classes:
                    continue

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    frame_id=self.frame_count,
                    timestamp=timestamp
                )

                detections.append(detection)

        self.frame_count += 1
        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[float]] = None
    ) -> List[List[Detection]]:
        """
        Detect fish in a batch of frames (more efficient)

        Args:
            frames: List of input image frames
            timestamps: Optional list of timestamps

        Returns:
            List of detection lists, one per frame
        """
        if timestamps is None:
            timestamps = [self.frame_count / 30.0 + i / 30.0 for i in range(len(frames))]

        # Run batch inference
        results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        all_detections = []

        for frame_idx, (result, timestamp) in enumerate(zip(results, timestamps)):
            frame_detections = []
            boxes = result.boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]

                # Filter out unwanted classes
                if cls_id in self.filter_classes:
                    continue

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    frame_id=self.frame_count + frame_idx,
                    timestamp=timestamp
                )

                frame_detections.append(detection)

            all_detections.append(frame_detections)

        self.frame_count += len(frames)
        return all_detections

    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_id: bool = False
    ) -> np.ndarray:
        """
        Draw detections on frame

        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores
            show_id: Whether to show detection IDs

        Returns:
            Frame with drawn detections
        """
        vis_frame = frame.copy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label
            label = det.class_name
            if show_confidence:
                label += f" {det.confidence:.2f}"
            if show_id:
                label = f"ID:{idx} {label}"

            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_frame,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

            # Draw center point
            center = det.center
            cv2.circle(vis_frame, center, 3, (0, 0, 255), -1)

        return vis_frame

    def reset_frame_count(self):
        """Reset the frame counter"""
        self.frame_count = 0
