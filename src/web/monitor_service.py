"""
Monitor Service — bridges FishHealthMonitor with WebSocket clients.
Runs the monitor in a background thread and broadcasts frames + health data.
"""

import asyncio
import threading
import time
import base64
import cv2
import numpy as np
from typing import Optional


class MonitorService:
    def __init__(self, source, monitor, connection_manager, target_fps: float = 15.0):
        """
        Args:
            source: video file path (str) or camera ID (int)
            monitor: FishHealthMonitor instance
            connection_manager: WebSocket ConnectionManager
            target_fps: max frames per second to broadcast
        """
        self.source = source
        self.monitor = monitor
        self.manager = connection_manager
        self.target_fps = target_fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.latest_frame_b64: Optional[str] = None
        self.latest_health_data: Optional[dict] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._run_capture, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def apply_tuning(self, param: str, value):
        """Apply a tuning parameter change from the dashboard sliders."""
        if param == "confidence":
            self.monitor.detector.confidence_threshold = float(value)
        elif param == "nms_iou":
            self.monitor.detector.iou_threshold = float(value)
        elif param == "max_age":
            self.monitor.tracker.max_age = int(value)
        elif param == "min_hits":
            self.monitor.tracker.min_hits = int(value)
        elif param == "tracker_iou":
            self.monitor.tracker.iou_threshold = float(value)
        elif param == "appearance_weight":
            self.monitor.tracker.appearance_weight = float(value)

    def _run_capture(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source: {self.source}")
            return

        frame_interval = 1.0 / self.target_fps

        while self._running and cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    # Loop video files
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.monitor.reset()
                    continue
                break

            try:
                vis_frame, reports, alerts = self.monitor.process_frame(frame)
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue

            # Encode frame as JPEG -> base64
            _, jpeg = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            # Build payload
            health_data = self._build_payload(reports, alerts)

            self.latest_frame_b64 = frame_b64
            self.latest_health_data = health_data

            payload = {
                "type": "frame_update",
                "frame": frame_b64,
                "health": health_data,
            }

            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast(payload),
                    self._loop,
                )

            # Throttle
            elapsed = time.time() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        cap.release()

    def _build_payload(self, reports, alerts) -> dict:
        fish_cards = []
        for r in reports:
            bl = self.monitor._fish_baselines.get(r.track_id)
            fish_cards.append({
                "id": r.track_id,
                "score": round(r.overall_health_score, 3),
                "status": self.monitor._fish_status.get(r.track_id, "Unknown"),
                "visual_score": round(r.visual_health.overall_health_score, 3),
                "behavior_score": round(r.behavioral_health.overall_behavior_score, 3),
                "score_history": self.monitor.get_score_history(r.track_id)[-20:],
                "confidence": round(r.confidence, 2),
                "reasons": self.monitor._concern_reasons.get(r.track_id, []),
                "baseline_ready": bl.is_ready if bl else False,
            })

        alert_list = [a.to_dict() for a in alerts]
        active_alerts = [a.to_dict() for a in self.monitor.alert_system.get_active_alerts()[-20:]]

        summary = self.monitor.get_health_summary()
        baseline = self.monitor.get_baseline_status()

        avg_time = np.mean(self.monitor.processing_times) if self.monitor.processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "fish": fish_cards,
            "new_alerts": alert_list,
            "active_alerts": active_alerts,
            "summary": summary,
            "baseline": baseline,
            "stats": {
                "fps": round(fps, 1),
                "fish_count": len(fish_cards),
                "frame_count": self.monitor.frame_count,
            },
            "tuning": {
                "confidence": round(self.monitor.detector.confidence_threshold, 2),
                "nms_iou": round(self.monitor.detector.iou_threshold, 2),
                "max_age": self.monitor.tracker.max_age,
                "min_hits": self.monitor.tracker.min_hits,
                "tracker_iou": round(self.monitor.tracker.iou_threshold, 2),
                "appearance_weight": round(self.monitor.tracker.appearance_weight, 2),
            },
        }
