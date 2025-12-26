"""
Alert Generation and Notification System
Generates alerts based on health assessments and manages notifications
"""

import json
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from collections import deque
import sqlite3
from pathlib import Path


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of health alerts"""
    VISUAL_HEALTH = "visual_health"
    BEHAVIORAL_HEALTH = "behavioral_health"
    COMBINED_HEALTH = "combined_health"
    FISH_LOST = "fish_lost"
    NEW_FISH = "new_fish"
    SYSTEM = "system"


@dataclass
class Alert:
    """Represents a health alert"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    alert_level: AlertLevel
    fish_id: Optional[int]
    title: str
    message: str
    metrics: Dict
    recommendations: List[str]
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['alert_type'] = self.alert_type.value
        data['alert_level'] = self.alert_level.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class AlertSystem:
    """
    Manages health alert generation and notifications
    """

    def __init__(
        self,
        db_path: str = "alerts.db",
        alert_history_size: int = 1000,
        notification_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize alert system

        Args:
            db_path: Path to SQLite database for alert storage
            alert_history_size: Number of alerts to keep in memory
            notification_callbacks: List of functions to call when alert is generated
        """
        self.db_path = db_path
        self.alert_history = deque(maxlen=alert_history_size)
        self.notification_callbacks = notification_callbacks or []
        self.alert_counter = 0

        # Initialize database
        self._init_database()

        # Alert thresholds
        self.thresholds = {
            'critical_health': 0.3,
            'warning_health': 0.5,
            'critical_behavior': 0.3,
            'warning_behavior': 0.5
        }

    def _init_database(self):
        """Initialize SQLite database for alert storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TEXT,
                alert_type TEXT,
                alert_level TEXT,
                fish_id INTEGER,
                title TEXT,
                message TEXT,
                metrics TEXT,
                recommendations TEXT,
                acknowledged INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    def generate_alert(
        self,
        alert_type: AlertType,
        alert_level: AlertLevel,
        title: str,
        message: str,
        fish_id: Optional[int] = None,
        metrics: Optional[Dict] = None,
        recommendations: Optional[List[str]] = None
    ) -> Alert:
        """
        Generate a new alert

        Args:
            alert_type: Type of alert
            alert_level: Severity level
            title: Alert title
            message: Alert message
            fish_id: ID of affected fish
            metrics: Associated metrics
            recommendations: List of recommended actions

        Returns:
            Generated Alert object
        """
        alert_id = f"ALERT_{int(time.time())}_{self.alert_counter}"
        self.alert_counter += 1

        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            alert_type=alert_type,
            alert_level=alert_level,
            fish_id=fish_id,
            title=title,
            message=message,
            metrics=metrics or {},
            recommendations=recommendations or []
        )

        # Store alert
        self.alert_history.append(alert)
        self._save_to_database(alert)

        # Trigger notifications
        self._notify(alert)

        return alert

    def check_visual_health(
        self,
        fish_id: int,
        visual_metrics: 'VisualHealthMetrics'
    ) -> Optional[Alert]:
        """
        Check visual health metrics and generate alert if needed

        Args:
            fish_id: Fish identifier
            visual_metrics: Visual health metrics

        Returns:
            Alert if generated, None otherwise
        """
        score = visual_metrics.overall_health_score

        if score < self.thresholds['critical_health']:
            level = AlertLevel.CRITICAL
            title = f"Critical Visual Health Issue Detected - Fish #{fish_id}"
        elif score < self.thresholds['warning_health']:
            level = AlertLevel.WARNING
            title = f"Visual Health Concern - Fish #{fish_id}"
        else:
            return None

        # Build message
        message = f"Fish #{fish_id} shows visual health concerns:\n"
        message += f"Overall Health Score: {score:.2f}\n"
        message += f"Status: {visual_metrics.health_status.value}\n"

        if visual_metrics.symptoms:
            message += f"Symptoms: {', '.join(visual_metrics.symptoms)}\n"

        # Get recommendations
        recommendations = self._get_visual_health_recommendations(visual_metrics)

        return self.generate_alert(
            alert_type=AlertType.VISUAL_HEALTH,
            alert_level=level,
            title=title,
            message=message,
            fish_id=fish_id,
            metrics=asdict(visual_metrics),
            recommendations=recommendations
        )

    def check_behavioral_health(
        self,
        fish_id: int,
        behavior_metrics: 'BehaviorMetrics'
    ) -> Optional[Alert]:
        """
        Check behavioral health metrics and generate alert if needed

        Args:
            fish_id: Fish identifier
            behavior_metrics: Behavioral health metrics

        Returns:
            Alert if generated, None otherwise
        """
        score = behavior_metrics.overall_behavior_score

        if score < self.thresholds['critical_behavior']:
            level = AlertLevel.CRITICAL
            title = f"Critical Behavioral Issue Detected - Fish #{fish_id}"
        elif score < self.thresholds['warning_behavior']:
            level = AlertLevel.WARNING
            title = f"Behavioral Concern - Fish #{fish_id}"
        else:
            return None

        # Build message
        message = f"Fish #{fish_id} shows behavioral concerns:\n"
        message += f"Overall Behavior Score: {score:.2f}\n"
        message += f"Status: {behavior_metrics.behavior_status.value}\n"

        if behavior_metrics.behavioral_symptoms:
            message += f"Symptoms: {', '.join(behavior_metrics.behavioral_symptoms)}\n"

        # Get recommendations
        recommendations = self._get_behavioral_recommendations(behavior_metrics)

        return self.generate_alert(
            alert_type=AlertType.BEHAVIORAL_HEALTH,
            alert_level=level,
            title=title,
            message=message,
            fish_id=fish_id,
            metrics=asdict(behavior_metrics),
            recommendations=recommendations
        )

    def _get_visual_health_recommendations(self, metrics: 'VisualHealthMetrics') -> List[str]:
        """Generate recommendations based on visual symptoms"""
        recommendations = []

        if "faded_coloration" in metrics.symptoms:
            recommendations.append("Check water quality (pH, ammonia, nitrite levels)")
            recommendations.append("Review diet - fish may need more varied nutrition")

        if "skin_lesions_or_fungus" in metrics.symptoms:
            recommendations.append("URGENT: Isolate fish to quarantine tank")
            recommendations.append("Consult with aquatic veterinarian")
            recommendations.append("Consider antifungal treatment")

        if "fin_damage" in metrics.symptoms:
            recommendations.append("Check for aggressive tank mates")
            recommendations.append("Inspect tank for sharp objects")
            recommendations.append("Improve water quality to promote healing")

        if "cloudy_eyes" in metrics.symptoms:
            recommendations.append("Test water parameters immediately")
            recommendations.append("Check for bacterial infection")

        if "abnormal_body_shape" in metrics.symptoms:
            recommendations.append("Monitor feeding - may indicate bloating or dropsy")
            recommendations.append("Reduce feeding temporarily")

        if not recommendations:
            recommendations.append("Monitor closely for 24-48 hours")
            recommendations.append("Maintain optimal water conditions")

        return recommendations

    def _get_behavioral_recommendations(self, metrics: 'BehaviorMetrics') -> List[str]:
        """Generate recommendations based on behavioral symptoms"""
        recommendations = []

        if "lethargy" in metrics.behavioral_symptoms:
            recommendations.append("Check water temperature")
            recommendations.append("Test oxygen levels")
            recommendations.append("Observe for other symptoms")

        if "hyperactivity" in metrics.behavioral_symptoms:
            recommendations.append("Check for water quality issues")
            recommendations.append("Ensure proper tank aeration")
            recommendations.append("Look for signs of parasites")

        if "social_isolation" in metrics.behavioral_symptoms:
            recommendations.append("Monitor for bullying by other fish")
            recommendations.append("Watch for development of other symptoms")

        if "excessive_surface_time" in metrics.behavioral_symptoms:
            recommendations.append("URGENT: Check oxygen levels")
            recommendations.append("Increase aeration")
            recommendations.append("Test for ammonia/nitrite")

        if "bottom_sitting" in metrics.behavioral_symptoms:
            recommendations.append("Check for swim bladder issues")
            recommendations.append("Review feeding practices")

        if "erratic_movements" in metrics.behavioral_symptoms:
            recommendations.append("Check for parasites (flashing behavior)")
            recommendations.append("Test water parameters")

        if not recommendations:
            recommendations.append("Continue monitoring behavior")
            recommendations.append("Document any changes")

        return recommendations

    def _save_to_database(self, alert: Alert):
        """Save alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.timestamp.isoformat(),
            alert.alert_type.value,
            alert.alert_level.value,
            alert.fish_id,
            alert.title,
            alert.message,
            json.dumps(alert.metrics),
            json.dumps(alert.recommendations),
            int(alert.acknowledged)
        ))

        conn.commit()
        conn.close()

    def _notify(self, alert: Alert):
        """Send notifications via registered callbacks"""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Notification callback error: {e}")

    def acknowledge_alert(self, alert_id: str):
        """Mark alert as acknowledged"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True

                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?',
                    (alert_id,)
                )
                conn.commit()
                conn.close()
                break

    def get_active_alerts(self, fish_id: Optional[int] = None) -> List[Alert]:
        """Get unacknowledged alerts"""
        alerts = [a for a in self.alert_history if not a.acknowledged]

        if fish_id is not None:
            alerts = [a for a in alerts if a.fish_id == fish_id]

        return alerts

    def get_alert_summary(self) -> Dict:
        """Get summary of current alerts"""
        active = self.get_active_alerts()

        summary = {
            'total_active': len(active),
            'critical': len([a for a in active if a.alert_level == AlertLevel.CRITICAL]),
            'warning': len([a for a in active if a.alert_level == AlertLevel.WARNING]),
            'info': len([a for a in active if a.alert_level == AlertLevel.INFO]),
            'by_type': {}
        }

        for alert_type in AlertType:
            count = len([a for a in active if a.alert_type == alert_type])
            if count > 0:
                summary['by_type'][alert_type.value] = count

        return summary

    def add_notification_callback(self, callback: Callable):
        """Add notification callback"""
        self.notification_callbacks.append(callback)

    def remove_notification_callback(self, callback: Callable):
        """Remove notification callback"""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
