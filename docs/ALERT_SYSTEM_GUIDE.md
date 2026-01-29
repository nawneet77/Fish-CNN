# Alert System Guide

## Overview

The Fish Health Monitoring System includes an intelligent alert system that notifies you of potential health issues. This guide explains how the system works and how to avoid false alarms.

## Problem: Alert Spam (SOLVED ✅)

**Previous Issue**: The system could generate hundreds or thousands of false alerts because:
- It didn't know what "normal" looked like for your specific fish
- Alerts were generated every frame (30-60 times per second!)
- No deduplication = same alert repeated endlessly
- Generic thresholds didn't match your fish species/tank

**Example**: After 30 seconds of monitoring, you might see 1000+ "WARNING" alerts, even though your fish are perfectly healthy.

## Solution: Smart Alert System

The system now includes **three key features** to prevent false alarms:

### 1. Baseline Collection Mode (30 seconds)

**What it does**:
- Monitors your fish for 30 seconds when you first start
- Learns what "normal" behavior looks like for YOUR specific fish
- Calculates average health scores for each fish
- Does NOT generate alerts during this period

**Why it's important**:
- Every tank is different (species, lighting, water clarity, etc.)
- What's "normal" for goldfish ≠ what's "normal" for bettas
- Your tank's "normal" = your fish's baseline

**Visual Indicator**:
```
BASELINE MODE: Collecting normal behavior data... 45%
Alerts will activate after baseline collection (prevents false alarms)
```

### 2. Alert Deduplication (60-second cooldown)

**What it does**:
- Tracks when each alert was last generated for each fish
- Doesn't repeat the same alert within 60 seconds
- Prevents "alert spam" from single issues

**Example**:
- Frame 100: "WARNING - Fish #1 behavioral concern" ✅ Generated
- Frame 101-1800: Same issue detected, but alert suppressed ❌
- Frame 1801 (60 seconds later): Alert can be generated again ✅

**Result**: Maximum 1 alert per fish per issue per minute (instead of 1800 alerts per minute!)

### 3. Adjusted Thresholds

**Default Thresholds** (more lenient):
- **Critical Health**: < 0.3 (instead of 0.4)
- **Warning Health**: < 0.5 (unchanged)
- **Critical Behavior**: < 0.3 (instead of 0.4)
- **Warning Behavior**: < 0.5 (unchanged)

These thresholds are automatically calibrated using baseline data.

## How to Use

### Starting Monitoring (Recommended - With Baseline)

```bash
python examples/mac_camera_monitoring.py
```

1. System shows baseline collection info
2. Press ENTER to continue with baseline collection
3. Wait 30 seconds while system learns
4. Alert monitoring activates automatically

**Timeline**:
```
0s  ────────────► 30s ────────────────────────────►
   BASELINE MODE        ALERT MONITORING ACTIVE
   (no alerts)          (accurate alerts only)
```

### Skipping Baseline (Not Recommended)

If you're confident your fish are healthy and want immediate alerts:

```bash
python examples/mac_camera_monitoring.py
# When prompted, type 'skip' instead of pressing ENTER
```

**Warning**: Skipping baseline may cause false positives!

### Manual Control (Advanced)

```python
from src.health_monitor import FishHealthMonitor

monitor = FishHealthMonitor(
    detector_model_path="models/custom_trained/best.pt",
    detection_confidence=0.5
)

# Option 1: Use baseline (recommended)
# monitor.baseline_mode = True  # This is default

# Option 2: Skip baseline
monitor.skip_baseline_collection()

# Option 3: Check baseline status
status = monitor.get_baseline_status()
print(f"Collecting: {status['collecting']}")
print(f"Progress: {status['progress']}%")

# Option 4: Adjust cooldown period (default: 60 seconds)
monitor.alert_cooldown = 120.0  # 2 minutes between alerts

# Start monitoring
monitor.process_camera(camera_id=0, display=True)
```

## Alert Levels

The system generates three levels of alerts:

### 🔵 INFO
- **Purpose**: General information
- **Examples**:
  - "New fish detected - Fish #3"
  - "Baseline collection complete"
- **Action**: No immediate action needed

### 🟡 WARNING
- **Trigger**: Health score drops below 0.5
- **Examples**:
  - "Visual Health Concern - Fish #2"
  - "Behavioral Concern - Fish #1"
- **Action**: Monitor closely, check water quality

### 🔴 CRITICAL
- **Trigger**: Health score drops below 0.3
- **Examples**:
  - "Critical Visual Health Issue - Fish #1"
  - "Critical Behavioral Issue - Fish #2"
- **Action**: Immediate intervention needed
  - Check water parameters (pH, ammonia, nitrite)
  - Isolate if showing signs of disease
  - Consult aquatic veterinarian

## Understanding Alerts

### Visual Health Alerts

**What's analyzed**:
- Color vibrancy and uniformity
- Texture (lesions, fungus)
- Body condition and symmetry
- Fin condition and spread
- Eye clarity

**Common symptoms**:
- `faded_coloration` - May indicate stress or poor water quality
- `skin_lesions_or_fungus` - Requires immediate attention
- `clamped_fins` - Sign of stress or illness
- `cloudy_eyes` - Possible infection

### Behavioral Health Alerts

**What's analyzed**:
- Swimming speed (lethargy or hyperactivity)
- Swimming pattern (erratic movements)
- Activity level compared to baseline
- Social isolation
- Depth preference (excessive surface/bottom time)

**Common symptoms**:
- `lethargy` - Swimming slower than baseline
- `erratic_movement` - Unusual swimming patterns
- `isolation` - Avoiding other fish
- `gasping_at_surface` - Oxygen deficiency

## Calibrating Alert Sensitivity

### If you get too many false positives:

1. **Increase cooldown period**:
   ```python
   monitor.alert_cooldown = 120.0  # 2 minutes
   ```

2. **Adjust thresholds** (in `configs/mac_config.yaml`):
   ```yaml
   alerts:
     thresholds:
       critical_health: 0.2  # Only alert if score < 0.2 (very strict)
       warning_health: 0.35  # Only alert if score < 0.35
   ```

3. **Collect longer baseline**:
   ```python
   monitor.baseline_frames_needed = 1800  # 60 seconds at 30 FPS
   ```

### If you're missing real alerts:

1. **Decrease cooldown period**:
   ```python
   monitor.alert_cooldown = 30.0  # 30 seconds
   ```

2. **Adjust thresholds** (more sensitive):
   ```yaml
   alerts:
     thresholds:
       critical_health: 0.4
       warning_health: 0.6
   ```

3. **Check lighting and water clarity** - Poor visibility = poor detection

## Best Practices

### ✅ DO:
- **Always use baseline collection** for accurate alerts
- **Monitor during active hours** when fish are naturally active
- **Ensure good lighting** for accurate visual analysis
- **Keep water clear** for better detection
- **Review alerts daily** to catch patterns
- **Act on CRITICAL alerts immediately**

### ❌ DON'T:
- **Don't skip baseline** unless you're very confident
- **Don't panic on single WARNING** - wait for patterns
- **Don't ignore repeated CRITICAL alerts**
- **Don't rely solely on automation** - also do visual checks
- **Don't forget about lighting/glare** affecting detection

## Troubleshooting

### "Still getting too many alerts after baseline"

**Possible causes**:
1. **Tank conditions changed** - Water quality, lighting, temperature
2. **New fish added** - Baseline doesn't include them
3. **Fish are actually unwell** - Alerts may be accurate!

**Solutions**:
- Reset system and recollect baseline: `monitor.reset()`
- Check actual water parameters with test kit
- Increase alert cooldown: `monitor.alert_cooldown = 180.0`

### "Not getting any alerts, even when fish looks sick"

**Possible causes**:
1. **Still in baseline mode** - Wait for collection to complete
2. **Baseline included sick fish** - System thinks illness is "normal"
3. **Thresholds too lenient** - Not triggering on issues

**Solutions**:
- Check baseline status: `monitor.get_baseline_status()`
- Collect new baseline when fish are healthy
- Adjust thresholds to be more sensitive (see above)

### "Baseline never completes"

**Possible causes**:
1. **No fish detected** - Model can't see your fish
2. **Frame rate too low** - Taking longer than expected
3. **System error** - Check terminal for errors

**Solutions**:
- Verify fish detection: Run `examples/test_custom_model.py`
- Improve lighting and camera position
- Skip baseline if needed: `monitor.skip_baseline_collection()`

## Alert History

View alert history:

```python
# Get all alerts
all_alerts = monitor.alert_system.get_alert_history()

# Get active alerts only
active_alerts = monitor.alert_system.get_active_alerts()

# Get alert summary
summary = monitor.alert_system.get_alert_summary()
print(f"Total active: {summary['total_active']}")
print(f"Critical: {summary['critical']}")
print(f"Warning: {summary['warning']}")
```

Alerts are also stored in SQLite database: `data/alerts.db`

## Advanced: Custom Alert Logic

Create custom alert handlers:

```python
def my_alert_handler(alert):
    """Custom alert handler"""
    if alert.alert_level == AlertLevel.CRITICAL:
        # Send email, SMS, webhook, etc.
        send_emergency_notification(alert)
    elif alert.alert_level == AlertLevel.WARNING:
        # Log to file
        log_warning(alert)

# Process frame with custom handler
vis_frame, reports, alerts = monitor.process_frame(frame)
for alert in alerts:
    my_alert_handler(alert)
```

## Summary

The new alert system with baseline collection, deduplication, and smart thresholds solves the "1000+ false warnings" problem:

**Before** ❌:
- 1000+ alerts in 30 seconds
- All false positives
- System unusable

**After** ✅:
- 30-second baseline collection
- 0-5 alerts per minute (only real issues)
- Accurate and actionable

**Key takeaway**: Always use baseline collection for best results!
