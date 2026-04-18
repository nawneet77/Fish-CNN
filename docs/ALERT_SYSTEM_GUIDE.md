# Alert System Architecture Guide

## Overview

The Fish Health Monitoring System uses a **multi-stage intelligent pipeline** to detect and report fish health anomalies in real-time. Rather than relying on fixed thresholds, the system learns each fish's individual behavioral baseline and uses statistical anomaly detection to identify genuine health concerns while minimizing false alarms.

```
Camera/Video
    |
    v
 YOLOv8 Detection -----> SORT Tracker (Kalman + Re-ID)
                              |
              +---------------+----------------+
              |                                |
     Visual Health Analyzer          Behavior Analyzer
     (Color, Texture, Shape)         (Speed, Pattern, Social)
              |                                |
              +----------- Combined -----------+
                               |
                   Per-Fish Adaptive Baseline
                     (Welford's Algorithm)
                               |
                     Z-Score Anomaly Detection
                               |
                     EMA Smoothing (alpha=0.15)
                               |
                     Hysteresis State Machine
                     (Healthy/Concern/Warning/Critical)
                               |
                     Multi-Frame Sustained Check
                               |
                     Alert Generation + Dashboard
```

---

## Stage 1: Detection & Tracking

### Fish Detection (YOLOv8)

Every frame is processed by a YOLOv8 object detector trained specifically on aquarium fish footage (mAP50: 92.5%). The detector outputs bounding boxes with confidence scores for each fish.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Confidence Threshold | 0.35 | Minimum detection certainty |
| NMS IoU Threshold | 0.50 | Duplicate box suppression |

### Fish Tracking (Enhanced SORT)

Detections are linked across frames using an enhanced SORT (Simple Online and Realtime Tracking) algorithm with:

- **Kalman Filter**: Predicts fish position using an 8-dimensional state vector `[x, y, w, h, vx, vy, vw, vh]` covering position, size, and velocity.
- **Appearance Re-ID**: Extracts 40-dimensional HSV color histograms + spatial color moments per fish. When a fish disappears and reappears, it can be re-identified by its appearance (not just position).
- **Hungarian Algorithm**: Optimal assignment matching detections to tracks using combined IoU + appearance similarity.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Max Age | 360 | Frames to keep lost tracks alive (~12s at 30fps) |
| Min Hits | 6 | Detections needed to confirm a track |
| Tracker IoU | 0.20 | Minimum overlap for detection-to-track matching |
| Appearance Weight | 0.45 | Balance: 55% position + 45% appearance |

---

## Stage 2: Health Assessment

Each tracked fish is assessed on two independent dimensions.

### Visual Health Analysis

The visual analyzer extracts the fish's bounding box region and computes five features:

| Feature | Method | Weight | What It Detects |
|---------|--------|--------|-----------------|
| **Color Score** | HSV saturation + brightness + uniformity | 35% | Faded coloration, discoloration |
| **Texture Score** | Canny edge density + local variance | 30% | Lesions, fungus, scale damage |
| **Body Condition** | Aspect ratio + contour compactness | 35% | Bloating, emaciation, deformities |
| Fin Condition* | Peripheral edge analysis | 20% | Fin rot, fin damage |
| Eye Clarity* | Hough circle detection | 15% | Cloudy eyes, infections |

*\*Fin and eye analysis are computationally expensive and disabled by default for real-time performance. When enabled, all five features are used with adjusted weights (25/20/20/20/15).*

**Color Analysis Algorithm:**
```
HSV = convert(fish_roi, BGR -> HSV)
saturation = mean(S channel) / 255
brightness = mean(V channel) / 255
uniformity = 1.0 - min(std(roi) / 50, 1.0)
color_score = 0.4 * saturation + 0.3 * brightness + 0.3 * uniformity
```

### Behavioral Health Analysis

The behavior analyzer uses the fish's position history (last 100 frames) and computes seven metrics:

| Metric | What It Measures | Healthy Range |
|--------|------------------|---------------|
| **Swimming Speed** | Average pixels/frame displacement | 1.0 - 15.0 px/frame |
| **Swimming Pattern** | Directional smoothness (60%) + path efficiency (40%) | > 0.7 |
| **Activity Level** | Speed relative to normal range | 1.0 (optimal) |
| **Social Isolation** | Distance to nearest neighbor, normalized by tank size | < 0.5 |
| **Surface Time** | Proportion of time in top 20% of tank | < 20% |
| **Bottom Time** | Proportion of time in bottom 20% of tank | < 30% |
| **Erratic Movement** | Acceleration variance (jerkiness) | < 0.3 |

**Overall Behavior Score:**
```
surface_penalty = max(0, surface_ratio - 0.2) / 0.8
bottom_penalty  = max(0, bottom_ratio - 0.3) / 0.7

overall = 0.20 * pattern
        + 0.20 * activity
        + 0.15 * (1 - isolation)
        + 0.15 * (1 - surface_penalty)
        + 0.15 * (1 - bottom_penalty)
        + 0.15 * (1 - erratic)
```

**Dynamic Tank Dimensions:** The system uses actual video frame dimensions (not hardcoded values) for isolation distance normalization and surface/bottom zone calculations.

---

## Stage 3: Adaptive Baseline Learning

### Why Per-Fish Baselines?

Different fish species and individuals have vastly different normal behaviors. A naturally slow-moving goldfish would constantly trigger "lethargy" alerts if compared against universal thresholds. Instead, the system learns what is normal for **each individual fish**.

### Welford's Online Algorithm

Each fish maintains its own `FishBaseline` that tracks running mean and variance for 9 metrics using Welford's numerically stable online algorithm:

```
On each new observation x:
    n = n + 1
    delta = x - mean
    mean = mean + delta / n
    delta2 = x - mean
    M2 = M2 + delta * delta2

Variance = M2 / (n - 1)
Std = sqrt(Variance)
```

**Properties:**
- O(1) memory per metric per fish (no history storage needed)
- Numerically stable for streaming data
- Variance converges after ~30 samples

**Baseline readiness:** A fish's baseline becomes "ready" after 150 samples (~25 seconds at 30fps with analysis every 5 frames). Before this, the system uses conservative fixed weights (30% visual + 70% behavioral).

### Body-Length Speed Normalization

Raw swimming speed (pixels/frame) depends on fish size and camera distance. The system normalizes speed by the fish's bounding box diagonal:

```
body_length = sqrt((x2-x1)^2 + (y2-y1)^2)
normalized_speed = raw_speed / body_length * 100
```

This makes speed measurements comparable across fish of different sizes.

---

## Stage 4: Z-Score Anomaly Detection

Once a fish's baseline is ready, health scoring switches from absolute thresholds to **relative anomaly detection**: "Is this fish behaving differently from its OWN normal?"

### Z-Score Computation

For each of the 9 metrics:
```
z = |current_value - baseline_mean| / max(baseline_std, 0.01)
```

A Z-score of 2.0 means the fish is behaving 2 standard deviations from its personal normal -- statistically significant.

### Multi-Indicator Consensus

Instead of alerting on a single anomalous metric (which could be noise), the system uses **weighted multi-indicator consensus**:

| Metric | Weight | Category |
|--------|--------|----------|
| Swimming Speed | 15% | Behavioral |
| Activity Level | 12% | Behavioral |
| Swimming Pattern | 12% | Behavioral |
| Erratic Movement | 12% | Behavioral |
| Color/Appearance | 12% | Visual |
| Body Shape | 11% | Visual |
| Social Isolation | 10% | Behavioral |
| Surface Time | 8% | Behavioral |
| Bottom Time | 8% | Behavioral |

**Total: 73% Behavioral + 27% Visual** -- behavioral indicators are weighted more heavily because research shows behavioral changes precede visible symptoms by 4-5 days.

### Penalty Calculation

```
for each metric:
    if z_score > 1.5:
        penalty += (z_score - 1.5) * weight

health_score = max(0.0, 1.0 - penalty / total_weights)
```

**Thresholds:**
- Z > 1.5: Penalty applied (metric is unusual)
- Z > 2.0: Added to concern reasons (metric is significantly anomalous)

### Human-Readable Reasons

When Z > 2.0, the system generates plain English descriptions:

| Metric | Concern Reason |
|--------|---------------|
| speed | "Unusual swimming speed" |
| activity | "Abnormal activity level" |
| pattern | "Irregular swimming pattern" |
| isolation | "Isolating from other fish" |
| surface | "Too much time at surface" |
| bottom | "Sitting at bottom" |
| erratic | "Erratic/jerky movements" |
| color | "Color change detected" |
| body_shape | "Body shape looks different" |

---

## Stage 5: Score Stabilization

### Exponential Moving Average (EMA)

Raw health scores are smoothed using an EMA to prevent rapid fluctuations:

```
EMA_t = alpha * raw_score + (1 - alpha) * EMA_{t-1}
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| alpha | 0.15 | 15% new data, 85% history |
| Response time | ~7 seconds | Time to reach 63% of a new trend |
| History window | 30 values | Stored for sparkline display |

### Hysteresis State Machine

To prevent status flickering (e.g., toggling between "Healthy" and "Concern" every few seconds), the system uses dead-band hysteresis -- different thresholds for entering and leaving each state:

```
                    +---------------------------------------+
                    |                                       |
    Score 1.0  -----+         HEALTHY                      |
                    |                                       |
    Score 0.70 -----+ - - - - - - - - recovery threshold - |
                    |         (must reach 0.70 to recover)  |
    Score 0.60 -----+ - - - - - - - - entry threshold - -  |
                    |                                       |
                    |         CONCERN                       |
                    |                                       |
    Score 0.55 -----+ - - - - - - - - recovery threshold   |
    Score 0.40 -----+ - - - - - - - - entry threshold      |
                    |                                       |
                    |         WARNING                       |
                    |                                       |
    Score 0.35 -----+ - - - - - - - - recovery threshold   |
    Score 0.25 -----+ - - - - - - - - entry threshold      |
                    |                                       |
                    |         CRITICAL                      |
                    |                                       |
    Score 0.0  -----+---------------------------------------+
```

**Dead-band gaps prevent flickering:**

| Transition | Threshold | Gap |
|-----------|-----------|-----|
| Healthy -> Concern | Score drops below 0.60 | |
| Concern -> Healthy | Score rises above 0.70 | 10-point dead band |
| Concern -> Warning | Score drops below 0.40 | |
| Warning -> Concern | Score rises above 0.55 | 15-point dead band |
| Warning -> Critical | Score drops below 0.25 | |
| Critical -> Warning | Score rises above 0.35 | 10-point dead band |

---

## Stage 6: Alert Generation

Alerts pass through a **four-stage filter** before reaching the user:

### Filter 1: Baseline Collection

During the first 300 frames (~10 seconds), the system is collecting baseline data. No alerts are generated regardless of scores.

### Filter 2: Track Maturity

Only fish tracked for 30+ consecutive frames (well-established tracks) can trigger alerts. This prevents false alarms from brief misdetections.

### Filter 3: Sustained Threshold

The system counts consecutive analysis cycles where the smoothed score falls below the warning threshold (0.50). An alert is only generated after **4 consecutive** low cycles. Combined with EMA smoothing, this means a genuinely concerning situation must persist for several seconds.

```
if smoothed_score < 0.50:
    consecutive_low[fish_id] += 1
else:
    consecutive_low[fish_id] = 0

if consecutive_low[fish_id] >= 4:
    generate_alert(...)
```

### Filter 4: Alert Cooldown

Once an alert fires for a fish, no new alert of the same type will fire for 60 seconds. This prevents alert spam.

### Alert Severity

| Level | Trigger | Color |
|-------|---------|-------|
| WARNING | Smoothed score < 0.50 | Amber |
| CRITICAL | Smoothed score < 0.25 | Red |

### Alert Content

Each alert includes:
- **Title**: "Health Concern - Fish #4 (3 indicators)"
- **Message**: "Fish #4: Irregular swimming pattern; Color change detected; Sitting at bottom"
- **Recommendations**: Actionable advice (e.g., "Check water quality", "Monitor closely")
- **Timestamp**: When the alert was generated

### Alert Storage

Alerts are stored in both:
- **In-memory deque** (last 1000 alerts) for fast access
- **SQLite database** (`alerts.db`) for persistence and history

---

## Stage 7: Dashboard & Visualization

### Live Video Feed

The processed video frame shows:
- Color-coded bounding boxes (green/amber/orange/red by health status)
- Fish ID labels (`Fish:0`, `Fish:1`, etc.) above each box
- Movement trails (last 30 positions)

### Fish Health Cards

Each tracked fish gets a card showing:
- Health score percentage with status color
- Visual and Behavioral sub-scores
- Sparkline chart of recent score history
- Concern reasons (when status is not Healthy)

Cards update in-place (no DOM rebuild) to prevent visual flickering.

### Alert Feed

Thin, scrollable alert rows showing:
- Severity badge (WARNING/CRITICAL)
- Alert message with specific reasons
- Timestamp

Only rebuilds when the alert list actually changes (change-detection via ID hashing).

### Live Tuning

Six real-time adjustable parameters via WebSocket:

| Slider | Range | Default | Affects |
|--------|-------|---------|---------|
| Confidence | 0.10 - 0.90 | 0.35 | Detection sensitivity |
| NMS IoU | 0.10 - 0.80 | 0.50 | Duplicate suppression |
| Max Age | 30 - 500 | 360 | Track persistence |
| Min Hits | 1 - 10 | 6 | Track confirmation speed |
| Tracker IoU | 0.05 - 0.50 | 0.20 | Matching strictness |
| Appearance Wt | 0.00 - 1.00 | 0.45 | Position vs. appearance balance |

---

## Complete Data Flow

```
Frame (BGR) ---------------------------------------------------------------->
    |
    +-> YOLOv8 --> Detections[] --> SORT Tracker --> Active Tracks[]
    |                                                      |
    |                              +--------- -----------+
    |                              |                       |
    |                    Visual Analyzer            Behavior Analyzer
    |                    (5 features)               (7 metrics)
    |                              |                       |
    |                              +---- Combined ---------+
    |                                       |
    |                          Welford's Baseline Update (per fish)
    |                                       |
    |                          Z-Score Anomaly Detection (9 metrics)
    |                                       |
    |                          EMA Smoothing (alpha=0.15)
    |                                       |
    |                          Hysteresis State Machine
    |                                       |
    |                          Sustained Alert Check (4 frames)
    |                                       |
    |                          Alert Generation + SQLite Storage
    |                                       |
    +--------------- Visualization ---------+
                                            |
                               Monitor Service Thread
                               (JPEG encode + JSON payload)
                                            |
                               WebSocket Broadcast (15fps)
                                            |
                               Browser Dashboard
                               (Video + Cards + Alerts + Sliders)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Per-fish baselines** over absolute thresholds | Different species/individuals have different "normal" -- a slow fish isn't sick |
| **70/30 behavioral/visual** weighting | Research: behavioral changes precede visible symptoms by 4-5 days |
| **Welford's algorithm** over stored history | O(1) memory, numerically stable, no buffer management |
| **Z-score** over raw thresholds | Statistically grounded, adapts to each fish's variance |
| **Multi-indicator consensus** over single-metric alerts | Reduces false positives from noisy individual metrics |
| **Hysteresis state machine** over simple thresholds | Prevents rapid status flickering at boundary scores |
| **EMA smoothing** over raw scores | Removes frame-to-frame noise while staying responsive |
| **4-frame sustained check** over instant alerts | Ensures persistence of concern before alerting |
| **60-second cooldown** over unlimited alerts | Prevents alert fatigue |
| **Body-length speed normalization** | Makes speed comparable across fish of different sizes |
| **Dynamic tank dimensions** | Adapts to any camera resolution instead of hardcoded 640x480 |
| **In-place DOM updates** over innerHTML rebuild | Eliminates visual flickering in dashboard cards |

---

## References

- **Welford's Online Algorithm** (B.P. Welford, 1962) -- Numerically stable single-pass variance computation
- **Z-Score Anomaly Detection** -- Standard statistical method for identifying outliers relative to a distribution
- **Exponential Moving Average** -- Signal processing technique for noise reduction in time-series data
- **Hysteresis Control** -- Engineering control theory for preventing oscillation at decision boundaries
- **SORT Tracking** (Bewley et al., 2016) -- Simple Online and Realtime Tracking with Kalman filter
- **YOLOv8** (Ultralytics, 2023) -- State-of-the-art real-time object detection architecture
- **Hungarian Algorithm** (Kuhn, 1955) -- Optimal assignment problem solution for detection-to-track matching
- **Kalman Filter** (Kalman, 1960) -- Optimal state estimation for linear dynamic systems
