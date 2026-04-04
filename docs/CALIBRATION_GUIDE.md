# Calibration Guide — Fish Health Monitoring System

This document explains how to calibrate the fish detection and tracking parameters for your specific video/camera setup using `scripts/calibrate_video.py`.

## Why Calibrate?

The default parameters work for general use, but every tank is different — lighting, fish size, water clarity, number of fish, camera angle. Calibrating finds the best settings for **your** specific setup so the system detects all your fish reliably without false positives.

## How to Run

```bash
# Full calibration: automated sweep + interactive tuning
python scripts/calibrate_video.py --video YOUR_VIDEO.mp4 --model models/custom_trained/best.pt

# Just the automated sweep (no GUI)
python scripts/calibrate_video.py --video YOUR_VIDEO.mp4 --mode sweep

# Just the interactive viewer with sliders
python scripts/calibrate_video.py --video YOUR_VIDEO.mp4 --mode interactive

# Specify output config path
python scripts/calibrate_video.py --video YOUR_VIDEO.mp4 --output-config configs/my_tank_config.yaml
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | `VID-20260327-WA0001.mp4` | Path to the video file |
| `--model` | `models/custom_trained/best.pt` | Path to YOLO model weights |
| `--mode` | `both` | `sweep`, `interactive`, or `both` |
| `--sample-frames` | `20` | Number of frames to sample in sweep mode |
| `--output-config` | `configs/calibrated_config.yaml` | Where to save the config |
| `--device` | auto | `cuda`, `mps`, `cpu`, or auto-detect |

## Interactive Mode Controls

| Key | Action |
|-----|--------|
| **Sliders** | Drag to adjust parameters — changes apply in real time |
| **SPACE** | Pause / resume video playback |
| **LEFT/RIGHT arrow** | Seek backward / forward 30 frames |
| **R** | Reset tracker (clears all track IDs and history) |
| **S** | Save a screenshot of the current frame |
| **Q** | Quit and save your chosen configuration |

## Parameter Reference

### Detection Parameters

These control how the YOLO model decides what counts as a "fish" in each frame.

#### Confidence Threshold

- **What it does:** Sets the minimum certainty the model needs before it draws a box around something and calls it a fish. The model assigns a score from 0.0 (no idea) to 1.0 (absolutely sure) to every potential detection.
- **Range:** 0.05 to 0.95
- **Default:** 0.50
- **How to adjust:**
  - **Too many false detections** (boxes on rocks, decorations, reflections) — **increase** to 0.55, 0.60, or higher. This makes the model pickier.
  - **Missing real fish** (fish are visible but no box appears) — **decrease** to 0.40, 0.30, or even 0.25. This makes the model less picky.
- **Trade-off:** Lower confidence catches more fish but also more false positives. Higher confidence is more precise but may miss fish that are partially hidden, far away, or in poor lighting.

#### NMS IoU Threshold (Non-Maximum Suppression)

- **What it does:** When the model finds overlapping boxes on the same area, this decides whether to keep both or merge them into one. "IoU" stands for Intersection over Union — it measures how much two boxes overlap (0.0 = no overlap, 1.0 = identical boxes).
- **Range:** 0.05 to 0.90
- **Default:** 0.45
- **How to adjust:**
  - **Seeing duplicate boxes on the same fish** — **decrease** to 0.30 or 0.35. This removes overlapping boxes more aggressively.
  - **Two fish swimming close together but only one gets a box** — **increase** to 0.55 or 0.60. This lets overlapping boxes survive, so nearby fish each get their own detection.
- **Trade-off:** Lower values give cleaner single boxes but might merge two close fish into one detection. Higher values keep more boxes but might show duplicates.

### Tracking Parameters

These control how the system follows individual fish across frames and maintains their identity (ID numbers).

#### Max Age (frames)

- **What it does:** How many frames the system remembers a fish after it disappears from view. If a fish goes behind a plant or the model misses it for a few frames, the tracker keeps its ID alive for this many frames. If the fish reappears within this window, it gets the same ID back.
- **Range:** 10 to 500
- **Default:** 150 (about 5 seconds at 30 FPS)
- **How to adjust:**
  - **Fish keep getting new IDs after brief occlusions** (e.g., Fish #1 disappears behind a rock and comes back as Fish #5) — **increase** to 200-300.
  - **Ghost tracks linger too long** (old IDs stay on screen after a fish has clearly left) — **decrease** to 60-100.
- **Calculation:** Max Age / Video FPS = seconds of memory. At 30 FPS: 60 frames = 2 sec, 150 frames = 5 sec, 300 frames = 10 sec.

#### Min Hits

- **What it does:** How many consecutive detections are needed before the system considers something a "confirmed" fish and starts displaying its track. This prevents random one-frame flickers or noise from becoming tracked objects.
- **Range:** 1 to 10
- **Default:** 3
- **How to adjust:**
  - **Seeing brief phantom tracks that appear and vanish** — **increase** to 4-5. The system will wait longer to confirm.
  - **Fish appear in the tank but take too long to get an ID** — **decrease** to 2 or even 1.
- **Rule of thumb:** Keep between 2-5. Values above 5 cause noticeable delay before new fish are tracked.

#### Tracker IoU Threshold

- **What it does:** The minimum spatial overlap required to match a new detection (in the current frame) to an existing track (from previous frames). Each frame, the tracker compares where it predicts each fish should be with where the detector actually found fish, using IoU (overlap score).
- **Range:** 0.05 to 0.50
- **Default:** 0.20
- **How to adjust:**
  - **Fish IDs keep swapping** (Fish #1 and #2 trade identities when they cross paths) — **increase** to 0.25-0.30. Stricter matching means less cross-contamination.
  - **Fast-moving fish lose their ID every few frames** (because the predicted position is too far from the actual position) — **decrease** to 0.10-0.15. More flexible matching tolerates bigger jumps.
- **Trade-off:** Lower values handle fast movement better but may incorrectly merge two fish. Higher values are more precise but struggle with fast or erratic swimmers.

#### Appearance Weight

- **What it does:** Controls the balance between two ways of matching fish to tracks:
  - **Position-based** (IoU): "Is this detection where I expected the fish to be?"
  - **Appearance-based**: "Does this detection look like the same fish (color, pattern)?"
  
  A value of 0.0 means 100% position, 1.0 means 100% appearance. The default 0.4 means 60% position + 40% appearance.
- **Range:** 0.0 to 1.0
- **Default:** 0.40
- **How to adjust:**
  - **Fish look very similar** (same species, same color) — **decrease** to 0.1-0.2. Appearance matching won't help distinguish them, so rely more on position.
  - **Fish look distinct but keep swapping IDs when they cross** — **increase** to 0.5-0.6. Appearance helps tell them apart even when positions overlap.
  - **Lighting changes a lot** (sun/shadow, auto white balance) — **decrease** to 0.1-0.2. Appearance features become unreliable with changing lighting.
- **Note:** Setting this to 0.0 also disables Re-ID (re-identification of lost tracks). Any value above 0.0 enables Re-ID.

## Calibration Results

### Sweep Results (from VID-20260327-WA0001.mp4)

The automated sweep tested 15 detection parameter combinations and 108 tracker combinations.

**Best detection config found:**

| Parameter | Default | Calibrated | Why |
|-----------|---------|------------|-----|
| confidence_threshold | 0.50 | 0.25 | Fish in this video need lower threshold to be detected consistently |
| iou_threshold (NMS) | 0.45 | 0.60 | Fish swim close together, higher value keeps both detections |

**Best tracker config found:**

| Parameter | Default | Calibrated | Why |
|-----------|---------|------------|-----|
| max_age | 150 | 60 | Fish don't occlude for long in this tank, shorter memory reduces ghost tracks |
| min_hits | 3 | 2 | Faster track confirmation since detections are reliable |
| tracker_iou | 0.20 | 0.15 | More flexible matching for the fish movement speed |
| appearance_weight | 0.40 | 0.20 | Fish look similar, rely more on position |

### Using the Calibrated Config

After calibration, the config is saved at `configs/calibrated_config.yaml`. Use it in your code:

```python
from src.health_monitor import FishHealthMonitor

monitor = FishHealthMonitor(
    detector_model_path="models/custom_trained/best.pt",
    detection_confidence=0.25,
    detection_iou_threshold=0.6,
    tracker_max_age=60,
    enable_reid=True,
)

# Process your video
reports = monitor.process_video("VID-20260327-WA0001.mp4", display=True)
```

## Tips

- **Calibrate per tank/camera:** Different setups need different settings. Save separate configs (e.g., `configs/tank1_config.yaml`, `configs/tank2_config.yaml`).
- **Re-calibrate when conditions change:** New lighting, new fish added, camera moved — re-run calibration.
- **Start with sweep, finish with interactive:** The sweep gives a good starting point. Interactive mode lets you fine-tune visually.
- **Watch the detection count:** In the interactive viewer, the top overlay shows how many fish are detected per frame. If this number is stable and matches your actual fish count, you have good settings.
- **Watch the track IDs:** If IDs keep incrementing (Fish #1, #5, #12, #20...) instead of staying stable (Fish #1, #2, #3), your tracker settings need adjustment — usually increase max_age or decrease tracker IoU.
