# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time fish health monitoring system for aquariums using YOLOv8 object detection, Kalman filter tracking, and heuristic-based health assessment. Python 3.8+, runs on CPU/CUDA/Apple Silicon (MPS).

## Common Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run live camera monitoring (primary entry point)
python examples/mac_camera_monitoring.py

# Process a video file
python examples/basic_video_monitoring.py

# Test custom-trained model
python examples/test_custom_model.py

# Collect training data from camera
python scripts/collect_training_data.py --duration 300

# Train custom fish detector
python scripts/train_fish_detector.py --data path/to/data.yaml

# Install as package (editable)
pip install -e .
```

No test suite exists yet (`tests/` directory is empty).

## Architecture

The pipeline processes each video frame through five stages in sequence:

1. **Detection** (`src/detection/fish_detector.py`) — YOLOv8 via `ultralytics` library detects fish bounding boxes
2. **Tracking** (`src/tracking/fish_tracker.py`) — SORT algorithm with Kalman filtering + appearance-based re-identification tracks fish across frames
3. **Visual Health** (`src/health_assessment/visual_analyzer.py`) — Heuristic analysis of color, texture, body shape, fin condition, eye clarity
4. **Behavior Analysis** (`src/behavior_analysis/behavior_analyzer.py`) — Monitors swimming speed, patterns, activity level, isolation, depth preference
5. **Alerts** (`src/alerts/alert_system.py`) — SQLite-backed alert system with severity levels (INFO/WARNING/CRITICAL) and cooldown deduplication

`src/health_monitor.py` is the **integration module** (`FishHealthMonitor` class) that orchestrates all five stages. It owns the frame processing loop, baseline collection (first 900 frames), and visualization overlay.

Overall health score = 50% visual + 50% behavioral. Health thresholds: Healthy ≥0.7, Mild Concern ≥0.5, Moderate ≥0.3, Severe <0.3.

## Key Design Decisions

- **Analysis interval**: Health analysis runs every N frames (default 5) for performance; cached results are reused between analyses
- **Baseline mode**: System collects 900 frames (~30s) of baseline data before generating alerts to prevent false alarms on startup
- **Alert cooldown**: Same alert type won't repeat for a fish within 60 seconds
- **Tracker persistence**: `max_age=150` keeps tracks alive for ~5 seconds without detection; appearance-based re-ID reconnects lost tracks
- **Device auto-detection**: CUDA → MPS → CPU fallback chain

## Configuration

`configs/default_config.yaml` — primary config for all thresholds and settings. `configs/mac_config.yaml` — macOS-specific overrides.

## Models

- Generic: `yolov8n.pt` (auto-downloaded by ultralytics on first use)
- Custom trained: `models/custom_trained/best.pt` (89.96% mAP50, trained on fish tank footage)

## Dependencies

Core: `torch`, `ultralytics` (YOLOv8), `opencv-python`, `filterpy` (Kalman filter), `scipy`, `scikit-learn`, `loguru`

## File Conventions

- All source modules are under `src/` with `__init__.py` files exporting key classes
- Each module defines dataclasses for its output (e.g., `Detection`, `Track`, `VisualHealthMetrics`, `BehaviorMetrics`, `Alert`)
- `examples/` contains runnable scripts; `scripts/` contains training/data collection utilities
