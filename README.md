# Fish Health Monitoring System

A real-time, AI powered fish health monitoring system for aquarium environments using computer vision and deep learning.

## Overview

This system automatically detects multiple fish in live video streams and assesses their health condition by analyzing both visual symptoms and behavioral patterns over time. It provides early detection of health issues before they become severe, enabling timely intervention.

## Features

- **Real-time Multi-Fish Detection**: Uses YOLOv8 for accurate detection and localization of multiple fish
- **Visual Health Assessment**: Analyzes color, texture, body condition, fin integrity, and eye clarity
- **Behavioral Analysis**: Monitors swimming patterns, activity levels, and social behavior
- **Temporal Tracking**: Tracks individual fish across frames using Kalman filtering
- **Alert System**: Generates alerts with severity levels and actionable recommendations
- **Live Monitoring**: Supports both video files and live camera feeds
- **Visualization**: Real-time visualization with health status overlays
- **Mac Support**: Full support for macOS including Apple Silicon (M1/M2/M3) with GPU acceleration

## Quick Start for Mac Users 🍎

Perfect for monitoring your home fish tank! Here's the fastest way to get started on macOS:

### 1. Install (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/Fish-CNN.git
cd Fish-CNN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Live Monitoring

```bash
# Start monitoring from Mac camera
python examples/mac_camera_monitoring.py
```

**First time**: macOS will ask for camera permissions. Grant access and restart the script.

### 3. Fine-Tune on Your Tank (Later)

After initial testing, train a custom model on your specific fish:

```bash
# Collect training data (5-10 minutes of footage)
python scripts/collect_training_data.py --duration 300

# Annotate fish using Roboflow (free): roboflow.com
# Then train custom model
python scripts/train_fish_detector.py --data path/to/your/data.yaml
```

📖 **Detailed guides**: See [Mac Installation Guide](docs/MAC_INSTALLATION.md) and [Fine-Tuning Guide](docs/FINE_TUNING_GUIDE.md)

## Using Custom Trained Model 🎯

**Good news**: A custom fish detection model is included and ready to use! This model is trained specifically on fish (not people/cars like generic YOLO models).

### Model Performance
- **mAP50**: 89.96% (excellent detection accuracy)
- **Precision**: 95.30% (very few false positives)
- **Recall**: 84.85% (catches most fish in frame)
- **Location**: `models/custom_trained/best.pt`

### Quick Test (30 seconds)
```bash
# Test the custom model on your Mac camera
python examples/test_custom_model.py
```

This will show you:
- Detection accuracy on your specific setup
- Real-time FPS performance
- Whether adjustments are needed

### Full Monitoring with Custom Model
```bash
# The mac_camera_monitoring.py script uses the custom model by default
python examples/mac_camera_monitoring.py
```

**What makes this better than generic models?**
- ✅ Trained on real fish tank footage
- ✅ Won't detect your face/hands as fish
- ✅ 3x better accuracy (89.96% vs 30-50%)
- ✅ Optimized for real-time Mac performance

### Calibration Tips

**If you get too many detections** (false positives):
- Increase confidence in `examples/mac_camera_monitoring.py` line 156: `detection_confidence=0.6` or `0.7`

**If you miss some fish** (low recall):
- Decrease confidence: `detection_confidence=0.3` or `0.4`
- Improve tank lighting
- Adjust camera angle

📖 **Full details**: See [models/custom_trained/MODEL_INFO.md](models/custom_trained/MODEL_INFO.md)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Video Input                              │
│              (Camera / Video File)                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Fish Detection (YOLOv8)                         │
│           Detect and localize all fish                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Multi-Fish Tracking (SORT Algorithm)                 │
│      Track individual fish across frames                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│ Visual Health    │          │ Behavioral       │
│ Assessment       │          │ Analysis         │
│ - Color          │          │ - Speed          │
│ - Texture        │          │ - Patterns       │
│ - Body shape     │          │ - Isolation      │
│ - Fin condition  │          │ - Position       │
│ - Eye clarity    │          │ - Erratic moves  │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
        ┌────────────────────────┐
        │  Health Score          │
        │  Calculation           │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │  Alert Generation      │
        │  & Notifications       │
        └────────────────────────┘
```

## Installation

> **Mac Users**: See the dedicated [Mac Installation Guide](docs/MAC_INSTALLATION.md) for macOS-specific instructions, including Apple Silicon (M1/M2/M3) GPU acceleration setup.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Fish-CNN.git
cd Fish-CNN
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO model (automatic on first run):
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

## Quick Start

### Process a Video File

```python
from src.health_monitor import FishHealthMonitor

# Initialize monitor
monitor = FishHealthMonitor(
    detector_model_path="yolov8n.pt",
    detection_confidence=0.5
)

# Process video
reports = monitor.process_video(
    video_path="path/to/aquarium_video.mp4",
    output_path="path/to/output_video.mp4",
    display=True
)

# Get health summary
summary = monitor.get_health_summary()
print(summary)
```

### Live Camera Monitoring

```python
from src.health_monitor import FishHealthMonitor

# Initialize monitor
monitor = FishHealthMonitor()

# Start live monitoring
monitor.process_camera(camera_id=0, display=True)
```

### Using Individual Components

```python
from src.detection.fish_detector import FishDetector
from src.tracking.fish_tracker import FishTracker
from src.health_assessment.visual_analyzer import VisualHealthAnalyzer

# Initialize components
detector = FishDetector(model_path="yolov8n.pt")
tracker = FishTracker()
analyzer = VisualHealthAnalyzer()

# Process frame
detections = detector.detect(frame)
tracks = tracker.update(detections)

for track in tracks:
    bbox = track.get_current_bbox()
    health = analyzer.analyze(frame, bbox)
    print(f"Fish {track.track_id}: Health Score = {health.overall_health_score:.2f}")
```

## Configuration

Edit `configs/default_config.yaml` to customize the system:

```yaml
detection:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5

visual_health:
  thresholds:
    healthy: 0.7
    mild_concern: 0.5

behavior:
  normal_speed_range: [1.0, 15.0]
  history_length: 100

alerts:
  thresholds:
    critical_health: 0.3
    warning_health: 0.5
```

## Health Metrics

### Visual Health Indicators

- **Color Score**: Measures color vibrancy and uniformity
- **Texture Score**: Detects skin lesions, fungus, or irregularities
- **Body Condition**: Assesses body shape and symmetry
- **Fin Condition**: Evaluates fin integrity and spread
- **Eye Clarity**: Checks for cloudy or sunken eyes

### Behavioral Indicators

- **Swimming Speed**: Monitors for lethargy or hyperactivity
- **Swimming Pattern**: Detects irregular or erratic movements
- **Activity Level**: Compares to normal activity baselines
- **Isolation Score**: Identifies social withdrawal
- **Depth Preference**: Tracks excessive surface or bottom time

### Health Status Levels

- **Healthy** (≥0.7): Fish appears normal and healthy
- **Mild Concern** (0.5-0.7): Minor issues detected, monitor closely
- **Moderate Concern** (0.3-0.5): Significant issues, intervention recommended
- **Severe Concern** (<0.3): Critical condition, immediate action required

## Alert System

The system generates three levels of alerts:

- **INFO**: General information, new fish detected
- **WARNING**: Health concerns detected, monitor closely
- **CRITICAL**: Severe health issues, immediate action needed

Each alert includes:
- Timestamp and affected fish ID
- Health metrics and scores
- Identified symptoms
- Actionable recommendations

## Project Structure

```
Fish-CNN/
├── src/
│   ├── detection/          # Fish detection using YOLO
│   │   └── fish_detector.py
│   ├── tracking/           # Multi-object tracking
│   │   └── fish_tracker.py
│   ├── health_assessment/  # Visual health analysis
│   │   └── visual_analyzer.py
│   ├── behavior_analysis/  # Behavioral pattern analysis
│   │   └── behavior_analyzer.py
│   ├── alerts/            # Alert generation and management
│   │   └── alert_system.py
│   ├── utils/             # Utilities and helpers
│   │   ├── config_loader.py
│   │   └── logger.py
│   └── health_monitor.py  # Main integration module
├── configs/
│   └── default_config.yaml
├── models/                # Model weights (downloaded)
├── data/                  # Data storage
│   ├── raw/              # Raw videos
│   ├── processed/        # Processed videos
│   └── annotations/      # Annotations for training
├── examples/             # Example scripts
├── tests/                # Unit tests
├── deployment/           # Deployment configurations
├── requirements.txt
└── README.md
```

## Performance

On a typical setup:
- **Detection**: 30-60 FPS (GPU), 5-10 FPS (CPU)
- **Full Pipeline**: 20-30 FPS (GPU), 3-5 FPS (CPU)
- **Memory Usage**: ~2-4 GB
- **Model Size**: ~6 MB (YOLOv8n) to ~50 MB (YOLOv8x)

## Use Cases

1. **Home Aquariums**: Monitor pet fish health 24/7
2. **Commercial Aquaculture**: Early disease detection in fish farms
3. **Research**: Automated behavioral studies
4. **Pet Stores**: Monitor fish health in display tanks
5. **Public Aquariums**: Large-scale monitoring systems

## Limitations

- Requires clear water for accurate detection
- Performance depends on video quality and lighting
- Visual analysis is based on heuristics, not medical diagnosis
- May need calibration for specific fish species
- Not a replacement for professional veterinary care

## Future Enhancements

- [ ] Species-specific health models
- [ ] Deep learning-based health classification
- [ ] Water quality sensor integration
- [ ] Mobile app for remote monitoring
- [ ] Cloud-based analytics dashboard
- [ ] Multi-camera support
- [ ] Feeding behavior analysis
- [ ] Growth tracking over time

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{fish_health_monitor,
  title = {Fish Health Monitoring System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/Fish-CNN}
}
```

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team
- FilterPy for Kalman filtering

## Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Wiki](https://github.com/yourusername/Fish-CNN/wiki)

## Disclaimer

This system is designed for monitoring and early detection purposes. It is not a substitute for professional veterinary diagnosis or treatment. Always consult with qualified aquatic veterinarians for health concerns.
