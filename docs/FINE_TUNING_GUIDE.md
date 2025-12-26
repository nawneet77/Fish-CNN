# Fine-Tuning Guide: Training on Your Fish Tank

This guide walks you through fine-tuning the fish detection model on your specific fish tank to improve accuracy.

## Why Fine-Tune?

The default YOLOv8 model is trained on general objects (COCO dataset), not specifically on fish. Fine-tuning on your tank provides:

- **Higher Accuracy**: Better detection of your specific fish species
- **Fewer False Positives**: Reduced confusion with tank decorations
- **Better Performance**: Optimized for your lighting and water conditions
- **Species-Specific**: Trained on your exact fish types and colors

## Overview of Process

```
1. Collect Data (Videos/Images of your tank)
2. Annotate Fish (Label bounding boxes)
3. Prepare Dataset (Split train/val/test)
4. Train YOLO Model (Fine-tune on your data)
5. Evaluate Performance
6. Use Custom Model (Replace default model)
7. Calibrate Health Thresholds (Adjust for your fish)
```

## Step 1: Collect Data

### Recording Your Fish Tank

**Recommended Setup**:
- **Duration**: Record 5-10 minutes of footage
- **Conditions**: Capture various times of day, feeding times, resting periods
- **Angles**: Multiple camera positions if possible
- **Quality**: 1080p or 720p, 30 FPS
- **Lighting**: Include normal and low-light conditions

**Using Mac Camera**:

```bash
# Quick recording script
python examples/collect_training_data.py
```

Or manually with QuickTime:
1. Open QuickTime Player
2. File → New Movie Recording
3. Point camera at fish tank
4. Record for 5-10 minutes
5. Save to `data/raw/my_tank_video.mov`

**Tips**:
- Ensure fish are clearly visible
- Capture all fish in your tank
- Include different behaviors (swimming, eating, resting)
- Avoid glare and reflections

### Extracting Frames

```bash
# Extract frames from video (1 frame per second)
python scripts/extract_frames.py \
    --video data/raw/my_tank_video.mov \
    --output data/raw/frames \
    --fps 1
```

**How many images do you need?**
- **Minimum**: 100-200 annotated images
- **Good**: 500-1000 images
- **Excellent**: 2000+ images

## Step 2: Annotate Your Fish

You need to draw bounding boxes around each fish in your images.

### Option A: Using Roboflow (Recommended - Free & Easy)

1. **Create Account**: Go to [roboflow.com](https://roboflow.com) (free tier available)

2. **Create New Project**:
   - Project Name: "My Fish Tank"
   - Project Type: Object Detection
   - Annotation Group: Fish

3. **Upload Images**:
   - Upload frames from `data/raw/frames`
   - Roboflow will auto-organize

4. **Annotate**:
   - Draw boxes around each fish
   - Label all as "fish" (or by species: "goldfish", "guppy", etc.)
   - Use keyboard shortcuts for speed:
     - `B`: Box tool
     - `Enter`: Save annotation
     - `→`: Next image

5. **Export Dataset**:
   - Generate → YOLOv8
   - Download in YOLOv8 format
   - Extract to `data/annotations/my_fish_dataset`

### Option B: Using LabelImg (Free Desktop Tool)

```bash
# Install LabelImg
pip install labelImg

# Run annotation tool
labelImg data/raw/frames data/annotations/classes.txt
```

1. Open image
2. Press `W` to create box
3. Draw around fish
4. Label as "fish"
5. Press `D` for next image
6. Repeat for all images

### Option C: Using CVAT (Advanced)

See [CVAT Documentation](https://opencv.github.io/cvat/docs/) for self-hosted annotation.

## Step 3: Prepare Dataset

### Dataset Structure

Your dataset should look like this:

```
data/annotations/my_fish_dataset/
├── data.yaml           # Dataset configuration
├── train/
│   ├── images/         # Training images
│   └── labels/         # Training annotations (.txt)
├── valid/
│   ├── images/         # Validation images
│   └── labels/         # Validation annotations
└── test/
    ├── images/         # Test images (optional)
    └── labels/         # Test annotations
```

### Create data.yaml

Create `data/annotations/my_fish_dataset/data.yaml`:

```yaml
# Training/Validation/Test sets
train: train/images
val: valid/images
test: test/images  # optional

# Number of classes
nc: 1

# Class names
names: ['fish']

# Or if you have multiple species:
# nc: 3
# names: ['goldfish', 'guppy', 'tetra']
```

### Split Dataset (if not done by Roboflow)

```bash
python scripts/split_dataset.py \
    --input data/raw/frames \
    --output data/annotations/my_fish_dataset \
    --split 0.7 0.2 0.1  # 70% train, 20% val, 10% test
```

## Step 4: Train YOLO Model

### Training Script

Create `scripts/train_fish_detector.py`:

```python
from ultralytics import YOLO

# Load pretrained model (transfer learning)
model = YOLO('yolov8n.pt')  # Start from pretrained weights

# Train on your dataset
results = model.train(
    data='data/annotations/my_fish_dataset/data.yaml',
    epochs=100,           # More epochs for better accuracy
    imgsz=640,           # Image size
    batch=16,            # Adjust based on your Mac's RAM
    device='mps',        # Use 'mps' for Apple Silicon, 'cpu' for Intel
    patience=20,         # Early stopping patience
    save=True,
    project='models/training',
    name='my_fish_tank_v1',

    # Augmentation (helps with small datasets)
    augment=True,
    hsv_h=0.015,        # Hue augmentation
    hsv_s=0.7,          # Saturation augmentation
    hsv_v=0.4,          # Value augmentation
    degrees=10,         # Rotation
    translate=0.1,      # Translation
    scale=0.5,          # Scaling
    flipud=0.0,         # Vertical flip (usually 0 for fish)
    fliplr=0.5,         # Horizontal flip
)

# Print results
print(f"Training complete!")
print(f"Best model saved to: {results.save_dir}")
```

### Run Training

```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python scripts/train_fish_detector.py
```

**Training Time**:
- **Apple Silicon (M1/M2/M3)**: 1-3 hours for 100 epochs
- **Intel Mac**: 4-8 hours for 100 epochs

**Monitor Training**:
- Training progress will be printed to console
- Results saved to `models/training/my_fish_tank_v1/`
- TensorBoard logs available for visualization

### View Training Results

```bash
# View training metrics
tensorboard --logdir models/training/my_fish_tank_v1
```

Open browser to `http://localhost:6006`

## Step 5: Evaluate Performance

### Validation Metrics

After training, check `models/training/my_fish_tank_v1/results.csv`:

- **mAP50**: Should be > 0.8 (80%) for good performance
- **mAP50-95**: Should be > 0.6 (60%)
- **Precision**: How many detections are correct
- **Recall**: How many fish are detected

### Test on Video

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('models/training/my_fish_tank_v1/weights/best.pt')

# Test on a video
results = model.predict(
    source='data/raw/test_video.mov',
    save=True,
    conf=0.5
)

print(f"Results saved to: {results[0].save_dir}")
```

### Visual Inspection

```bash
# Test detection on new images
python scripts/test_model.py \
    --model models/training/my_fish_tank_v1/weights/best.pt \
    --source data/raw/test_frames \
    --output data/processed/predictions
```

Check predictions in `data/processed/predictions/` - are all fish detected?

## Step 6: Use Your Custom Model

### Update Configuration

Edit `configs/my_tank_config.yaml`:

```yaml
detection:
  model_path: "models/training/my_fish_tank_v1/weights/best.pt"  # Your model!
  confidence_threshold: 0.5
  device: "mps"  # or "cpu" for Intel Mac
```

### Use in Monitoring System

```python
from src.health_monitor import FishHealthMonitor

# Initialize with your custom model
monitor = FishHealthMonitor(
    detector_model_path="models/training/my_fish_tank_v1/weights/best.pt",
    detection_confidence=0.5,
    device="mps"  # Use MPS for Apple Silicon
)

# Run monitoring
monitor.process_camera(camera_id=0, display=True)
```

## Step 7: Calibrate Health Thresholds

Now calibrate health baselines for **your** fish:

### Collect Baseline Data

Run monitoring for 1-2 hours on healthy fish:

```python
# Collect baseline health data
python scripts/collect_baselines.py \
    --duration 7200 \  # 2 hours in seconds
    --output data/baselines/healthy_fish.json
```

### Analyze Baselines

```python
import json
import numpy as np

# Load baseline data
with open('data/baselines/healthy_fish.json', 'r') as f:
    baselines = json.load(f)

# Calculate normal ranges
speeds = [b['swimming_speed'] for b in baselines]
print(f"Normal speed range: {np.percentile(speeds, 5):.1f} - {np.percentile(speeds, 95):.1f}")

colors = [b['color_score'] for b in baselines]
print(f"Normal color score: {np.mean(colors):.2f} ± {np.std(colors):.2f}")
```

### Update Thresholds

Edit `configs/my_tank_config.yaml`:

```yaml
behavior:
  normal_speed_range: [2.0, 12.0]  # From your baseline data
  normal_surface_ratio: 0.15
  normal_bottom_ratio: 0.25

visual_health:
  thresholds:
    healthy: 0.75      # Adjust based on your fish
    mild_concern: 0.55
    moderate_concern: 0.35

alerts:
  thresholds:
    critical_health: 0.35    # When to trigger critical alert
    warning_health: 0.55
```

## Common Issues & Solutions

### Issue: Low mAP (<0.5)

**Solutions**:
- Collect more annotated images (aim for 500+)
- Check annotation quality (are boxes accurate?)
- Train for more epochs (try 150-200)
- Use data augmentation
- Ensure good image variety

### Issue: Missing Detections

**Solutions**:
- Lower confidence threshold (try 0.3-0.4)
- Collect more images of hard-to-detect poses
- Improve lighting in tank
- Train on more diverse data

### Issue: False Positives

**Solutions**:
- Raise confidence threshold (try 0.6-0.7)
- Add negative examples (images with no fish)
- Check if decorations look like fish
- More training epochs

### Issue: Slow Training on Mac

**Solutions**:
- Use smaller batch size (8 or 4)
- Use `yolov8n.pt` instead of larger models
- Train on fewer epochs
- Consider using Google Colab (free GPU)

## Advanced: Training on Google Colab (Free GPU)

If your Mac is too slow:

1. Upload dataset to Google Drive
2. Open Google Colab
3. Mount Google Drive
4. Install Ultralytics: `!pip install ultralytics`
5. Run training with GPU acceleration
6. Download trained model back to Mac

See `docs/COLAB_TRAINING.md` for detailed instructions.

## Iterative Improvement

1. **Start Small**: Train on 100-200 images first
2. **Test**: Evaluate on your tank
3. **Identify Gaps**: What's being missed?
4. **Collect More Data**: Focus on missed cases
5. **Retrain**: Include new data
6. **Repeat**: Until performance is satisfactory

## Species-Specific Models

If you have multiple fish species:

```yaml
# In data.yaml
nc: 3
names: ['goldfish', 'guppy', 'betta']
```

Then you can track health per species with different thresholds!

## Maintenance

### Periodic Retraining

Retrain every few months if:
- Tank setup changes (new decorations)
- Lighting changes (seasonal)
- New fish added
- Performance degrades

### Version Control

Keep track of model versions:
```
models/
├── my_tank_v1/  # Initial model
├── my_tank_v2/  # Improved with more data
└── my_tank_v3/  # Current best
```

## Getting Help

- **Low accuracy?** Share your data.yaml and results.csv in GitHub Issues
- **Training errors?** Include full error message and system info
- **Questions?** Join our Discord community

## Next Steps

1. ✅ Collect 100-200 images of your tank
2. ✅ Annotate fish using Roboflow
3. ✅ Train initial model
4. ✅ Test on your tank
5. ✅ Iterate and improve

Happy training! Your custom model will significantly outperform the generic one for your specific tank. 🐠
