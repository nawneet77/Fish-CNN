# Troubleshooting Common Issues

## Issue: App Crashes on Startup with "Critical Behavioral Issue" Alert

### Symptoms
- App opens camera
- Shows "Critical Behavioral Issue Detected - Fish #0"
- Immediately crashes or exits

### Cause
This was a bug where the system tried to analyze fish health before gathering enough tracking data (fixed in latest version).

### Solution

**Option 1: Update to Latest Version (Recommended)**
```bash
cd Fish-CNN
git pull origin claude/fish-health-monitoring-system-MKiQ8
```

**Option 2: Manual Fix (if you can't update)**
Edit `src/health_monitor.py` around line 173 and add this check:
```python
# Only generate alerts for well-tracked fish
MIN_FRAMES_FOR_ALERTS = 30
if track.hits >= MIN_FRAMES_FOR_ALERTS:
    # ... alert generation code
```

### Why This Happens
- The system needs 30 frames (~1 second at 30 FPS) to build reliable behavioral baseline
- On startup, fish are just detected and don't have enough history
- Default behavior score was set too low (0.5), triggering false alarms
- **Fixed**: Now requires minimum tracking time before generating alerts

---

## Issue: Camera Permission Denied (macOS)

### Symptoms
- "Camera Access Denied" message
- Black screen or no video feed

### Solution
```bash
# 1. Open System Settings → Privacy & Security → Camera
# 2. Enable camera for Terminal (or your IDE)
# 3. Restart Terminal/IDE
# 4. Run the app again
```

Or reset camera permissions:
```bash
tccutil reset Camera
```

Then grant permissions when prompted.

---

## Issue: Slow Performance on Mac

### Symptoms
- Very low FPS (< 5 FPS)
- Laggy video display
- High CPU usage

### Solutions

**1. Check if using GPU acceleration (Apple Silicon only)**
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

If False, reinstall PyTorch:
```bash
pip install --upgrade torch torchvision torchaudio
```

**2. Use smaller YOLO model**
Edit your config or code to use `yolov8n.pt` (nano) instead of larger models.

**3. Reduce resolution**
Lower your camera resolution in System Settings.

**4. Close other apps**
Video processing is intensive - close unused applications.

---

## Issue: No Fish Detected

### Symptoms
- Camera works but shows "Fish Tracked: 0"
- Bounding boxes don't appear around fish

### Causes & Solutions

**1. Fish too small in frame**
- Move camera closer to tank
- Ensure fish take up at least 5-10% of frame

**2. Low confidence threshold**
```python
monitor = FishHealthMonitor(
    detection_confidence=0.3  # Lower = more sensitive
)
```

**3. Poor lighting**
- Improve tank lighting
- Avoid backlighting
- Reduce glare and reflections

**4. Water clarity**
- Clean tank glass
- Improve water clarity

**5. Generic model not trained on fish**
- Fine-tune on your specific fish tank (see `docs/FINE_TUNING_GUIDE.md`)

---

## Issue: Too Many False Positives

### Symptoms
- Detects decorations, plants, or reflections as fish
- "Ghost" detections

### Solutions

**1. Increase confidence threshold**
```python
monitor = FishHealthMonitor(
    detection_confidence=0.7  # Higher = stricter
)
```

**2. Fine-tune model on your tank**
This teaches the model what is/isn't a fish in your specific setup.

---

## Issue: Installation Errors

### "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "ERROR: Failed building wheel for..."
```bash
# Update pip
pip install --upgrade pip setuptools wheel

# Try again
pip install -r requirements.txt
```

---

## Issue: YOLO Model Download Fails

### Symptoms
- "Failed to download yolov8n.pt"
- Network timeout errors

### Solutions

**1. Manual download**
```bash
# Download from browser:
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Place in project root or models/ folder
```

**2. Use different model**
```python
# Try smaller model
monitor = FishHealthMonitor(model_path="yolov8n.pt")
```

---

## Issue: Alerts for Healthy Fish

### Symptoms
- Getting warnings/critical alerts for obviously healthy fish
- Too sensitive

### Solution

**1. Wait for baseline to establish**
- System needs 30+ frames to establish accurate baseline
- First few seconds may show inaccurate readings

**2. Adjust thresholds in config**
Edit `configs/default_config.yaml`:
```yaml
alerts:
  thresholds:
    critical_health: 0.2    # Lower = less sensitive
    warning_health: 0.4     # Lower = less sensitive
```

**3. Calibrate for your fish**
See "Step 7: Calibrate Health Thresholds" in `docs/FINE_TUNING_GUIDE.md`

---

## Issue: Out of Memory (Apple Silicon)

### Symptoms
- "RuntimeError: MPS backend out of memory"
- App crashes during processing

### Solutions

**1. Reduce batch size**
```python
# In training scripts
batch_size = 4  # Instead of 16
```

**2. Use smaller model**
```python
model = YOLO('yolov8n.pt')  # Smallest
```

**3. Process at lower resolution**
```python
# In detection settings
imgsz = 416  # Instead of 640
```

---

## Issue: Training Very Slow

### Symptoms
- Training taking > 8 hours
- ETA shows days

### Solutions

**1. Use Google Colab (Free GPU)**
- Upload dataset to Google Drive
- Run training in Colab with free GPU
- Download trained model

**2. Reduce epochs**
```bash
python scripts/train_fish_detector.py --epochs 50  # Instead of 100
```

**3. Use smaller model**
```bash
python scripts/train_fish_detector.py --model n  # Nano size
```

**4. Reduce dataset size**
- Start with 100-200 images for quick iteration
- Add more data later for refinement

---

## Issue: Git Pull Conflicts

### Symptoms
- "error: Your local changes would be overwritten"
- Merge conflicts

### Solution
```bash
# Save your changes
git stash

# Pull updates
git pull origin claude/fish-health-monitoring-system-MKiQ8

# Reapply changes
git stash pop
```

See `docs/UPDATING.md` for more details.

---

## Issue: Database Locked Error

### Symptoms
- "database is locked" error
- Can't save alerts

### Solution
```bash
# Delete alert database (will lose alert history)
rm data/alerts.db

# Or specify different database path
monitor = FishHealthMonitor(alert_db_path="alerts_new.db")
```

---

## Getting Help

If none of these solutions work:

1. **Check version**
   ```bash
   git log --oneline -1
   ```

2. **Update to latest**
   ```bash
   git pull origin claude/fish-health-monitoring-system-MKiQ8
   ```

3. **Collect info**
   - Mac model (M1/M2/M3 or Intel)
   - macOS version
   - Python version: `python --version`
   - Error message (full text)
   - Screenshot if applicable

4. **Create GitHub Issue**
   - Include all info from step 3
   - Describe what you were trying to do
   - Include steps to reproduce

5. **Common Quick Fixes**
   ```bash
   # Restart everything
   deactivate
   source venv/bin/activate
   pip install --upgrade -r requirements.txt
   ```
