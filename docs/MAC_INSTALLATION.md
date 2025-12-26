# Installation Guide for macOS

This guide covers installation on macOS, including both Intel and Apple Silicon (M1/M2/M3) Macs.

## System Requirements

- **macOS**: 11.0 (Big Sur) or later recommended
- **Python**: 3.8 - 3.11 (3.11 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **Camera**: Built-in or external USB camera

## Installation Steps

### Step 1: Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python

```bash
# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/yourusername/Fish-CNN.git
cd Fish-CNN
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 5: Install Dependencies

#### For Apple Silicon Macs (M1/M2/M3):

```bash
# Install PyTorch with MPS (Metal Performance Shaders) support
pip install --upgrade pip
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

#### For Intel Macs:

```bash
# Install PyTorch
pip install --upgrade pip
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### Step 6: Grant Camera Permissions

**Important**: macOS requires explicit camera permissions.

1. Go to **System Settings** → **Privacy & Security** → **Camera**
2. Allow Terminal (or your IDE) to access the camera
3. If using iTerm2, PyCharm, or VS Code, grant permission to those apps

Alternatively, you can test camera access:

```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera access:', cap.isOpened()); cap.release()"
```

If this returns `Camera access: False`, check your privacy settings.

### Step 7: Verify Installation

```bash
# Test import
python3 -c "import torch; import cv2; print('PyTorch version:', torch.__version__); print('OpenCV version:', cv2.__version__)"

# Check if MPS is available (Apple Silicon only)
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Running on Mac

### Using Mac Camera

```bash
# Run live monitoring (camera ID 0 is usually built-in camera)
python examples/mac_camera_monitoring.py
```

### Performance on Mac

- **Apple Silicon (M1/M2/M3)**: 15-25 FPS (with MPS acceleration)
- **Intel Mac**: 3-8 FPS (CPU only)

The system automatically detects if you're on Apple Silicon and uses MPS acceleration.

## Mac-Specific Configurations

Edit `configs/mac_config.yaml`:

```yaml
detection:
  device: "mps"  # Use "mps" for Apple Silicon, "cpu" for Intel
  model_path: "yolov8n.pt"  # Smaller model for better Mac performance
  confidence_threshold: 0.5

video:
  camera_id: 0  # Built-in camera
  fps: 30.0
  enable_visualization: true
```

## Troubleshooting

### Camera Not Detected

```bash
# List available cameras
python3 -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"
```

### Slow Performance

1. **Use smaller YOLO model**: Change to `yolov8n.pt` (nano) instead of larger models
2. **Reduce resolution**: Lower camera resolution in settings
3. **For Apple Silicon**: Ensure MPS is being used:
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

### Permission Denied

```bash
# Reset camera permissions (macOS 13+)
tccutil reset Camera
```

Then restart Terminal and grant permissions again.

## Apple Silicon Optimization

For best performance on M1/M2/M3 Macs:

```python
# The system automatically detects and uses MPS
# But you can explicitly set it:
monitor = FishHealthMonitor(
    device="mps"  # Metal Performance Shaders
)
```

## Updating

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove project (from outside the project directory)
rm -rf Fish-CNN
```

## Next Steps

1. **Test with Mac Camera**: Run `python examples/mac_camera_monitoring.py`
2. **Collect Your Fish Tank Data**: Record videos of your fish tank
3. **Fine-tune Model**: Follow `docs/FINE_TUNING_GUIDE.md` to train on your specific fish

## Additional Notes

- **Battery Life**: Running on Mac laptops will drain battery quickly. Use power adapter for extended monitoring.
- **Fan Noise**: Intensive processing may increase fan speed, especially on Intel Macs.
- **External Camera**: USB cameras work great and often provide better angles for fish tanks.

## Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/yourusername/Fish-CNN/issues)
3. Create a new issue with:
   - Mac model and chip (M1/M2/M3 or Intel)
   - macOS version
   - Error messages
   - Output of `python3 -c "import torch; print(torch.__version__)"`
