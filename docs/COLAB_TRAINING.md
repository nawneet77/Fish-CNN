# Training on Google Colab

This guide shows you how to train your YOLOv8 model on Google Colab (free GPU!).

## Why Google Colab?

✅ **Free GPU** - T4 GPU for faster training  
✅ **No setup** - Everything runs in browser  
✅ **Fast training** - 30-60 minutes vs 1-3 hours on Mac  
✅ **No local resources** - Doesn't use your computer  

## Quick Start

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click **File → Upload Notebook**
4. Upload `colab_training.ipynb` from this project

OR create a new notebook and copy the cells from `colab_training.ipynb`

### Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### Step 3: Upload Your Dataset

**Option A: Upload from Computer**
1. Click the **folder icon** (📁) on the left sidebar
2. Click the **upload icon** (📤)
3. Upload your entire `fishy-3` folder
4. Wait for upload to complete

**Option B: Download from Roboflow**
- Uncomment the Roboflow download cell in the notebook
- Run it to download fresh from Roboflow

### Step 4: Run Training

1. Run all cells in order (Runtime → Run all)
2. Or run cells one by one:
   - Install dependencies
   - Verify dataset
   - Train model
   - View results
   - Download model

### Step 5: Download Your Model

After training completes:
1. Run the download cell
2. The `best.pt` file will download automatically
3. Save it to your local machine

## Detailed Steps

### 1. Install Dependencies

```python
!pip install ultralytics roboflow -q
```

### 2. Verify Dataset

The notebook will check:
- Dataset folder exists
- data.yaml is present
- Image counts
- Class information

### 3. Train Model

The training cell will:
- Check for GPU
- Load YOLOv8n (nano model)
- Train for 100 epochs
- Save results

**Training time on Colab GPU: 30-60 minutes**

### 4. View Results

- Training curves (mAP, precision, recall)
- Final metrics
- Test predictions

### 5. Download Model

- Download `best.pt` (best model)
- Or download entire results folder as ZIP

## Using Your Trained Model

After downloading `best.pt`:

1. **Save to your project:**
   ```bash
   # Create models directory
   mkdir -p models/training/fishy_v3_colab
   mkdir -p models/training/fishy_v3_colab/weights
   
   # Move downloaded best.pt
   mv ~/Downloads/best.pt models/training/fishy_v3_colab/weights/
   ```

2. **Use in your code:**
   ```python
   from src.health_monitor import FishHealthMonitor
   
   monitor = FishHealthMonitor(
       detector_model_path="models/training/fishy_v3_colab/weights/best.pt",
       detection_confidence=0.5,
       analysis_interval=5
   )
   
   monitor.process_camera(camera_id=0, display=True)
   ```

## Colab Tips

### GPU Availability

- **Free tier**: Limited GPU hours per day
- **Pro tier**: More GPU hours ($10/month)
- If no GPU available, training will use CPU (slower)

### Session Timeout

- Colab sessions timeout after ~90 minutes of inactivity
- Training will continue even if you close the browser
- Check back periodically to see progress

### Saving Work

- Download your model immediately after training
- Colab files are deleted when session ends
- Use Google Drive to save permanently (optional)

### Upload Speed

- Dataset upload may take 5-10 minutes
- Be patient, it's a one-time upload
- Consider using Roboflow download instead

## Troubleshooting

### "GPU not available"

**Solution:**
1. Runtime → Change runtime type → GPU → Save
2. Runtime → Restart runtime
3. Check GPU: `!nvidia-smi`

### "Out of memory"

**Solution:**
- Reduce batch size: `batch=8` or `batch=4`
- Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`

### "Dataset not found"

**Solution:**
1. Check folder name matches (should be `fishy-3`)
2. Upload folder to Colab (not just files)
3. Verify folder structure: `fishy-3/train/images/`, `fishy-3/valid/images/`

### "Session disconnected"

**Solution:**
- Training continues in background
- Reconnect and check progress
- Results are saved automatically

## Comparison: Colab vs Local

| Feature | Google Colab | Local Mac |
|---------|-------------|-----------|
| **GPU** | ✅ Free T4 GPU | ❌ No GPU (or expensive) |
| **Speed** | 30-60 min | 1-3 hours (M1/M2) |
| **Setup** | Browser only | Requires setup |
| **Cost** | Free (limited) | Free |
| **Internet** | Required | Not required |
| **Best for** | Training | Inference/monitoring |

## Next Steps

1. ✅ Train on Colab: Upload notebook and dataset
2. ✅ Download model: Get `best.pt` file
3. ✅ Use locally: Load model in your monitoring system
4. ✅ Test: Run on your aquarium

## Resources

- **Colab Notebook**: `colab_training.ipynb`
- **Training Guide**: `docs/FINE_TUNING_GUIDE.md`
- **Google Colab**: https://colab.research.google.com/

---

**Pro Tip**: Train on Colab, use locally! Best of both worlds. 🚀
