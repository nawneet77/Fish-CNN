"""
Train Custom Fish Detection Model
Fine-tunes YOLOv8 on your fish tank dataset
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import platform


def detect_device():
    """Detect best device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def train_model(
    data_yaml,
    model_size='n',
    epochs=100,
    batch_size=16,
    imgsz=640,
    device=None,
    project='models/training',
    name='my_fish_model',
    pretrained=True
):
    """
    Train fish detection model

    Args:
        data_yaml: Path to dataset YAML file
        model_size: YOLO model size (n/s/m/l/x)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if out of memory)
        imgsz: Input image size
        device: Device to use (auto-detected if None)
        project: Project directory
        name: Experiment name
        pretrained: Use pretrained weights
    """
    # Validate data.yaml exists
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset file not found: {data_yaml}")

    # Detect device
    if device is None:
        device = detect_device()

    print("="*70)
    print("FISH DETECTION MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {data_yaml}")
    print(f"  Model: YOLOv8{model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {imgsz}")
    print(f"  Device: {device.upper()}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Output: {project}/{name}")

    # Platform info
    print(f"\nSystem Info:")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    if device == 'mps':
        print(f"  Using Apple Silicon GPU acceleration")
    elif device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print(f"  Using CPU (training will be slow)")
        print(f"  Consider using Google Colab for GPU training")

    # Load model
    model_path = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)

    # Adjust batch size for device
    if device == 'cpu':
        batch_size = min(batch_size, 4)
        print(f"  Reduced batch size to {batch_size} for CPU")
    elif device == 'mps':
        batch_size = min(batch_size, 8)
        print(f"  Reduced batch size to {batch_size} for MPS")

    # Training parameters
    print(f"\nStarting training...")
    print(f"  This may take 1-3 hours on Apple Silicon")
    print(f"  This may take 4-8 hours on Intel Mac")
    print(f"  Progress will be displayed below\n")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=20,  # Early stopping
            save=True,
            project=project,
            name=name,
            exist_ok=True,

            # Data augmentation
            augment=True,
            hsv_h=0.015,      # Hue
            hsv_s=0.7,        # Saturation
            hsv_v=0.4,        # Value
            degrees=10,       # Rotation
            translate=0.1,    # Translation
            scale=0.5,        # Scaling
            flipud=0.0,       # No vertical flip (fish don't swim upside down)
            fliplr=0.5,       # Horizontal flip

            # Performance
            workers=4,
            verbose=True,
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)

        save_dir = Path(project) / name
        print(f"\nResults saved to: {save_dir}")
        print(f"\nBest model: {save_dir}/weights/best.pt")
        print(f"Last model: {save_dir}/weights/last.pt")

        # Print final metrics
        print(f"\nFinal Metrics:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f}")
            print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.3f}")

        print(f"\nNext steps:")
        print(f"  1. Review training results in: {save_dir}")
        print(f"  2. Test model: python scripts/test_model.py --model {save_dir}/weights/best.pt")
        print(f"  3. Use in monitoring: Update model_path in config to {save_dir}/weights/best.pt")

        return save_dir

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return None

    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train custom fish detection model"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file (e.g., data/annotations/my_fish_dataset/data.yaml)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge). Nano recommended for Mac.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16, will auto-adjust for device)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=None,
        help='Device to use (auto-detected if not specified)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='my_fish_model',
        help='Experiment name (default: my_fish_model)'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Train from scratch (not recommended)'
    )

    args = parser.parse_args()

    # Train model
    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        pretrained=not args.no_pretrained
    )


if __name__ == "__main__":
    main()
