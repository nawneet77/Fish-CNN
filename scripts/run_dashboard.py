"""
Launch the Fish Health Dashboard web server.

Usage:
    python scripts/run_dashboard.py --video VID-20260327-WA0001.mp4
    python scripts/run_dashboard.py --camera 0
    python scripts/run_dashboard.py --video VID.mp4 --port 8000 --confidence 0.45
"""

import argparse
import asyncio
import webbrowser
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.health_monitor import FishHealthMonitor
from src.web.app import app, manager
from src.web.monitor_service import MonitorService
import src.web.app as web_app


def main():
    parser = argparse.ArgumentParser(description="Fish Health Dashboard")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to video file")
    group.add_argument("--camera", type=int, help="Camera device ID")
    parser.add_argument("--model", default="models/custom_trained/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser automatically")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="Detection confidence threshold")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Dashboard target FPS (default: 15)")
    args = parser.parse_args()

    source = args.video if args.video else args.camera

    print("=" * 60)
    print("FISH HEALTH DASHBOARD")
    print("=" * 60)
    print(f"  Source: {source}")
    print(f"  Model: {args.model}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Dashboard FPS: {args.fps}")
    print(f"  URL: http://{args.host}:{args.port}")
    print("=" * 60)

    monitor = FishHealthMonitor(
        detector_model_path=args.model,
        detection_confidence=args.confidence,
        detection_iou_threshold=0.50,
        tracker_max_age=360,
        enable_visualization=True,
    )
    monitor.tracker.min_hits = 6
    monitor.tracker.iou_threshold = 0.20
    monitor.tracker.appearance_weight = 0.45

    service = MonitorService(source, monitor, manager, target_fps=args.fps)
    web_app.monitor_service = service

    @app.on_event("startup")
    async def startup():
        loop = asyncio.get_event_loop()
        service.start(loop)
        if not args.no_browser:
            webbrowser.open(f"http://{args.host}:{args.port}")

    @app.on_event("shutdown")
    async def shutdown():
        service.stop()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
