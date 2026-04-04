"""
Video Calibration Script for Fish Health Monitoring System

Finds optimal detection and tracking thresholds by running the model
on a video with different parameter combinations.

Modes:
  sweep       - Automated grid search over parameters, prints ranked results
  interactive - OpenCV window with live sliders for visual tuning
  both        - Sweep first, then interactive pre-loaded with best params

Usage:
  python scripts/calibrate_video.py --video VID-20260327-WA0001.mp4
  python scripts/calibrate_video.py --video VID-20260327-WA0001.mp4 --mode sweep
  python scripts/calibrate_video.py --video VID-20260327-WA0001.mp4 --mode interactive
"""

import sys
import os
import argparse
import time

import cv2
import numpy as np
import yaml

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.fish_detector import FishDetector, Detection
from src.tracking.fish_tracker import FishTracker


# ─────────────────────────────────────────────────────────────────────
# Section 1: Frame Sampling
# ─────────────────────────────────────────────────────────────────────

def sample_frames(video_path, num_frames=20):
    """Sample evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append((frame, idx / fps, int(idx)))
    cap.release()
    return frames, fps, total


def read_clip(video_path, start_sec, duration_sec, skip=0):
    """Read consecutive frames from a clip segment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    idx = start_frame
    while idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame, idx / fps, idx))
        idx += 1
        # Skip frames if requested
        for _ in range(skip):
            cap.read()
            idx += 1

    cap.release()
    return frames, fps


# ─────────────────────────────────────────────────────────────────────
# Section 2: Automated Grid Sweep
# ─────────────────────────────────────────────────────────────────────

def run_detection_sweep(detector, frames):
    """Sweep detection confidence and NMS IoU, return ranked results."""
    conf_values = [0.25, 0.35, 0.45, 0.55, 0.65]
    iou_values = [0.3, 0.45, 0.6]

    results = []
    total_combos = len(conf_values) * len(iou_values)
    combo_idx = 0

    for conf in conf_values:
        for iou in iou_values:
            combo_idx += 1
            detector.confidence_threshold = conf
            detector.iou_threshold = iou

            counts = []
            confidences = []
            area_ratios = []

            for frame, ts, fidx in frames:
                h, w = frame.shape[:2]
                frame_area = h * w
                dets = detector.detect(frame, timestamp=ts)
                counts.append(len(dets))
                for d in dets:
                    confidences.append(d.confidence)
                    x1, y1, x2, y2 = d.bbox
                    area_ratios.append((x2 - x1) * (y2 - y1) / frame_area)

            mean_count = np.mean(counts) if counts else 0
            std_count = np.std(counts) if counts else 0
            mean_conf = np.mean(confidences) if confidences else 0
            mean_area = np.mean(area_ratios) if area_ratios else 0

            # Heuristic score: favor consistent count with high confidence
            max_std = max(std_count, 1.0)
            score = mean_count * (1 - std_count / (max_std + 5.0)) * mean_conf

            results.append({
                "conf": conf,
                "nms_iou": iou,
                "mean_detections": round(mean_count, 1),
                "std_detections": round(std_count, 2),
                "mean_confidence": round(mean_conf, 3),
                "mean_bbox_area": round(mean_area, 4),
                "score": round(score, 3),
            })

            print(f"  [{combo_idx}/{total_combos}] conf={conf:.2f} iou={iou:.2f} "
                  f"→ avg {mean_count:.1f} fish (±{std_count:.1f}), "
                  f"avg conf={mean_conf:.3f}, score={score:.3f}")

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def run_tracker_sweep(detector, clip_frames, best_det_conf, best_det_iou):
    """Sweep tracker parameters using cached detections from a clip."""
    # Step 1: Run detection once on the clip and cache results
    detector.confidence_threshold = best_det_conf
    detector.iou_threshold = best_det_iou

    print(f"\n  Running detection on {len(clip_frames)} clip frames "
          f"(conf={best_det_conf}, iou={best_det_iou})...")
    cached_detections = []
    for frame, ts, fidx in clip_frames:
        dets = detector.detect(frame, timestamp=ts)
        cached_detections.append((dets, frame))

    # Step 2: Sweep tracker parameters
    max_age_values = [60, 150, 300]
    min_hits_values = [2, 3, 5]
    tracker_iou_values = [0.15, 0.2, 0.3]
    appearance_values = [0.0, 0.2, 0.4, 0.6]

    total_combos = (len(max_age_values) * len(min_hits_values)
                    * len(tracker_iou_values) * len(appearance_values))
    print(f"  Testing {total_combos} tracker parameter combinations...")

    results = []
    combo_idx = 0

    for max_age in max_age_values:
        for min_hits in min_hits_values:
            for t_iou in tracker_iou_values:
                for app_w in appearance_values:
                    combo_idx += 1
                    tracker = FishTracker(
                        max_age=max_age,
                        min_hits=min_hits,
                        iou_threshold=t_iou,
                        appearance_weight=app_w,
                        enable_reid=(app_w > 0),
                    )

                    active_counts = []
                    all_ids = set()

                    for dets, frame in cached_detections:
                        tracks = tracker.update(dets, frame)
                        active_counts.append(len(tracks))
                        for t in tracks:
                            all_ids.add(t.track_id)

                    mean_active = np.mean(active_counts) if active_counts else 0
                    std_active = np.std(active_counts) if active_counts else 0
                    unique_ids = len(all_ids)

                    # Score: penalize ID fragmentation and instability
                    # Ideal: unique_ids close to mean_active, low std
                    if mean_active > 0:
                        frag_ratio = unique_ids / mean_active  # 1.0 = perfect
                        stability = 1 / (1 + std_active)
                        score = mean_active * stability / frag_ratio
                    else:
                        score = 0

                    results.append({
                        "max_age": max_age,
                        "min_hits": min_hits,
                        "tracker_iou": t_iou,
                        "appearance_weight": app_w,
                        "mean_active_tracks": round(mean_active, 1),
                        "std_active_tracks": round(std_active, 2),
                        "unique_ids": unique_ids,
                        "score": round(score, 3),
                    })

                    if combo_idx % 20 == 0:
                        print(f"    [{combo_idx}/{total_combos}] ...")

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def run_sweep(video_path, model_path, device, num_sample_frames=20):
    """Run full detection + tracker sweep."""
    print("=" * 70)
    print("AUTOMATED PARAMETER SWEEP")
    print("=" * 70)

    # Load model once
    print(f"\nLoading model: {model_path}")
    detector = FishDetector(
        model_path=model_path,
        confidence_threshold=0.5,
        device=device,
    )

    # Sample frames for detection sweep
    print(f"\nSampling {num_sample_frames} frames from video...")
    frames, fps, total_frames = sample_frames(video_path, num_sample_frames)
    duration = total_frames / fps
    print(f"  Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

    # Detection sweep
    print(f"\n--- Detection Parameter Sweep ({len(frames)} frames) ---")
    det_results = run_detection_sweep(detector, frames)

    print(f"\n  Top 5 detection configs:")
    print(f"  {'Rank':<5} {'Conf':<6} {'NMS IoU':<8} {'Avg Fish':<10} "
          f"{'Std':<6} {'Avg Conf':<10} {'Score':<8}")
    print(f"  {'-'*55}")
    for i, r in enumerate(det_results[:5]):
        print(f"  {i+1:<5} {r['conf']:<6.2f} {r['nms_iou']:<8.2f} "
              f"{r['mean_detections']:<10.1f} {r['std_detections']:<6.2f} "
              f"{r['mean_confidence']:<10.3f} {r['score']:<8.3f}")

    best_det = det_results[0]
    print(f"\n  Best detection: conf={best_det['conf']}, nms_iou={best_det['nms_iou']}")

    # Tracker sweep on a 10-second clip from the middle
    mid_sec = max(0, duration / 2 - 5)
    clip_duration = min(10, duration)
    print(f"\n--- Tracker Parameter Sweep (clip: {mid_sec:.0f}s-{mid_sec+clip_duration:.0f}s) ---")
    clip_frames, _ = read_clip(video_path, mid_sec, clip_duration)
    tracker_results = run_tracker_sweep(
        detector, clip_frames, best_det["conf"], best_det["nms_iou"]
    )

    print(f"\n  Top 5 tracker configs:")
    print(f"  {'Rank':<5} {'MaxAge':<7} {'MinHits':<8} {'TrkIoU':<8} "
          f"{'AppWt':<7} {'AvgTrk':<8} {'IDs':<5} {'Score':<8}")
    print(f"  {'-'*60}")
    for i, r in enumerate(tracker_results[:5]):
        print(f"  {i+1:<5} {r['max_age']:<7} {r['min_hits']:<8} "
              f"{r['tracker_iou']:<8.2f} {r['appearance_weight']:<7.1f} "
              f"{r['mean_active_tracks']:<8.1f} {r['unique_ids']:<5} "
              f"{r['score']:<8.3f}")

    best_trk = tracker_results[0]
    print(f"\n  Best tracker: max_age={best_trk['max_age']}, "
          f"min_hits={best_trk['min_hits']}, "
          f"iou={best_trk['tracker_iou']}, "
          f"appearance={best_trk['appearance_weight']}")

    return best_det, best_trk


# ─────────────────────────────────────────────────────────────────────
# Section 3: Interactive Preview
# ─────────────────────────────────────────────────────────────────────

def interactive_preview(video_path, model_path, device, initial_params=None):
    """Interactive OpenCV window with trackbar sliders for parameter tuning."""
    if initial_params is None:
        initial_params = {}

    # Defaults
    conf = initial_params.get("conf", 0.5)
    nms_iou = initial_params.get("nms_iou", 0.45)
    max_age = initial_params.get("max_age", 150)
    min_hits = initial_params.get("min_hits", 3)
    tracker_iou = initial_params.get("tracker_iou", 0.2)
    appearance_weight = initial_params.get("appearance_weight", 0.4)

    # Load model
    print("\nLoading model for interactive mode...")
    detector = FishDetector(
        model_path=model_path,
        confidence_threshold=conf,
        iou_threshold=nms_iou,
        device=device,
    )

    tracker = FishTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=tracker_iou,
        appearance_weight=appearance_weight,
        enable_reid=(appearance_weight > 0),
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create window and trackbars
    win = "Fish Detection Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(width, 1280), min(height, 720))

    # Trackbar ranges (integer values, we divide to get float)
    cv2.createTrackbar("Confidence x100", win, int(conf * 100), 95, lambda x: None)
    cv2.createTrackbar("NMS IoU x100", win, int(nms_iou * 100), 90, lambda x: None)
    cv2.createTrackbar("Max Age", win, max_age, 500, lambda x: None)
    cv2.createTrackbar("Min Hits", win, min_hits, 10, lambda x: None)
    cv2.createTrackbar("Trk IoU x100", win, int(tracker_iou * 100), 50, lambda x: None)
    cv2.createTrackbar("Appear Wt x100", win, int(appearance_weight * 100), 100, lambda x: None)

    paused = False
    prev_tracker_params = (max_age, min_hits, tracker_iou, appearance_weight)

    print("\n" + "=" * 70)
    print("INTERACTIVE CALIBRATION MODE")
    print("=" * 70)
    print("Controls:")
    print("  Sliders  - adjust parameters in real time")
    print("  SPACE    - pause / resume")
    print("  LEFT/RIGHT arrow - seek ±30 frames")
    print("  R        - reset tracker")
    print("  S        - save screenshot")
    print("  Q        - quit and save config")
    print("=" * 70)

    frame_idx = 0
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                frame_idx = 0
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Read trackbar values
        new_conf = max(cv2.getTrackbarPos("Confidence x100", win), 5) / 100.0
        new_nms = max(cv2.getTrackbarPos("NMS IoU x100", win), 5) / 100.0
        new_max_age = max(cv2.getTrackbarPos("Max Age", win), 10)
        new_min_hits = max(cv2.getTrackbarPos("Min Hits", win), 1)
        new_trk_iou = max(cv2.getTrackbarPos("Trk IoU x100", win), 5) / 100.0
        new_app_wt = cv2.getTrackbarPos("Appear Wt x100", win) / 100.0

        # Update detector params (lightweight, just attribute changes)
        detector.confidence_threshold = new_conf
        detector.iou_threshold = new_nms

        # Recreate tracker if params changed
        new_tracker_params = (new_max_age, new_min_hits, new_trk_iou, new_app_wt)
        if new_tracker_params != prev_tracker_params:
            tracker = FishTracker(
                max_age=new_max_age,
                min_hits=new_min_hits,
                iou_threshold=new_trk_iou,
                appearance_weight=new_app_wt,
                enable_reid=(new_app_wt > 0),
            )
            prev_tracker_params = new_tracker_params

        # Run detection and tracking
        t0 = time.time()
        dets = detector.detect(frame, timestamp=frame_idx / fps)
        tracks = tracker.update(dets, frame)
        proc_time = time.time() - t0

        # Draw visualization
        vis = frame.copy()

        # Draw tracks
        for track in tracks:
            bbox = track.get_current_bbox()
            x1, y1, x2, y2 = bbox

            # Color by track confidence
            avg_conf = (np.mean(list(track.confidence_history))
                        if track.confidence_history else 0.5)
            if avg_conf >= 0.7:
                color = (0, 255, 0)
            elif avg_conf >= 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 100, 255)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"ID:{track.track_id} ({avg_conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Trail
            positions = track.get_position_history(n=30)
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    cv2.line(vis, positions[i - 1], positions[i], color, 1)

        # Draw unmatched detections (lighter)
        tracked_ids = {t.track_id for t in tracks}
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Info overlay
        overlay = vis.copy()
        panel_h = 140
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 20
        sp = 22
        white = (255, 255, 255)
        yellow = (0, 255, 255)

        proc_fps = 1 / proc_time if proc_time > 0 else 0
        cv2.putText(vis, f"Frame: {frame_idx}/{total_frames}  "
                    f"Detections: {len(dets)}  Tracks: {len(tracks)}  "
                    f"FPS: {proc_fps:.1f}",
                    (10, y), font, 0.5, white, 1)
        y += sp
        cv2.putText(vis, f"Detection: conf={new_conf:.2f}  nms_iou={new_nms:.2f}",
                    (10, y), font, 0.5, yellow, 1)
        y += sp
        cv2.putText(vis, f"Tracker: max_age={new_max_age}  min_hits={new_min_hits}  "
                    f"iou={new_trk_iou:.2f}  appear_wt={new_app_wt:.2f}",
                    (10, y), font, 0.5, yellow, 1)
        y += sp

        if paused:
            cv2.putText(vis, "PAUSED", (10, y), font, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(vis, "SPACE=pause  Q=quit+save  R=reset  S=screenshot",
                        (10, y), font, 0.4, (180, 180, 180), 1)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1 if not paused else 50) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            tracker.reset()
            print("  Tracker reset.")
        elif key == ord("s"):
            fname = f"calibration_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, vis)
            print(f"  Screenshot saved: {fname}")
        elif key == 2:  # Left arrow
            new_pos = max(0, frame_idx - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            tracker.reset()
        elif key == 3:  # Right arrow
            new_pos = min(total_frames - 1, frame_idx + 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            tracker.reset()

    cap.release()
    cv2.destroyAllWindows()

    # Return final params
    return {
        "conf": new_conf,
        "nms_iou": new_nms,
        "max_age": new_max_age,
        "min_hits": new_min_hits,
        "tracker_iou": new_trk_iou,
        "appearance_weight": new_app_wt,
    }


# ─────────────────────────────────────────────────────────────────────
# Section 4: Config Output
# ─────────────────────────────────────────────────────────────────────

def output_config(params, output_path):
    """Print recommended config and optionally save to YAML."""
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)
    print(f"  detection.confidence_threshold: {params['conf']}")
    print(f"  detection.iou_threshold:        {params['nms_iou']}")
    print(f"  tracking.max_age:               {params['max_age']}")
    print(f"  tracking.min_hits:              {params['min_hits']}")
    print(f"  tracking.iou_threshold:         {params['tracker_iou']}")
    print(f"  tracking.appearance_weight:     {params['appearance_weight']}")
    print(f"  tracking.enable_reid:           {params['appearance_weight'] > 0}")

    print(f"\n  Python snippet:")
    print(f"  ───────────────")
    print(f"  monitor = FishHealthMonitor(")
    print(f'      detector_model_path="models/custom_trained/best.pt",')
    print(f"      detection_confidence={params['conf']},")
    print(f"      tracker_max_age={params['max_age']},")
    print(f"      enable_reid={params['appearance_weight'] > 0},")
    print(f"  )")

    # Build config dict
    config = {
        "detection": {
            "model_path": "models/custom_trained/best.pt",
            "confidence_threshold": params["conf"],
            "iou_threshold": params["nms_iou"],
            "device": None,
        },
        "tracking": {
            "max_age": params["max_age"],
            "min_hits": params["min_hits"],
            "iou_threshold": params["tracker_iou"],
            "appearance_weight": params["appearance_weight"],
            "enable_reid": params["appearance_weight"] > 0,
        },
    }

    # Load existing default config as base and merge
    default_config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "default_config.yaml"
    )
    if os.path.exists(default_config_path):
        with open(default_config_path, "r") as f:
            base = yaml.safe_load(f)
        base["detection"].update(config["detection"])
        base["tracking"].update(config["tracking"])
        config = base

    print(f"\n  Save calibrated config to {output_path}? [y/N] ", end="")
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer == "y":
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  Saved to: {output_path}")
    else:
        print("  Config not saved.")

    # Note about iou_threshold gap
    if params["nms_iou"] != 0.45:
        print(f"\n  NOTE: Your optimal NMS IoU ({params['nms_iou']}) differs from the "
              f"default (0.45).")
        print(f"  FishHealthMonitor now accepts detection_iou_threshold to forward this.")

    print("=" * 70)
    return config


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate fish detection/tracking parameters on a video"
    )
    parser.add_argument(
        "--video", default="VID-20260327-WA0001.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--model", default="models/custom_trained/best.pt",
        help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--mode", choices=["sweep", "interactive", "both"], default="both",
        help="Calibration mode (default: both)"
    )
    parser.add_argument(
        "--sample-frames", type=int, default=20,
        help="Number of frames to sample for detection sweep"
    )
    parser.add_argument(
        "--output-config", default="configs/calibrated_config.yaml",
        help="Output path for calibrated config YAML"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda, mps, cpu, or auto (default: auto)"
    )
    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    best_params = None

    if args.mode in ("sweep", "both"):
        best_det, best_trk = run_sweep(
            args.video, args.model, args.device, args.sample_frames
        )
        best_params = {
            "conf": best_det["conf"],
            "nms_iou": best_det["nms_iou"],
            "max_age": best_trk["max_age"],
            "min_hits": best_trk["min_hits"],
            "tracker_iou": best_trk["tracker_iou"],
            "appearance_weight": best_trk["appearance_weight"],
        }

    if args.mode in ("interactive", "both"):
        final_params = interactive_preview(
            args.video, args.model, args.device, best_params
        )
        best_params = final_params

    if best_params:
        output_config(best_params, args.output_config)


if __name__ == "__main__":
    main()
