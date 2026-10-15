"""Generate calibrated camera detections from caller-owned images and metric-depth weights."""
import argparse
from pathlib import Path
from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with image,timestamp columns; timestamps in seconds")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--detector-config", default=str(ROOT / "configs/rtdetrv2_r50vd_m.yml"))
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--depth-checkpoint", required=True)
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitb")
    parser.add_argument("--max-depth", type=float, default=50.)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=.6)
    parser.add_argument("--output", required=True, help="Destination camera observations JSON path")
    args = parser.parse_args()
    import csv
    import math
    import cv2
    from PIL import Image
    from radar_vision_fusion.calibration.config import load_calibration
    from radar_vision_fusion.perception.models import RTDETRDetector, MetricDepthEstimator
    from radar_vision_fusion.perception.observations import observations_from_image
    from radar_vision_fusion.io import save_json
    intrinsics, transform = load_calibration(args.calibration)
    manifest = Path(args.manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not r.get("image") or not r.get("timestamp") for r in rows):
        raise ValueError("Image manifest must contain nonempty image,timestamp rows")
    times = [float(row["timestamp"]) for row in rows]
    if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Image timestamps must be finite seconds in increasing order")
    detector = RTDETRDetector(ROOT / "src/third_party/RT-DETR/rtdetrv2_pytorch", args.detector_config,
                              args.detector_checkpoint, device=args.device, threshold=args.threshold)
    depth = MetricDepthEstimator(ROOT / "src/third_party/Depth-Anything-V2", args.depth_checkpoint,
                                  encoder=args.encoder, max_depth=args.max_depth, device=args.device)
    frames = []
    for row, timestamp in zip(rows, times):
        path = Path(row["image"])
        if not path.is_absolute():
            path = manifest.parent / path
        pil = Image.open(path).convert("RGB")
        import numpy as np
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        metric = depth(bgr)
        if metric.shape != (pil.height, pil.width):
            metric = cv2.resize(metric, pil.size, interpolation=cv2.INTER_CUBIC)
        records = observations_from_image(detector(pil), metric, intrinsics, transform)
        frames.append({"timestamp": timestamp, "camera": records})
    save_json(args.output, {"schema_version": 1, "frames": frames})
    print(f"Saved {len(frames)} camera frames to {args.output}")


if __name__ == "__main__":
    main()
