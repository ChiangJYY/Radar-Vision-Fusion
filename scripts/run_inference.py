"""Run tracking and region monitoring on caller-provided detection JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support a source checkout as well as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_vision_fusion.fusion.inference import FixedNoisePredictor, LearnedNoisePredictor
from radar_vision_fusion.io import load_detection_sequence, load_json, save_json
from radar_vision_fusion.monitoring import event_list, region_relation
from radar_vision_fusion.tracking import TrackConfig, Tracker, TrackerConfig


def run_monitoring(frame_pairs, config: dict, *, checkpoint=None, device="cpu") -> dict:
    polygon = config.get("region")
    if polygon is None:
        raise ValueError("Fill config.region with your stored metric polygon or pass --region-file")
    region_relation([0., 0.], polygon)
    mode = config.get("mode", "fixed")
    if mode == "learned":
        if checkpoint is None:
            raise ValueError("learned mode requires an explicit --checkpoint")
        predictor = LearnedNoisePredictor(checkpoint, device=device)
    elif mode == "fixed":
        if checkpoint is not None:
            raise ValueError("--checkpoint requires --mode learned")
        predictor = FixedNoisePredictor(r_cam=config.get("camera_noise", [3, 3]),
                                        r_rad=config.get("radar_noise", [3, 3, 1, 1]),
                                        q=float(config.get("process_noise", 0.025)))
    else:
        raise ValueError("mode must be fixed or learned")
    window = int(config.get("temporal_window", 25))
    if mode == "learned" and window > predictor.max_seq_len:
        raise ValueError("temporal_window exceeds the checkpoint's supported history")
    tracker = Tracker(
        scene_name="sequence", predictor=predictor,
        sensor_mode=config.get("sensor_mode", "fusion"),
        config=TrackerConfig(camera_gating=float(config.get("camera_gating", 2)),
                             radar_gating=float(config.get("radar_gating", 2))),
        track_config=TrackConfig(temporal_window=window, confirm_hits=int(config.get("confirm_hits", 2))),
    )
    frames, histories = [], {}
    for index, (camera_frame, radar_frame) in enumerate(frame_pairs):
        tracks = tracker.step(camera_frame, radar_frame)
        records = []
        for track in tracks:
            position, velocity = track.position(), track.velocity()
            record = {"track_id": int(track.id()), "xy": [position.x, position.y],
                      "velocity": [velocity.x, velocity.y],
                      **region_relation([position.x, position.y], polygon)}
            records.append(record)
            histories.setdefault(track.id(), []).append((index, camera_frame.timestamp, record["signed_distance"]))
        frames.append({"frame": index, "timestamp": camera_frame.timestamp, "tracks": records,
                       "inside_count": sum(item["inside"] for item in records)})
    events = []
    for track_id, rows in histories.items():
        frame, timestamp, distance = zip(*rows)
        for event in event_list(frame, timestamp, distance,
                                minimum_run=int(config.get("event_minimum_run", 3)),
                                maximum_gap=float(config.get("event_maximum_gap", 0.35))):
            events.append({"track_id": int(track_id), **event})
    events.sort(key=lambda item: (item["time"], item["track_id"]))
    return {"schema_version": 1, "mode": mode, "region": polygon, "frames": frames, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/inference.json")
    parser.add_argument("--region-file", type=Path, help="run_boundary.py output or JSON with a region field")
    parser.add_argument("--output", type=Path, required=True, help="Destination tracks and events JSON path")
    parser.add_argument("--mode", choices=["fixed", "learned"])
    parser.add_argument("--sensor-mode", choices=["fusion", "camera_only", "radar_only"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_json(args.config)
    if args.mode:
        config["mode"] = args.mode
    if args.sensor_mode:
        config["sensor_mode"] = args.sensor_mode
    if args.region_file:
        config["region"] = load_json(args.region_file)["region"]
    output = run_monitoring(load_detection_sequence(args.input), config, checkpoint=args.checkpoint, device=args.device)
    save_json(args.output, output)
    print(json.dumps({"mode": output["mode"], "frames": len(output["frames"]),
                      "events": len(output["events"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
