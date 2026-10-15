"""Evaluate user-generated boundaries or tracks without the private experiment system."""
import argparse
from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["boundary", "tracking"])
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, help="Destination metrics JSON path")
    args = parser.parse_args()
    import numpy as np
    from radar_vision_fusion.io import load_json, save_json
    prediction, reference = load_json(args.prediction), load_json(args.reference)
    if args.task == "boundary":
        from radar_vision_fusion.evaluation.boundary import boundary_metrics
        metrics = boundary_metrics(prediction["region"], reference["region"])
    else:
        from radar_vision_fusion.evaluation.ospa import perf_asses, convert_to_track_array
        predicted_frames, reference_frames = prediction["frames"], reference["frames"]
        times = [f["timestamp"] for f in predicted_frames]
        if not times or times != [f["timestamp"] for f in reference_frames]:
            raise ValueError("Prediction and reference must have matching nonempty frame timestamps; no implicit GT alignment")
        def array(frames):
            data = {i: [[track["track_id"], *track["xy"]] for track in f["tracks"]] for i, f in enumerate(frames, 1)}
            return convert_to_track_array(data, len(frames))
        ospa, localization, cardinality, _, _ = perf_asses(array(reference_frames), array(predicted_frames))
        metrics = {"ospa_mean_m": float(np.mean(ospa)), "localization_mean_m": float(np.mean(localization)),
                   "cardinality_mean_m": float(np.mean(cardinality)), "frames": len(times), "cutoff_m": 3., "order": 2}
    save_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
