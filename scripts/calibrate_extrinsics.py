"""Calibrate radar-to-camera transform from user-provided correspondences."""
import argparse
from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV: radar_x,radar_y,radar_z,u,v")
    parser.add_argument("--intrinsics", required=True, help="JSON object with fx,fy,cx,cy")
    parser.add_argument("--initial", type=float, nargs=6, required=True, metavar=("tx", "ty", "tz", "rx", "ry", "rz"))
    parser.add_argument("--output", required=True, help="Destination calibration JSON path")
    args = parser.parse_args()
    import csv
    from radar_vision_fusion.calibration.extrinsics import calibrate_extrinsics
    from radar_vision_fusion.geometry.projection import CameraIntrinsics
    from radar_vision_fusion.io import load_json, save_json
    with open(args.input, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    xyz = [[float(r[k]) for k in ("radar_x", "radar_y", "radar_z")] for r in rows]
    uv = [[float(r[k]) for k in ("u", "v")] for r in rows]
    intrinsics = CameraIntrinsics(**load_json(args.intrinsics))
    transform, metrics = calibrate_extrinsics(xyz, uv, intrinsics, args.initial)
    save_json(args.output, {"intrinsics": intrinsics.__dict__, "camera_from_radar": {
        "rotation": transform.rotation.tolist(), "translation": transform.translation.tolist()}, "quality": metrics})
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
