"""Reconstruct a metric stored region using a caller-supplied ECDR checkpoint."""
import argparse
from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV columns u,v,z_raw")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", required=True, help="Destination reconstructed boundary JSON path")
    args = parser.parse_args()
    import csv
    from radar_vision_fusion.boundary.inference import reconstruct_boundary
    from radar_vision_fusion.calibration.config import load_calibration
    from radar_vision_fusion.io import save_json
    with open(args.input, encoding="utf-8-sig", newline="") as handle:
        rows = [[float(row[k]) for k in ("u", "v", "z_raw")] for row in csv.DictReader(handle)]
    intrinsics, transform = load_calibration(args.calibration)
    output = reconstruct_boundary(rows, args.checkpoint, intrinsics, transform, k=args.k, device=args.device)
    save_json(args.output, output)
    print(f"Saved stored region and corrected observations to {args.output}")


if __name__ == "__main__":
    main()
