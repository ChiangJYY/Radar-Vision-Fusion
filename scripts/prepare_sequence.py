"""Convert camera JSON/DAT and radar DAT into the common inference format."""
import argparse
from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    camera = parser.add_mutually_exclusive_group()
    camera.add_argument("--camera-json", help="Output of run_visual.py")
    camera.add_argument("--camera-dat")
    parser.add_argument("--radar-dat")
    parser.add_argument("--calibration", help="Required for DAT in camera coordinates")
    parser.add_argument("--camera-coordinates", choices=["camera", "shared"], default="camera")
    parser.add_argument("--camera-xyz-order", choices=["xyz", "xzy"], default="xyz")
    parser.add_argument("--camera-label", type=int, default=0, help="DAT camera ID is the detector class label; default COCO person=0")
    parser.add_argument("--timestamp-format", choices=["hhmmssmmm", "seconds", "unix_ms"], default="hhmmssmmm")
    parser.add_argument("--tolerance", type=float, default=.05)
    parser.add_argument("--output", required=True, help="Destination aligned detections JSON path")
    args = parser.parse_args()
    if not any((args.camera_json, args.camera_dat, args.radar_dat)):
        parser.error("Provide at least one camera or radar input")
    from radar_vision_fusion.io import load_json, save_json
    from radar_vision_fusion.calibration.config import load_calibration
    from radar_vision_fusion.preprocessing.dat_io import read_sensor_dat
    from radar_vision_fusion.preprocessing.alignment import align_camera_radar
    camera_frames, radar_frames = [], []
    if args.camera_json:
        camera_frames = [{"timestamp": f["timestamp"], "detections": f["camera"]} for f in load_json(args.camera_json)["frames"]]
    if args.camera_dat:
        camera_frames = read_sensor_dat(args.camera_dat, "camera", timestamp_format=args.timestamp_format,
                                        camera_xyz_order=args.camera_xyz_order)
        transform = None
        if args.camera_coordinates == "camera":
            if not args.calibration:
                parser.error("Camera-coordinate DAT requires --calibration")
            _, transform = load_calibration(args.calibration)
        for frame in camera_frames:
            frame["detections"] = [record for record in frame["detections"] if record["id"] == args.camera_label]
            for detection in frame["detections"]:
                xyz = detection.pop("xyz")
                shared = transform.inverse().apply(xyz) if transform else xyz
                detection["xy"] = list(shared[:2])
    if args.radar_dat:
        radar_frames = read_sensor_dat(args.radar_dat, "radar", timestamp_format=args.timestamp_format)
        for frame in radar_frames:
            for detection in frame["detections"]:
                detection["xy"] = detection.pop("xyz")[:2]
    output = align_camera_radar(camera_frames, radar_frames, tolerance=args.tolerance)
    save_json(args.output, output)
    print(f"Saved {len(output['frames'])} aligned frames to {args.output}")


if __name__ == "__main__":
    main()
