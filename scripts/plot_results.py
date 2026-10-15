"""Plot caller-generated trajectories, stored region and signed distances."""
import argparse
from pathlib import Path
from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Output of run_inference.py")
    parser.add_argument("--output", required=True, help="Destination figure path")
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from radar_vision_fusion.io import load_json
    data = load_json(args.input)
    histories = {}
    for frame in data["frames"]:
        for track in frame["tracks"]:
            histories.setdefault(track["track_id"], []).append((frame["timestamp"], *track["xy"], track["signed_distance"]))
    figure, (plan, distance) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    polygon = np.asarray(data["region"])
    closed = np.vstack([polygon, polygon[0]])
    plan.plot(closed[:, 0], closed[:, 1], "k--", label="Stored region")
    for identifier, values in histories.items():
        rows = np.asarray(values)
        plan.plot(rows[:, 1], rows[:, 2], label=f"Track {identifier}")
        distance.plot(rows[:, 0], rows[:, 3], label=f"Track {identifier}")
    plan.set(xlabel="X (m)", ylabel="Y (m)", aspect="equal")
    distance.set(xlabel="Time (s)", ylabel="Signed distance (m)")
    distance.axhline(0, color="black", linewidth=.8)
    plan.legend()
    if histories:
        distance.legend()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
