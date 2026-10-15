"""Monotonic one-use synchronization without GT-driven alignment or interpolation."""
import math


def align_camera_radar(camera_frames, radar_frames, *, tolerance=0.05):
    """Pair frames within tolerance; retain unmatched frames and their empty sensor.

    A pair uses the later sensor timestamp. Every input frame is used once.
    Timestamps use the same seconds-based clock before calling this function.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    for frames in (camera_frames, radar_frames):
        times = [float(frame["timestamp"]) for frame in frames]
        if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Each sensor clock must increase strictly")
    i = j = 0
    result = []
    while i < len(camera_frames) or j < len(radar_frames):
        camera = camera_frames[i] if i < len(camera_frames) else None
        radar = radar_frames[j] if j < len(radar_frames) else None
        can_pair = camera is not None and radar is not None and abs(camera["timestamp"] - radar["timestamp"]) <= tolerance
        if can_pair and camera["timestamp"] < radar["timestamp"] and i + 1 < len(camera_frames):
            can_pair = camera_frames[i + 1]["timestamp"] > radar["timestamp"]
        if can_pair and radar["timestamp"] < camera["timestamp"] and j + 1 < len(radar_frames):
            can_pair = radar_frames[j + 1]["timestamp"] > camera["timestamp"]
        if can_pair:
            result.append({"timestamp": max(camera["timestamp"], radar["timestamp"]),
                           "camera": camera["detections"], "radar": radar["detections"]})
            i += 1
            j += 1
        elif radar is None or (camera is not None and camera["timestamp"] < radar["timestamp"]):
            result.append({"timestamp": camera["timestamp"], "camera": camera["detections"], "radar": []})
            i += 1
        else:
            result.append({"timestamp": radar["timestamp"], "camera": [], "radar": radar["detections"]})
            j += 1
    return {"schema_version": 1, "frames": result}
