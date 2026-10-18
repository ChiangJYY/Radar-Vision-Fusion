"""Strict readers for frame-block camera, LRPD radar and UWB DAT files."""
from pathlib import Path
import math


def timestamp_seconds(value, format="hhmmssmmm"):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Timestamp must be finite")
    if format == "seconds":
        return number
    if format == "unix_ms":
        return number / 1000.0
    if format != "hhmmssmmm":
        raise ValueError("Timestamp format must be hhmmssmmm, seconds or unix_ms")
    integer = int(round(number))
    if abs(number - integer) > 1e-3 or integer < 0:
        raise ValueError("HHMMSSmmm timestamps must be nonnegative integers")
    milliseconds = integer % 1000
    seconds = integer // 1000 % 100
    minutes = integer // 100000 % 100
    hours = integer // 10000000
    if hours > 23 or minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid HHMMSSmmm timestamp: {value}")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def read_sensor_dat(path, sensor, *, timestamp_format="hhmmssmmm", camera_xyz_order="xyz"):
    """Read detections without dropping empty frames or guessing clock units.

    Camera rows: id, three coordinates, x1,y1,x2,y2,u,v,score.
    ``camera_xyz_order`` explicitly declares the coordinate columns; the raw
    camera export uses x,y,z. Radar rows: id,x,y,z,vx,vy,vz,ax,ay,az,g[,confidence].
    UWB rows: id,x,y,z. Raw coordinates are preserved; calibration is separate.
    """
    if sensor not in {"camera", "radar", "uwb"}:
        raise ValueError("sensor must be camera, radar or uwb")
    if camera_xyz_order not in {"xyz", "xzy"}:
        raise ValueError("camera_xyz_order must be xyz or xzy")
    lines = [(i, line.strip()) for i, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1) if line.strip()]
    if not lines:
        raise ValueError(f"Empty DAT file: {path}")
    expected = int(lines[0][1])
    if expected < 0:
        raise ValueError("Frame count must be nonnegative")
    cursor, frames, day_offset, previous = 1, [], 0., None
    for _ in range(expected):
        if cursor >= len(lines):
            raise ValueError("DAT ended before all frame headers were read")
        line_no, header = lines[cursor]
        cursor += 1
        fields = header.split()
        if len(fields) != 3:
            raise ValueError(f"Malformed frame header on line {line_no}")
        index, count = int(fields[0]), int(fields[1])
        if count < 0:
            raise ValueError(f"Negative detection count on line {line_no}")
        timestamp = timestamp_seconds(fields[2], timestamp_format) + day_offset
        if previous is not None and timestamp < previous and timestamp_format == "hhmmssmmm" and previous - timestamp > 43200:
            day_offset += 86400
            timestamp += 86400
        if previous is not None and timestamp <= previous:
            raise ValueError(f"DAT timestamps must increase strictly (line {line_no})")
        previous = timestamp
        detections = []
        for _ in range(count):
            if cursor >= len(lines):
                raise ValueError("DAT ended inside a detection block")
            row_no, row = lines[cursor]
            cursor += 1
            values = [float(value) for value in row.split()]
            minimum = {"camera": 11, "radar": 11, "uwb": 4}[sensor]
            if len(values) < minimum or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Invalid {sensor} detection on line {row_no}")
            xyz = values[1:4]
            if sensor == "camera" and camera_xyz_order == "xzy":
                xyz = [xyz[0], xyz[2], xyz[1]]
            record = {"id": int(values[0]), "xyz": xyz}
            if sensor == "camera":
                record.update(bbox=values[4:8], center_uv=values[8:10], score=values[10])
            elif sensor == "radar":
                record.update(velocity=values[4:6], acceleration=values[7:9], g=values[10],
                              confidence=values[11] if len(values) > 11 else 1.)
            detections.append(record)
        frames.append({"frame": index, "timestamp": timestamp, "detections": detections})
    if cursor != len(lines):
        raise ValueError("Unexpected trailing records after the declared DAT frames")
    return frames
