#!/usr/bin/env python3
"""Parse LRPD radar text logs into a flat .dat file.

Input format (as in LRPD_data_*.txt):
- Frames begin with a line like:
    Timestamp: 2026-01-08 17:34:03.760, Frame: 6819, Points: 5, Tracks: 1
- A tracks block follows:
    Tracks (TID,X,Y,Z,Vx,Vy,Vz,Ax,Ay,Az,G,ConfidenceLevel):
      7,0.407,5.822,2.000,0.135,0.044,-0.000,0.055,0.247,-0.000,4.000,1.000

Output format:
- First line: total_frames  (unique Timestamp lines)
- For each frame:
    <frame_index> <num_tracks> <timestamp9>
    <source_tid> <x> <y> <z> <vx> <vy> <vz> <ax> <ay> <az> <g> <confidence>

Timestamp9 definition:
- Use HHMMSSmmm from the Timestamp time portion.
  Example: 2026-01-08 17:34:03.760 -> 173403760

Usage:
    # Single file
    python parse_radar_lrpd_to_dat.py --in /path/to/LRPD_data_xxx.txt --out /path/to/out.dat

    # Folder (recommended)
    python parse_radar_lrpd_to_dat.py --dir /path/to/radar_folder --out /path/to/out.dat

Optional filtering (inclusive):
    --start 20260108_173402_549 --end 20260108_173403_351
    --start 173402549 --end 173403351

Specify the output file explicitly with --out.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Any


# You can edit these defaults directly, or override via CLI args.
# Recommended: point to a folder; the script will read all LRPD_data_*.txt inside.
DEFAULT_INPUT_PATH = "dataset/raw/radar"

# Optional time range (inclusive). Leave as None to disable filtering.
# Examples:
#   "20260108_173402_549"  (YYYYMMDD_HHMMSS_mmm)
#   "173402549"           (HHMMSSmmm)
DEFAULT_START_TS = None
DEFAULT_END_TS = None


_TS_LINE_RE = re.compile(
    r"^Timestamp:\s*(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s*,"
)


def _timestamp9_from_dt_str(dt_str: str) -> int:
    # dt_str: YYYY-MM-DD HH:MM:SS.mmm
    # We only need HHMMSSmmm
    # Example: 2026-01-08 17:34:03.760 -> 173403760
    time_part = dt_str.split()[1]
    hh, mm, rest = time_part.split(":")
    ss, mmm = rest.split(".")
    return int(f"{int(hh):02d}{int(mm):02d}{int(ss):02d}{int(mmm):03d}")


def _timestamp17_from_dt_str(dt_str: str) -> int:
    # dt_str: YYYY-MM-DD HH:MM:SS.mmm -> YYYYMMDDHHMMSSmmm (17 digits)
    date_part, time_part = dt_str.split()
    y, mo, d = date_part.split("-")
    hh, mm, rest = time_part.split(":")
    ss, mmm = rest.split(".")
    return int(f"{int(y):04d}{int(mo):02d}{int(d):02d}{int(hh):02d}{int(mm):02d}{int(ss):02d}{int(mmm):03d}")


@dataclass
class Track:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    g: float
    track_id: int = 0
    confidence: float = 1.0


@dataclass
class Frame:
    timestamp9: int
    timestamp17: int
    tracks: List[Track]


def _parse_track_csv_line(line: str) -> Optional[Track]:
    # Expected CSV: TID,X,Y,Z,Vx,Vy,Vz,Ax,Ay,Az,G,ConfidenceLevel
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 11:
        return None

    try:
        track_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        vx = float(parts[4])
        vy = float(parts[5])
        vz = float(parts[6])
        ax = float(parts[7])
        ay = float(parts[8])
        az = float(parts[9])
        g = float(parts[10])
        confidence = float(parts[11]) if len(parts) > 11 else 1.0
        return Track(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, ax=ax, ay=ay, az=az,
                     g=g, track_id=track_id, confidence=confidence)
    except Exception:
        return None


def _read_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            yield ln.rstrip("\n")


def parse_lrpd_file(path: Path) -> List[Frame]:
    frames: List[Frame] = []

    current_ts9: Optional[int] = None
    current_ts17: Optional[int] = None
    current_tracks: List[Track] = []
    in_tracks_block = False

    def flush_frame() -> None:
        nonlocal current_ts9, current_ts17, current_tracks, in_tracks_block
        if current_ts9 is None or current_ts17 is None:
            return
        frames.append(Frame(timestamp9=int(current_ts9), timestamp17=int(current_ts17), tracks=list(current_tracks)))
        current_ts9 = None
        current_ts17 = None
        current_tracks = []
        in_tracks_block = False

    for raw_line in _read_lines(path):
        line = raw_line.strip()
        if not line:
            in_tracks_block = False
            continue

        m = _TS_LINE_RE.match(line)
        if m:
            flush_frame()
            dt = m.group("dt")
            current_ts9 = _timestamp9_from_dt_str(dt)
            current_ts17 = _timestamp17_from_dt_str(dt)
            continue

        if line.startswith("Tracks ("):
            in_tracks_block = True
            continue

        if in_tracks_block:
            # Track CSV lines typically start with an integer TID.
            # We keep it lenient: try parse and accept if successful.
            tr = _parse_track_csv_line(line)
            if tr is not None:
                current_tracks.append(tr)
            continue

    flush_frame()
    return frames


def _iter_radar_files(directory: Path, recursive: bool) -> list[Path]:
    # Typical naming: LRPD_data_YYYYMMDD_HHMMSS.txt
    pat = "LRPD_data_*.txt"
    if recursive:
        files = [p for p in directory.rglob(pat) if p.is_file()]
    else:
        files = [p for p in directory.glob(pat) if p.is_file()]
    files.sort()
    return files


def _extract_key_from_digits(s: Any, n_digits: int) -> int | None:
    if s is None:
        return None
    s = str(s)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < n_digits:
        return None
    return int(digits[-n_digits:])


def _pick_range_digits(start: Any, end: Any) -> int:
    # Prefer 17-digit comparisons when possible.
    s17 = _extract_key_from_digits(start, 17)
    e17 = _extract_key_from_digits(end, 17)
    return 17 if (s17 is not None or e17 is not None) else 9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path",
        nargs="?",
        default=None,
        help=f"Input path (file or folder). Default: {DEFAULT_INPUT_PATH}",
    )
    ap.add_argument(
        "--path",
        dest="input_path",
        default=None,
        help=f"Input path (file or folder). Default: {DEFAULT_INPUT_PATH}",
    )
    # Backward-compatible aliases (you can still use these):
    ap.add_argument("--in", dest="input_txt", default=None, help="Alias for --path when passing a single file.")
    ap.add_argument("--dir", dest="input_dir", default=None, help="Alias for --path when passing a folder.")
    ap.add_argument("--out", dest="output_dat", required=True, help="Destination .dat path")
    ap.add_argument(
        "--start",
        default=DEFAULT_START_TS,
        help=(
            "Optional start timestamp (inclusive). Example: 20260108_173402_549 or 173402549. "
            "If 17-digit is provided, compares by YYYYMMDDHHMMSSmmm; otherwise compares by HHMMSSmmm."
        ),
    )
    ap.add_argument(
        "--end",
        default=DEFAULT_END_TS,
        help=(
            "Optional end timestamp (inclusive). Example: 20260108_173403_351 or 173403351. "
            "If 17-digit is provided, compares by YYYYMMDDHHMMSSmmm; otherwise compares by HHMMSSmmm."
        ),
    )

    rec = ap.add_mutually_exclusive_group()
    rec.add_argument("--recursive", action="store_true", help="Search radar files recursively (default).")
    rec.add_argument("--no-recursive", action="store_true", help="Only read radar files directly under --dir.")
    args = ap.parse_args()

    recursive = True
    if bool(args.no_recursive):
        recursive = False

    input_path_str = (
        args.input_path
        or args.path
        or args.input_txt
        or args.input_dir
        or DEFAULT_INPUT_PATH
    )

    input_path = Path(str(input_path_str)).expanduser().resolve()

    frames: List[Frame] = []
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if input_path.is_file():
        frames = parse_lrpd_file(input_path)
    elif input_path.is_dir():
        radar_files = _iter_radar_files(input_path, recursive=recursive)
        if not radar_files:
            raise SystemExit(f"No LRPD_data_*.txt found under: {input_path}")
        for p in radar_files:
            frames.extend(parse_lrpd_file(p))
    else:
        raise SystemExit(f"Input path must be a file or directory: {input_path}")

    out_path = Path(args.output_dat).expanduser().resolve()
    if out_path.exists() and out_path.is_dir():
        out_path = out_path / "radar_lrpd_summary.dat"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional time range filtering.
    if args.start is not None or args.end is not None:
        use_digits = _pick_range_digits(args.start, args.end)
        start_key = _extract_key_from_digits(args.start, use_digits) if args.start is not None else None
        end_key = _extract_key_from_digits(args.end, use_digits) if args.end is not None else None
        if start_key is None and args.start is not None:
            raise SystemExit(f"Invalid --start: {args.start}")
        if end_key is None and args.end is not None:
            raise SystemExit(f"Invalid --end: {args.end}")
        if start_key is not None and end_key is not None and start_key > end_key:
            raise SystemExit(f"--start must be <= --end (got {args.start} > {args.end})")

        filtered: List[Frame] = []
        for fr in frames:
            k = fr.timestamp17 if use_digits == 17 else fr.timestamp9
            if start_key is not None and k < start_key:
                continue
            if end_key is not None and k > end_key:
                continue
            filtered.append(fr)
        frames = filtered

    # Sort frames by absolute time (17-digit), stable by timestamp9.
    frames.sort(key=lambda fr: (fr.timestamp17, fr.timestamp9))

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{len(frames)}\n")
        for frame_idx, fr in enumerate(frames, start=1):
            f.write(f"{frame_idx} {len(fr.tracks)} {fr.timestamp9}\n")
            for track_idx, tr in enumerate(fr.tracks, start=1):
                f.write(
                    f"{tr.track_id} {tr.x:.6f} {tr.y:.6f} {tr.z:.6f} "
                    f"{tr.vx:.6f} {tr.vy:.6f} {tr.vz:.6f} "
                    f"{tr.ax:.6f} {tr.ay:.6f} {tr.az:.6f} {tr.g:.6f} {tr.confidence:.6f}\n"
                )

    print(f"[OK] wrote: {out_path}  (frames={len(frames)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
