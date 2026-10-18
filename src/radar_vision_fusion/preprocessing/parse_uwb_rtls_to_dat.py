#!/usr/bin/env python3
"""Parse UWB RTLS text logs into a flat .dat file.

This script extracts only tag position lines like:
	16:53:41:655 TAG0[-1.206,-0.350,2.300]

Frames are defined by timestamps (HH:MM:SS:mmm). If two consecutive frame
timestamps are very close (< merge_window_s, default 0.1s), the later frame is
merged into the earlier one (later detections are treated as detected at the
previous moment).

Optional time range filtering:
- Use --start/--end to keep only frames in a timestamp range (inclusive).
- The filter is applied AFTER merging so that near-by detections are still
	considered part of the earlier frame.

Output format:
	<total_frames>

	<frame_index> <num_targets> <timestamp9>
	<target_index> <x> <y> <z>

Where timestamp9 is HHMMSSmmm (e.g. 16:53:41:505 -> 165341505).

Usage:
	# Single file -> stdout
	python parse_uwb_rtls_to_dat.py --in /path/to/RTLS1_log.txt --out -

	# Folder (reads *.txt) -> output file
	python parse_uwb_rtls_to_dat.py --dir /path/to/uwb --out /path/to/out.dat
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT_PATH = "dataset/raw/uwb"
DEFAULT_MERGE_WINDOW_S = 0.1

# Optional time range (inclusive). Leave as None to disable filtering.
# Examples:
#   "20260115_165341_505" (YYYYMMDD_HHMMSS_mmm)
#   "165341505"          (HHMMSSmmm)
DEFAULT_START_TS = None
DEFAULT_END_TS = None


_HEADER_DATE_RE = re.compile(r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})_")

_TAG_POS_RE = re.compile(
	r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}):(?P<ms>\d{3})\s+"
	r"TAG(?P<tag>\d+)\[(?P<x>[-+]?\d*\.?\d+),(?P<y>[-+]?\d*\.?\d+),(?P<z>[-+]?\d*\.?\d+)\]$"
)


@dataclass(frozen=True)
class Detection:
	abs_ms: int
	ts9: int
	ts17: int
	tag_id: int
	x: float
	y: float
	z: float


@dataclass
class Frame:
	abs_ms: int
	ts9: int
	ts17: int
	# tag_id -> (x,y,z, abs_ms_of_measurement)
	tags: Dict[int, Tuple[float, float, float, int]]


def _read_lines(path: Path) -> Iterable[str]:
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for ln in f:
			yield ln.rstrip("\n")


def _tod_ms(h: int, m: int, s: int, ms: int) -> int:
	return ((h * 60 + m) * 60 + s) * 1000 + ms


def _ts9_from_hmsms(h: int, m: int, s: int, ms: int) -> int:
	return int(f"{h:02d}{m:02d}{s:02d}{ms:03d}")



def _find_header_date(lines: Iterable[str]) -> Optional[_dt.date]:
	for line in lines:
		m = _HEADER_DATE_RE.match(line.strip())
		if not m:
			continue
		try:
			y = int(m.group("y"))
			mo = int(m.group("mo"))
			d = int(m.group("d"))
			return _dt.date(y, mo, d)
		except Exception:
			return None
	return None


def parse_uwb_file(path: Path) -> List[Detection]:
	raw_lines = list(_read_lines(path))
	date = _find_header_date(raw_lines)
	date_ordinal = date.toordinal() if date else 0

	dets: List[Detection] = []
	for raw in raw_lines:
		line = raw.strip()
		m = _TAG_POS_RE.match(line)
		if not m:
			continue

		h = int(m.group("h"))
		mi = int(m.group("m"))
		s = int(m.group("s"))
		ms = int(m.group("ms"))
		tag_id = int(m.group("tag"))
		x = float(m.group("x"))
		y = float(m.group("y"))
		z = float(m.group("z"))

		tod = _tod_ms(h, mi, s, ms)
		abs_ms = date_ordinal * 86_400_000 + tod
		ts9 = _ts9_from_hmsms(h, mi, s, ms)
		if date is not None:
			ts17 = int(f"{date.year:04d}{date.month:02d}{date.day:02d}{h:02d}{mi:02d}{s:02d}{ms:03d}")
		else:
			ts17 = -1
		dets.append(Detection(abs_ms=abs_ms, ts9=ts9, ts17=ts17, tag_id=tag_id, x=x, y=y, z=z))

	dets.sort(key=lambda d: (d.abs_ms, d.tag_id))
	return dets


def _iter_txt_files(directory: Path, recursive: bool) -> List[Path]:
	if recursive:
		files = [p for p in directory.rglob("*.txt") if p.is_file()]
	else:
		files = [p for p in directory.glob("*.txt") if p.is_file()]
	files.sort()
	return files


def merge_detections_to_frames(dets: List[Detection], merge_window_s: float) -> List[Frame]:
	if not dets:
		return []

	merge_window_ms = int(round(merge_window_s * 1000.0))
	frames: List[Frame] = []

	for d in dets:
		if not frames:
			frames.append(
				Frame(abs_ms=d.abs_ms, ts9=d.ts9, ts17=d.ts17, tags={d.tag_id: (d.x, d.y, d.z, d.abs_ms)})
			)
			continue

		last = frames[-1]
		if (d.abs_ms - last.abs_ms) < merge_window_ms:
			prev = last.tags.get(d.tag_id)
			# Keep the latest measurement within the merged window.
			if prev is None or d.abs_ms >= prev[3]:
				last.tags[d.tag_id] = (d.x, d.y, d.z, d.abs_ms)
		else:
			frames.append(
				Frame(abs_ms=d.abs_ms, ts9=d.ts9, ts17=d.ts17, tags={d.tag_id: (d.x, d.y, d.z, d.abs_ms)})
			)

	return frames


def _fmt_xyz(v: float) -> str:
	# Match the typical 3-decimal formatting in logs.
	return f"{v:.3f}"


def write_dat(frames: List[Frame], out_path: Optional[Path]) -> None:
	lines: List[str] = []
	lines.append(str(len(frames)))
	lines.append("")

	for i, fr in enumerate(frames, start=1):
		tag_ids = sorted(fr.tags.keys())
		lines.append(f"{i} {len(tag_ids)} {fr.ts9}")
		for j, tag_id in enumerate(tag_ids):
			x, y, z, _t = fr.tags[tag_id]
			lines.append(f"{j} {_fmt_xyz(x)} {_fmt_xyz(y)} {_fmt_xyz(z)}")
		lines.append("")

	payload = "\n".join(lines).rstrip() + "\n"

	if out_path is None:
		print(payload, end="")
		return

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(payload, encoding="utf-8")


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


def filter_frames_by_range(frames: List[Frame], start: Any, end: Any) -> List[Frame]:
	if not start and not end:
		return frames

	use_digits = _pick_range_digits(start, end)
	start_key = _extract_key_from_digits(start or "", use_digits)
	end_key = _extract_key_from_digits(end or "", use_digits)
	if start_key is None and start:
		raise SystemExit(f"Invalid --start: {start}")
	if end_key is None and end:
		raise SystemExit(f"Invalid --end: {end}")
	if start_key is not None and end_key is not None and start_key > end_key:
		raise SystemExit(f"--start must be <= --end (got {start} > {end})")

	out: List[Frame] = []
	for fr in frames:
		k = fr.ts17 if use_digits == 17 else fr.ts9
		if use_digits == 17 and k < 0:
			# No date info in header; cannot do 17-digit filtering.
			continue
		if start_key is not None and k < start_key:
			continue
		if end_key is not None and k > end_key:
			continue
		out.append(fr)
	return out


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"path",
		nargs="?",
		default=None,
		help=f"Input path (file or folder). Default: {DEFAULT_INPUT_PATH}",
	)
	ap.add_argument("--path", dest="input_path", default=None, help="Input path (file or folder).")
	ap.add_argument("--in", dest="input_txt", default=None, help="Alias for --path when passing a single file.")
	ap.add_argument("--dir", dest="input_dir", default=None, help="Alias for --path when passing a folder.")
	ap.add_argument(
		"--out",
		required=True,
		help=(
			"Destination .dat path. Use '-' to write to stdout."
		),
	)
	ap.add_argument(
		"--merge-window-s",
		type=float,
		default=DEFAULT_MERGE_WINDOW_S,
		help=f"Merge frames when time diff < this value in seconds (default: {DEFAULT_MERGE_WINDOW_S}).",
	)
	ap.add_argument(
		"--start",
		default=DEFAULT_START_TS,
		help=(
			"Optional start timestamp (inclusive). Example: 20260115_165341_505 or 165341505. "
			"Supports either YYYYMMDDHHMMSSmmm (17 digits) or HHMMSSmmm (9 digits)."
		),
	)
	ap.add_argument(
		"--end",
		default=DEFAULT_END_TS,
		help=(
			"Optional end timestamp (inclusive). Example: 20260115_165343_105 or 165343105. "
			"Supports either YYYYMMDDHHMMSSmmm (17 digits) or HHMMSSmmm (9 digits)."
		),
	)
	rec = ap.add_mutually_exclusive_group()
	rec.add_argument("--recursive", action="store_true", help="Search *.txt recursively (default).")
	rec.add_argument("--no-recursive", action="store_true", help="Only read *.txt directly under --dir.")
	args = ap.parse_args()

	input_path_str = args.input_path or args.path or args.input_txt or args.input_dir or DEFAULT_INPUT_PATH
	input_path = Path(str(input_path_str)).expanduser().resolve()
	if not input_path.exists():
		raise SystemExit(f"Not found: {input_path}")

	recursive = True
	if bool(args.no_recursive):
		recursive = False

	txt_files: List[Path]
	if input_path.is_file():
		txt_files = [input_path]
	else:
		txt_files = _iter_txt_files(input_path, recursive=recursive)
		if not txt_files:
			raise SystemExit(f"No .txt files found under: {input_path}")

	all_dets: List[Detection] = []
	for p in txt_files:
		all_dets.extend(parse_uwb_file(p))

	all_dets.sort(key=lambda d: (d.abs_ms, d.tag_id))
	frames = merge_detections_to_frames(all_dets, merge_window_s=float(args.merge_window_s))
	frames = filter_frames_by_range(frames, start=args.start, end=args.end)

	out_arg = args.out
	out_path: Optional[Path]
	if out_arg == "-":
		out_path = None
	else:
		out_path = Path(out_arg).expanduser().resolve()

	write_dat(frames, out_path=out_path)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
