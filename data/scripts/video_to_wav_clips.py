#!/usr/bin/env python3
"""
Simplified: Download videos and cut IRMAS-like WAV clips from a JSON manifest

Hardcoded IRMAS spec:
- Stereo (2 ch), 44_100 Hz
- 3.0 s clip length (non-overlapping, NO padding)
- Output: data/instruments/<label>/<video_id>/rng_<start>_<end>/tXXXXXXms.wav
- Metadata CSV: data/instruments/metadata.csv

Usage:
    python data/scripts/video_to_wav_clips.py --json data/manifests/chinese_instruments_data.json
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import shutil
import subprocess as sp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

# ---- Hardcoded constants ----
OUT_ROOT = Path("data/instruments")
TMP_DIR  = Path(".cache/video_tmp")
METADATA_CSV = Path("data/instruments/metadata.csv")
SR = 44_100
CHANNELS = 2
CLIP_SECONDS = 3.0
STRIDE_SECONDS = CLIP_SECONDS   # no overlap

# --------------------------- utils ---------------------------

def run(cmd: List[str], check: bool = True) -> sp.CompletedProcess:
    return sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, check=check)

def which(name: str) -> bool:
    return shutil.which(name) is not None

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class YTDLResult:
    url: str
    video_id: str
    title: str
    downloaded_path: Path

# --------------------------- download ---------------------------

def ytdlp_download(url: str, tmp_dir: Path) -> YTDLResult:
    ensure_dir(tmp_dir)
    outtmpl = str(tmp_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp", "-f", "bestaudio/best",
        "-o", outtmpl,
        "--restrict-filenames", "--no-playlist", "--no-warnings",
        url,
    ]
    run(cmd, check=True)
    files = sorted(tmp_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    media = next((p for p in files if p.is_file() and p.suffix not in {".part", ".ytdl"}), None)
    if media is None:
        raise RuntimeError("yt-dlp produced no media file")
    try:
        title = run(["yt-dlp", "--get-title", url], check=False).stdout.strip()
    except Exception:
        title = ""
    return YTDLResult(url=url, video_id=media.stem, title=title, downloaded_path=media)

# --------------------------- ffmpeg ops ---------------------------

def ffmpeg_extract_range(in_path: Path, out_dir: Path, start_s: float, end_s: float) -> Path:
    ensure_dir(out_dir)
    out_path = out_dir / f"range_{int(start_s*1000):06d}_{int(end_s*1000):06d}.wav"
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", f"{start_s}", "-to", f"{end_s}",
        "-i", str(in_path),
        "-ac", str(CHANNELS), "-ar", str(SR),
        "-vn", "-acodec", "pcm_s16le",
        str(out_path),
    ]
    run(cmd, check=True)
    return out_path

def probe_duration_secs(path: Path) -> float:
    try:
        prob = run([
            "ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1", str(path)
        ], check=True)
        return float(prob.stdout.strip())
    except Exception:
        return 0.0

def chop_nonoverlap_no_pad(in_path: Path, out_dir: Path, range_start_s: float) -> List[Path]:
    """
    Make UNIQUE, NON-OVERLAPPING 3s windows. NO padding.
    If range duration < 3s -> 0 clips.
    Number of clips = floor((range_dur) / 3.0).
    Filenames use ABSOLUTE start times (relative to original video).
    """
    ensure_dir(out_dir)
    total = probe_duration_secs(in_path)
    if total < CLIP_SECONDS - 1e-9:
        return []  # nothing to extract if shorter than 3s

    nwin = int(math.floor((total - CLIP_SECONDS) / STRIDE_SECONDS) + 1)
    paths: List[Path] = []

    for i in range(nwin):
        rel_start = i * STRIDE_SECONDS
        rel_end   = rel_start + CLIP_SECONDS
        # Absolute start in the original video timeline:
        abs_start = range_start_s + rel_start
        out_name  = f"t{int(round(abs_start*1000)):06d}ms.wav"
        out_path  = out_dir / out_name

        # Trim exactly [rel_start, rel_end] WITHOUT padding:
        cmd = [
            "ffmpeg","-hide_banner","-nostdin","-y",
            "-i", str(in_path),
            "-af", f"atrim=start={rel_start}:end={rel_end}",
            "-ac", str(CHANNELS), "-ar", str(SR),
            "-vn","-acodec","pcm_s16le",
            str(out_path),
        ]
        run(cmd, check=True)
        paths.append(out_path)

    return paths

# --------------------------- metadata ---------------------------

def write_metadata(csv_path: Path, rows: List[dict]) -> None:
    ensure_dir(csv_path.parent)
    header = not csv_path.exists()
    fields = [
        "label","video_id","title","source_url","samplerate","channels",
        "clip_seconds","start_ms","rel_path","abs_path","duration_sec",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

# --------------------------- main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Simple JSON→IRMAS-like WAV clip harvester")
    ap.add_argument("--json", type=Path, required=True, help="Path to manifest JSON")
    args = ap.parse_args()

    # Tool checks
    if not which("ffmpeg") or not which("ffprobe"):
        raise SystemExit("ffmpeg/ffprobe must be installed and on PATH.")
    if not which("yt-dlp"):
        raise SystemExit("yt-dlp not found. Install with: pip install yt-dlp")

    # Load manifest
    entries: List[Dict[str, Any]] = json.loads(args.json.read_text(encoding="utf-8"))

    rows: List[dict] = []

    for item in entries:
        url   = item.get("video") or item.get("url")
        label = item.get("label", "unknown")
        ranges = item.get("clips", [])
        if not url or not isinstance(ranges, list):
            print(f"[skip] bad entry: {item}")
            continue

        # Download once per URL
        try:
            y = ytdlp_download(url, TMP_DIR)
        except Exception as e:
            print(f"[yt-dlp] failed: {url}: {e}")
            continue

        dest_base = OUT_ROOT / label / y.video_id
        ensure_dir(dest_base)

        for (start_m, end_m) in ranges:
            start_s = float(start_m) * 60.0
            end_s   = float(end_m) * 60.0
            if end_s <= start_s:
                continue
            
            rng_dir = dest_base / f"rng_{int(start_s*1000):06d}_{int(end_s*1000):06d}"
            ensure_dir(rng_dir)

            # Extract annotated range to canonical WAV
            rng_wav = ffmpeg_extract_range(y.downloaded_path, rng_dir, start_s, end_s)

            # Make unique, non-overlapping 3s clips (no padding)
            produced = chop_nonoverlap_no_pad(rng_wav, rng_dir, range_start_s=start_s)

            # Metadata rows
            for p in produced:
                m = re.search(r"t(\d{6})ms", p.name)
                start_ms = int(m.group(1)) if m else 0
                rows.append({
                    "label": label,
                    "video_id": y.video_id,
                    "title": y.title or "",
                    "source_url": url,
                    "samplerate": SR,
                    "channels": CHANNELS,
                    "clip_seconds": CLIP_SECONDS,
                    "start_ms": start_ms,
                    "rel_path": str(p.relative_to(OUT_ROOT)),
                    "abs_path": str(p.resolve()),
                    "duration_sec": f"{CLIP_SECONDS:.3f}",
                })

    if rows:
        write_metadata(METADATA_CSV, rows)
        print(f"INFO: Wrote {len(rows)} clips → {METADATA_CSV}")
    else:
        print("No clips created.")

if __name__ == "__main__":
    main()
