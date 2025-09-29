#!/usr/bin/env python3
"""
Generates IRMAS-style WAV clips from a minimal JSON manifest
"""

from __future__ import annotations
import argparse, json, math, re, shutil, subprocess as sp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------- Config ----------
OUT_ROOT   = Path("data/audio/chinese_instruments")
TMP_DIR    = Path(".cache/video_tmp")
CANON_DIR  = Path(".cache/canonical")
REPO_ROOT  = Path(__file__).resolve().parents[2]
SR         = 44_100
CHANNELS   = 2
CLIP_SEC   = 3.0
STRIDE_SEC = CLIP_SEC

# ---------- utils ----------
def run(cmd: List[str], check=True) -> sp.CompletedProcess:
    return sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding="utf-8", errors="ignore", check=check)



def run_checked(cmd: List[str]) -> sp.CompletedProcess:
    try:
        return run(cmd, check=True)
    except sp.CalledProcessError as e:
        print("---- command failed ----")
        print(" ".join(cmd))
        print(e.output)
        raise

def which(name: str) -> bool:
    return shutil.which(name) is not None

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class MediaSource:
    url_or_file: str
    local_path: Path

def norm_label(label: str) -> str:
    """Normalize label for filename/folders: lower, spaces→underscore."""
    return (label or "unknown").strip().lower().replace(" ", "_")

def parse_time(val) -> float:
    """
    Accepts:
      - 'MM:SS' or 'M:SS.FFF'
      - m.ss  (e.g., 0.18 -> 0m18s, 2.05 -> 2m05s)
      - raw seconds (int/float)
    JSON numeric floats like 0.18 will be interpreted as m.ss if < 60 and have <=2 decimals.
    """
    s = str(val).strip()

    # MM:SS or M:SS.FFF
    if ":" in s:
        mm, ss = s.split(":", 1)
        return int(mm) * 60.0 + float(ss)

    # numeric string / float
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        # decide m.ss vs seconds
        if "." in s:
            mm_str, ss_str = s.split(".", 1)
            # treat as m.ss if < 60 minutes and <= 2 fractional digits
            if len(ss_str) <= 2 and float(s) < 60:
                minutes = int(mm_str) if mm_str else 0
                seconds = int(ss_str.ljust(2, "0")[:2]) if ss_str else 0
                return minutes * 60.0 + seconds * 1.0
        # otherwise plain seconds
        return float(s)

    # fallback
    return float(s)

# ---------- ID allocation ----------
_ID_PATTERN = re.compile(r'^(?:\[[^\]]+\])+(?P<id>\d{4})__\d+\.wav$', re.IGNORECASE)

def next_irmas_id(label_dir: Path) -> int:
    """Scan existing files like [a][b]0001__2.wav and return the next integer (0002→2)."""
    max_id = 0
    for p in label_dir.glob("*.wav"):
        m = _ID_PATTERN.match(p.name)
        if m:
            try:
                max_id = max(max_id, int(m.group("id")))
            except ValueError:
                pass
    return max_id + 1

# ---------- I/O prep ----------
def ytdlp_download(url: str) -> Path:
    ensure_dir(TMP_DIR)
    outtmpl = str(TMP_DIR / "%(id)s.%(ext)s")
    format_candidates = ["bestaudio/best", "bestaudio", "best", None]

    last_error: Optional[sp.CalledProcessError] = None
    for fmt in format_candidates:
        cmd = [
            "yt-dlp",
            "-o",
            outtmpl,
            "--restrict-filenames",
            "--no-playlist",
            "--no-warnings",
        ]
        if fmt:
            cmd.extend(["-f", fmt])
        cmd.append(url)

        try:
            run(cmd, check=True)
            last_error = None
            break
        except sp.CalledProcessError as err:
            last_error = err
            if fmt is None:
                print("---- command failed ----")
                print(" ".join(cmd))
                print(err.output)
            else:
                print(f"[warn] yt-dlp format '{fmt}' failed; trying fallback")

    if last_error:
        raise last_error
    files = sorted(TMP_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    media = next((p for p in files if p.is_file() and p.suffix not in {".part",".ytdl"}), None)
    if media is None:
        raise RuntimeError("yt-dlp produced no media file")
    return media

def _resolve_local_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a manifest 'file' field relative to useful project roots."""

    path = Path(path_str).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            (base_dir / path).expanduser(),
            (REPO_ROOT / path).expanduser(),
        ]

    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists():
            return resolved

    search_list = "\n  - ".join(str(c.resolve(strict=False)) for c in candidates)
    raise FileNotFoundError(f"Local file not found. Checked:\n  - {search_list}")


def prepare_source(entry: Dict[str, Any], base_dir: Path) -> MediaSource:
    if entry.get("file"):
         p = _resolve_local_path(entry["file"], base_dir)
         return MediaSource(url_or_file=str(p), local_path=p)
    url = entry.get("video") or entry.get("url")    
    if not url:
        raise ValueError("Entry needs 'file' or 'video'/'url'.")
    if not which("yt-dlp"):
        raise RuntimeError("yt-dlp is required for 'video' URLs but was not found on PATH.")
    media = ytdlp_download(url)
    return MediaSource(url_or_file=url, local_path=media)

def ensure_canonical_wav(src_path: Path) -> Path:
    """Transcode once to 44.1kHz stereo PCM WAV for accurate slicing."""
    ensure_dir(CANON_DIR)
    out_wav = CANON_DIR / (src_path.stem + ".wav")
    if out_wav.exists():
        return out_wav
    cmd = ["ffmpeg","-hide_banner","-nostdin","-y","-i",str(src_path),
           "-ac",str(CHANNELS),"-ar",str(SR),"-vn","-acodec","pcm_s16le", str(out_wav)]
    run_checked(cmd)
    return out_wav

def probe_duration_secs(path: Path) -> float:
    try:
        prob = run(["ffprobe","-v","error","-show_entries","format=duration",
                    "-of","default=noprint_wrappers=1:nokey=1", str(path)], check=True)
        return float(prob.stdout.strip())
    except Exception:
        return 0.0

# ---------- Cutting ----------
def cut_windows_nonoverlap(in_wav: Path, start_s: float = 0.0, end_s: Optional[float] = None) -> List[tuple]:
    """Return (clip_start, clip_end) within [start_s, end_s), 3s, non-overlap, no padding."""
    total = probe_duration_secs(in_wav)
    if total <= 0:
        return []
    s0 = max(0.0, start_s)
    s1 = min(end_s, total) if end_s is not None else total
    span = s1 - s0
    if span < CLIP_SEC - 1e-9:
        return []
    nwin = int(math.floor((span - CLIP_SEC) / STRIDE_SEC) + 1)
    return [(s0 + i * STRIDE_SEC, s0 + i * STRIDE_SEC + CLIP_SEC) for i in range(nwin)]

def write_clip(in_wav: Path, out_path: Path, clip_start: float, clip_end: float) -> None:
    if out_path.exists():
        return
    cmd = ["ffmpeg","-hide_banner","-nostdin","-y","-i",str(in_wav),
           "-af", f"atrim=start={clip_start}:end={clip_end}",
           "-ac",str(CHANNELS),"-ar",str(SR),"-vn","-acodec","pcm_s16le", str(out_path)]
    run_checked(cmd)

# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build IRMAS-style files with all labels in filename.")
    ap.add_argument("--input", type=Path, required=True, help="Path to JSON manifest")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    if not which("ffmpeg") or not which("ffprobe"):
        raise SystemExit("ffmpeg/ffprobe must be installed and on PATH.")

    manifest_path = args.input.expanduser().resolve()
    entries: List[Dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_created = 0

    for entry in entries:
        labels_raw = entry.get("labels") or []
        if not labels_raw:
            labels_raw = ["unknown"]
        labels_norm = [norm_label(l) for l in labels_raw]

        primary = labels_norm[0]                    # IRMAS convention
        label_dir = OUT_ROOT / primary
        ensure_dir(label_dir)

        # Per-label numeric ID (NNNN)
        nnnn = next_irmas_id(label_dir)

        # Prepare media
        try:
            media = prepare_source(entry, manifest_path.parent)
        except Exception as e:
            print(f"[skip] source error: {e}")
            continue

        canon = ensure_canonical_wav(media.local_path)

        # Build clip schedule
        ranges = entry.get("time_stamps_to_extract", [])
        clip_times: List[tuple] = []
        if ranges:
            print(f"[info] using time_stamps_to_extract with {len(ranges)} ranges: {ranges}")
            for s, e in ranges:
                rs, re = parse_time(s), parse_time(e)
                print(f"  range {rs} to {re} sec")
                if re > rs:
                    clip_times.extend(cut_windows_nonoverlap(canon, rs, re))
        else:
            print("[info] no time_stamps_to_extract given; using full file")
            clip_times.extend(cut_windows_nonoverlap(canon))

        if not clip_times:
            print(f"[info] no 3s windows for {media.local_path}")
            continue

        
        # Filename prefix: [lab1][lab2]...[labN]
        prefix = "".join(f"[{lab}]" for lab in labels_norm)
        src_stem = Path(media.local_path).stem

        # Write files
        for idx, (cs, ce) in enumerate(clip_times, start=1):
            name = f"{prefix}{src_stem}__{idx}.wav"
            out_path = label_dir / name
            if args.overwrite and out_path.exists():
                out_path.unlink(missing_ok=True)
            write_clip(canon, out_path, cs, ce)
            total_created += 1

        print(f"[ok] {len(clip_times)} clips → {label_dir}/{prefix}{src_stem}__*.wav")

    print(f"Done. Created/kept ~{total_created} clips under {OUT_ROOT}")

if __name__ == "__main__":
    main()