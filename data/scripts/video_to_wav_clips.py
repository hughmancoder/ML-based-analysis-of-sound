#!/usr/bin/env python3
"""
 Download videos and cut IRMAS-like WAV clips from a JSON manifest, works with video url or absolute path to local video file.

Hardcoded IRMAS spec:
- Stereo (2 ch), 44_100 Hz
- 3.0 s clip length (non-overlapping, NO padding)
- Output: data/instruments/<label>/<video_id>/rng_<start>_<end>/tXXXXXXms.wav
- Metadata CSV: data/instruments/metadata.csv

Usage:
    python data/scripts/video_to_wav_clips.py --json data/manifests/manifest_data.json

Flags:
     --overwrite: overwrite audio and rewrite metadata rows

Manifest examples (single + multi-label):
[
  { "video": "https://www.youtube.com/watch?v=mL2r6E1E7sM",
    "label": "gong",
    "clips": [[0.18, 2.05]]           # m.ss format => 18s to 2m05s
  },
  { "file": "/abs/path/to/local_audio.mp3",
    "labels": ["gong","strings"], "primary": "gong",
    "task": "multilabel", "clips": []
  }
]
"""

from __future__ import annotations
import argparse, csv, json, math, re, shutil, subprocess as sp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Set

# ---- Hardcoded constants ----
OUT_ROOT = Path("data/chinese_instruments")
TMP_DIR  = Path(".cache/video_tmp")
CANON_DIR = Path(".cache/canonical")
METADATA_CSV = Path("data/chinese_instruments/metadata.csv")
SR = 44_100
CHANNELS = 2
CLIP_SECONDS = 3.0
STRIDE_SECONDS = CLIP_SECONDS   # no overlap

# --------------------------- utils ---------------------------

def run(cmd: List[str], check: bool = True) -> sp.CompletedProcess:
    return sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, check=check)

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
    video_id: str
    title: str

def parse_min_dot_ss(val) -> float:
    """
    Parse times like 0.18 -> 18s, 2.05 -> 2m05s, 0.03 -> 3s.
    Also accepts 'MM:SS' (or 'M:SS.FFF') and raw seconds.
    Returns seconds as float.
    """
    if isinstance(val, (int, float)) and ":" not in str(val):
        if isinstance(val, float) and val < 60 and "." in f"{val}":
            pass  # treat via string as m.ss
        else:
            return float(val)

    s = str(val).strip()

    if ":" in s:  # "MM:SS" or "M:SS.FFF"
        mm, ss = s.split(":", 1)
        return int(mm) * 60.0 + float(ss)

    if "." in s:  # m.ss where .ss are seconds
        mm_str, ss_str = s.split(".", 1)
        minutes = int(mm_str) if mm_str else 0
        sec_two = ss_str[:2] if len(ss_str) >= 1 else "0"
        frac_tail = ss_str[2:]
        seconds = int(sec_two) if sec_two else 0
        frac = float("0." + frac_tail) if frac_tail else 0.0
        return minutes * 60.0 + seconds + frac

    return float(s)

# --------------------------- download / open ---------------------------

def ytdlp_download(url: str, tmp_dir: Path) -> MediaSource:
    ensure_dir(tmp_dir)
    outtmpl = str(tmp_dir / "%(id)s.%(ext)s")
    cmd = ["yt-dlp","-f","bestaudio/best","-o",outtmpl,"--restrict-filenames","--no-playlist","--no-warnings",url]
    run_checked(cmd)
    files = sorted(tmp_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    media = next((p for p in files if p.is_file() and p.suffix not in {".part",".ytdl"}), None)
    if media is None:
        raise RuntimeError("yt-dlp produced no media file")
    try:
        title = run(["yt-dlp","--get-title",url], check=False).stdout.strip()
    except Exception:
        title = ""
    return MediaSource(url_or_file=url, local_path=media, video_id=media.stem, title=title or media.stem)

def open_local_file(file_path: str) -> MediaSource:
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local file not found: {p}")
    return MediaSource(url_or_file=str(p), local_path=p, video_id=p.stem, title=p.stem)

def get_media_source(entry: Dict[str, Any]) -> MediaSource:
    if entry.get("file"):
        return open_local_file(entry["file"])
    url = entry.get("video") or entry.get("url")
    if not url:
        raise ValueError("Entry must contain either 'file' or 'video'/'url'.")
    return ytdlp_download(url, TMP_DIR)

# --------------------------- canonical WAV cache ---------------------------

def ensure_canonical_wav(src_path: Path, video_id: str) -> Path:
    """Transcode source once to 44.1kHz stereo PCM for fast, repeatable trimming."""
    ensure_dir(CANON_DIR)
    out_wav = CANON_DIR / f"{video_id}.wav"
    if out_wav.exists():
        return out_wav
    cmd = ["ffmpeg","-hide_banner","-nostdin","-y","-i",str(src_path),
           "-ac",str(CHANNELS),"-ar",str(SR),"-vn","-acodec","pcm_s16le",str(out_wav)]
    run_checked(cmd)
    return out_wav

# --------------------------- ffmpeg ops ---------------------------

def probe_duration_secs(path: Path) -> float:
    try:
        prob = run(["ffprobe","-v","error","-show_entries","format=duration",
                    "-of","default=noprint_wrappers=1:nokey=1", str(path)], check=True)
        return float(prob.stdout.strip())
    except Exception:
        return 0.0

def ffmpeg_extract_range(in_path: Path, out_dir: Path, start_s: float, end_s: float) -> Path:
    """Extract a sub-range from the canonical WAV; clamp to file duration."""
    ensure_dir(out_dir)
    total = probe_duration_secs(in_path)
    if total <= 0:
        raise RuntimeError(f"Cannot probe duration for {in_path}")
    # clamp
    start = max(0.0, min(start_s, max(0.0, total - 1e-3)))
    end   = max(start, min(end_s, total))
    if end - start <= 1e-6:
        raise ValueError(f"Non-positive duration after clamping: start={start_s}, end={end_s}, total={total}")

    out_path = out_dir / f"range_{int(start*1000):06d}_{int(end*1000):06d}.wav"
    if out_path.exists():
        return out_path

    # For WAV, use accurate seek: -ss AFTER -i with -to absolute
    cmd = ["ffmpeg","-hide_banner","-nostdin","-y",
           "-i",str(in_path), "-ss", f"{start:.6f}", "-to", f"{end:.6f}",
           "-ac",str(CHANNELS),"-ar",str(SR),"-vn","-acodec","pcm_s16le", str(out_path)]
    run_checked(cmd)
    return out_path

def chop_nonoverlap_no_pad(in_path: Path, out_dir: Path, range_start_s: float) -> List[Path]:
    """Make unique, non-overlapping 3s windows from a (possibly extracted) range file. NO padding."""
    ensure_dir(out_dir)
    total = probe_duration_secs(in_path)
    if total < CLIP_SECONDS - 1e-9:
        return []
    nwin = int(math.floor((total - CLIP_SECONDS) / STRIDE_SECONDS) + 1)
    paths: List[Path] = []
    for i in range(nwin):
        rel_start = i * STRIDE_SECONDS
        rel_end   = rel_start + CLIP_SECONDS
        abs_start = range_start_s + rel_start
        out_name  = f"t{int(round(abs_start*1000)):06d}ms.wav"
        out_path  = out_dir / out_name
        if out_path.exists():
            paths.append(out_path); continue
        cmd = ["ffmpeg","-hide_banner","-nostdin","-y","-i",str(in_path),
               "-af", f"atrim=start={rel_start}:end={rel_end}",
               "-ac",str(CHANNELS),"-ar",str(SR),"-vn","-acodec","pcm_s16le", str(out_path)]
        run_checked(cmd)
        paths.append(out_path)
    return paths

def segment_fullfile_fast(in_wav: Path, out_dir: Path) -> List[Path]:
    """Cut 3s non-overlapping clips from full file in one decode via segment muxer."""
    ensure_dir(out_dir)
    existing = sorted(out_dir.glob("t*.wav"))
    if existing:
        return existing
    seg_pat = out_dir / "seg_%05d.wav"
    cmd = ["ffmpeg","-hide_banner","-nostdin","-y","-i",str(in_wav),
           "-f","segment","-segment_time", f"{CLIP_SECONDS}",
           "-reset_timestamps","1","-acodec","pcm_s16le", str(seg_pat)]
    run_checked(cmd)
    produced: List[Path] = []
    for idx, seg in enumerate(sorted(out_dir.glob("seg_*.wav"))):
        start_ms = int(round(idx * CLIP_SECONDS * 1000))
        new_name = out_dir / f"t{start_ms:06d}ms.wav"
        if new_name.exists():
            seg.unlink(missing_ok=True); produced.append(new_name)
        else:
            seg.rename(new_name); produced.append(new_name)
    return produced

# --------------------------- metadata ---------------------------

CSV_FIELDS = [
    "label","labels","task",
    "video_id","title","source","samplerate","channels",
    "clip_seconds","start_ms","rel_path","duration_sec",
]

def load_existing_rel_paths(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()
    seen = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rp = row.get("rel_path")
            if rp:
                seen.add(rp)
    return seen

def write_metadata(csv_path: Path, rows: List[dict], overwrite: bool) -> None:
    ensure_dir(csv_path.parent)
    write_header = overwrite or (not csv_path.exists())
    mode = "w" if overwrite else "a"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            for k in CSV_FIELDS:
                r.setdefault(k, "")
            w.writerow(r)

# --------------------------- main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Simple JSON→IRMAS-like WAV clip harvester")
    ap.add_argument("--json", type=Path, required=True, help="Path to manifest JSON")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite audio and rewrite metadata")
    args = ap.parse_args()

    # Tool checks
    if not which("ffmpeg") or not which("ffprobe"):
        raise SystemExit("ffmpeg/ffprobe must be installed and on PATH.")
    if not which("yt-dlp"):
        print("[info] yt-dlp not found. OK for local 'file' entries, but required for 'video' URLs.")

    # Existing metadata entries to avoid duplicates when not overwriting
    existing_rel: Set[str] = set()
    if not args.overwrite and METADATA_CSV.exists():
        existing_rel = load_existing_rel_paths(METADATA_CSV)

    entries: List[Dict[str, Any]] = json.loads(args.json.read_text(encoding="utf-8"))
    rows: List[dict] = []

    for item in entries:
        # Multi-label: keep 'label' (primary) for IRMAS-compat; 'labels' is semicolon-joined multi-labels
        labels_list = item.get("labels")
        primary = item.get("primary")
        
        task = item.get("task", "")

        if not labels_list:
            lab = item.get("label", "unknown")
            labels_list = [lab]
        if not primary:
            primary = labels_list[0]

        # Prepare media & canonical WAV
        try:
            media = get_media_source(item)
        except Exception as e:
            print(f"[skip] cannot prepare media for entry {item}: {e}")
            continue

        canon_wav = ensure_canonical_wav(media.local_path, media.video_id)

        dest_base = OUT_ROOT / primary / media.video_id
        ensure_dir(dest_base)

        ranges_min = item.get("clips", [])
        if not isinstance(ranges_min, list):
            print(f"[skip] bad 'clips' field (not a list): {item}")
            continue

        if len(ranges_min) == 0:
            # Full file → segment once
            produced = segment_fullfile_fast(canon_wav, dest_base)
            for p in produced:
                rel_path = str(p.relative_to(OUT_ROOT).as_posix())
                if (not args.overwrite) and p.exists() and rel_path in existing_rel:
                    continue
                m = re.search(r"t(\d{6})ms", p.name)
                start_ms = int(m.group(1)) if m else 0
                rows.append({
                    "label": primary,
                    "labels": ";".join(labels_list),
                    "task": task,
                    "video_id": media.video_id,
                    "title": media.title or "",
                    "source": media.url_or_file,
                    "samplerate": SR,
                    "channels": CHANNELS,
                    "clip_seconds": f"{CLIP_SECONDS:.3f}",
                    "start_ms": start_ms,
                    "rel_path": rel_path,
                    "duration_sec": f"{CLIP_SECONDS:.3f}",
                })
            continue

        # Annotated ranges (m.ss -> seconds)
        for (start_mark, end_mark) in ranges_min:
            start_s = parse_min_dot_ss(start_mark)
            end_s   = parse_min_dot_ss(end_mark)

            rng_dir = dest_base / f"rng_{int(start_s*1000):06d}_{int(end_s*1000):06d}"
            ensure_dir(rng_dir)

            try:
                rng_wav = ffmpeg_extract_range(canon_wav, rng_dir, start_s, end_s)
            except Exception as e:
                print(f"[skip] cannot extract range {start_s:.3f}-{end_s:.3f}s for {media.video_id}: {e}")
                continue

            produced = chop_nonoverlap_no_pad(rng_wav, rng_dir, range_start_s=start_s)

            for p in produced:
                rel_path = str(p.relative_to(OUT_ROOT).as_posix())
                if (not args.overwrite) and p.exists() and rel_path in existing_rel:
                    continue
                m = re.search(r"t(\d{6})ms", p.name)
                start_ms = int(m.group(1)) if m else 0
                rows.append({
                    "label": primary,
                    "labels": ";".join(labels_list),
                    "task": task,
                    "video_id": media.video_id,
                    "title": media.title or "",
                    "source": media.url_or_file,
                    "samplerate": SR,
                    "channels": CHANNELS,
                    "clip_seconds": f"{CLIP_SECONDS:.3f}",
                    "start_ms": start_ms,
                    "rel_path": rel_path,
                    "duration_sec": f"{CLIP_SECONDS:.3f}",
                })

    if rows:
        write_metadata(METADATA_CSV, rows, overwrite=args.overwrite)
        print(f"INFO: Wrote {len(rows)} rows → {METADATA_CSV}")
    else:
        print("No clips created.")

if __name__ == "__main__":
    main()
