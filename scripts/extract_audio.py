# Use to audio from films
"""
#!/usr/bin/env python3
import argparse, pathlib, ffmpeg

def extract_audio(video_path, out_dir, sr=None):
    vp = pathlib.Path(video_path)
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (vp.stem + ".wav")
    stream = ffmpeg.input(str(vp))
    if sr:
        stream = ffmpeg.output(stream.audio, str(out_path), ac=2, ar=sr)
    else:
        stream = ffmpeg.output(stream.audio, str(out_path), ac=2)
    ffmpeg.run(stream, overwrite_output=True)
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--out_dir", default="data/interim")
    ap.add_argument("--sr", type=int, default=None, help="Resample rate (e.g., 22050)")
    args = ap.parse_args()
    out = extract_audio(args.video, args.out_dir, args.sr)
    print(out)
"""