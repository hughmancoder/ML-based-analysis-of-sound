from pydub import AudioSegment
from pydub.utils import which
import math
import os
from tqdm import tqdm   # <-- 新增

# Explicitly tell pydub where ffmpeg and ffprobe are
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Input and output paths
input_file = r"E:\qingchaolaopian\Instrument Sound\Original Sound\Guzheng\Guzheng_Pure\Bawangxiejia.flac"
output_folder = r"E:\qingchaolaopian\Instrument Sound\Sound parts\Guzheng\Pure Guzheng\Bawangxieji"
segment_seconds = 3          # segment length
target_fs = 44100            # target sample rate (Hz)
bitrate = "192k"             # mp3 bitrate (can be "128k", "192k", "320k")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load audio
audio = AudioSegment.from_file(input_file)

# Resample to 44.1kHz and ensure stereo
audio = audio.set_frame_rate(target_fs)

if audio.channels == 1:
    # Mono → duplicate to stereo
    audio = AudioSegment.from_mono_audiosegments(audio, audio)
elif audio.channels > 2:
    # More than 2 channels → downmix to mono, then duplicate to stereo
    mono = audio.set_channels(1)
    audio = AudioSegment.from_mono_audiosegments(mono, mono)
else:
    # Already stereo
    audio = audio.set_channels(2)

# Calculate total segments
segment_ms = segment_seconds * 1000
num_segments = math.ceil(len(audio) / segment_ms)

# Export each segment with progress bar
base_name = os.path.splitext(os.path.basename(input_file))[0]
for k in tqdm(range(num_segments), desc="Exporting segments"):
    start_ms = k * segment_ms
    end_ms = min((k+1) * segment_ms, len(audio))
    segment = audio[start_ms:end_ms]

    # Pad with silence if shorter than required
    if len(segment) < segment_ms:
        silence = AudioSegment.silent(duration=segment_ms - len(segment), frame_rate=target_fs)
        segment = segment + silence

    # File name with zero-padded index
    out_name = f"{base_name}_part_{str(k+1).zfill(2)}.mp3"
    out_path = os.path.join(output_folder, out_name)

    # Export as MP3
    segment.export(out_path, format="mp3", bitrate=bitrate)

print(f"Done: Exported {num_segments} segments (3s / 44.1kHz / stereo MP3) to {output_folder}")
