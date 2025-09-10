from pydub import AudioSegment
from pydub.utils import which
import math
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Explicitly tell pydub where ffmpeg and ffprobe are
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Input: can be single file(s) or folder(s)
input_paths = [
    r"E:\qingchaolaopian\Instrument Sound\Original Sound\Suona\Suona with back ground instruments"
 
]

# Output folder
output_folder = r"E:\qingchaolaopian\Instrument Sound\Sound parts\instruments_Suona"

# Processing parameters
segment_seconds = 3          # length of each segment in seconds
target_fs = 44100            # target sample rate
bitrate = "192k"             # mp3 bitrate

# Supported formats
VALID_EXTS = (".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac")


def process_file(input_file, output_folder, segment_seconds, target_fs, bitrate):
    """Split a single file into segments and export as MP3"""
    try:
        audio = AudioSegment.from_file(input_file)

        # Resample to target sample rate
        audio = audio.set_frame_rate(target_fs)

        # Ensure stereo (2 channels)
        if audio.channels == 1:
            audio = AudioSegment.from_mono_audiosegments(audio, audio)
        elif audio.channels > 2:
            mono = audio.set_channels(1)
            audio = AudioSegment.from_mono_audiosegments(mono, mono)
        else:
            audio = audio.set_channels(2)

        # Calculate number of segments
        segment_ms = segment_seconds * 1000
        num_segments = math.ceil(len(audio) / segment_ms)

        # Base file name
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        # Export each segment
        for k in range(num_segments):
            start_ms = k * segment_ms
            end_ms = min((k + 1) * segment_ms, len(audio))
            segment = audio[start_ms:end_ms]

            # Pad with silence if shorter than required
            if len(segment) < segment_ms:
                silence = AudioSegment.silent(duration=segment_ms - len(segment), frame_rate=target_fs)
                segment = segment + silence

            out_name = f"{base_name}_part_{str(k+1).zfill(2)}.mp3"
            out_path = os.path.join(output_folder, out_name)
            segment.export(out_path, format="mp3", bitrate=bitrate)

        return f"✅ Done: {input_file} → {num_segments} segments"
    except Exception as e:
        return f"❌ Error processing {input_file}: {e}"


def expand_inputs(input_paths):
    """Expand input paths into a list of audio files"""
    all_files = []
    for p in input_paths:
        if os.path.isfile(p):
            if p.lower().endswith(VALID_EXTS):
                all_files.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(VALID_EXTS):
                        all_files.append(os.path.join(root, f))
    return all_files


if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    all_files = expand_inputs(input_paths)

    if not all_files:
        print("⚠️ No audio files found.")
    else:
        print(f"Found {len(all_files)} audio files. Starting parallel processing...")

        # Use all available CPU cores
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_file, f, output_folder, segment_seconds, target_fs, bitrate): f for f in all_files}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Overall progress"):
                result = future.result()
                print(result)
