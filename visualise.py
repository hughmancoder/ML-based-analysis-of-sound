from __future__ import annotations

from pathlib import Path
from typing import Iterable

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from src.classes import IRMAS_CLASSES, decode_label_bits


def load_npy(path: str | Path) -> np.ndarray:
    """Load a cached numpy array from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mel file not found: {path}")
    return np.load(path)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Return audio as (channels, frames) float array and sample rate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    try:
        audio, sample_rate = sf.read(path, always_2d=True)
        audio = audio.T  # channels x frames
        return audio.astype(np.float32, copy=False), sample_rate
    except RuntimeError as exc:
        # Fallback to librosa for exotic formats that soundfile cannot handle.
        try:
            import librosa
        except ImportError as import_exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "soundfile failed to load audio and librosa is not available"
            ) from import_exc

        data, sample_rate = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data.astype(np.float32, copy=False), sample_rate


def db_to_uint8(img_db: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Map dB-scaled mel data to uint8 [0, 255] for consistent plotting."""
    x = np.clip(img_db, db_min, db_max)
    x = (x - db_min) / (db_max - db_min + 1e-12)
    return (x * 255.0).round().astype(np.uint8)


def make_display_image(
    mel_2c_db: np.ndarray,
    tile: str = "v",
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> np.ndarray:
    """Return a 2D uint8 image from a mel tensor in dB."""
    if mel_2c_db.ndim != 3:
        raise ValueError(f"Expected mel shape (C, F, T); got {mel_2c_db.shape}")

    if mel_2c_db.shape[0] == 1:
        return db_to_uint8(mel_2c_db[0], db_min, db_max)

    if mel_2c_db.shape[0] != 2:
        raise ValueError(f"Expected channel dimension 1 or 2; got {mel_2c_db.shape}")

    left_u8 = db_to_uint8(mel_2c_db[0], db_min, db_max)
    right_u8 = db_to_uint8(mel_2c_db[1], db_min, db_max)
    if tile.lower().startswith("h"):
        return np.concatenate([left_u8, right_u8], axis=1)
    return np.concatenate([left_u8, right_u8], axis=0)


def plot_mel_npy(
    path: str | Path,
    *,
    tile: str = "v",
    db_min: float = -80.0,
    db_max: float = 0.0,
    title: str | None = None,
    single_channel: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Load a saved mel (2, F, T) in dB from .npy and plot it."""
    mel = load_npy(path)
    if single_channel and mel.ndim == 3 and mel.shape[0] > 1:
        mel = mel.mean(axis=0, keepdims=True)
    display = make_display_image(mel, tile=tile, db_min=db_min, db_max=db_max)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(display, aspect="auto", origin="lower")
    ax.set_title(title or f"Mel spectrogram: {Path(path).name}")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins (stacked L/R)")
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_wav_waveform(
    path: str | Path,
    *,
    max_seconds: float | None = None,
    title: str | None = None,
    single_channel: bool = False,
) -> tuple[plt.Figure, Iterable[plt.Axes]]:
    """
    Plot waveform for a wav file. Displays each channel separately unless single_channel is True.

    Parameters
    ----------
    path:
        Path to the wav file on disk.
    max_seconds:
        Optional cap on plotted audio duration.
    title:
        Optional plot title.
    single_channel:
        If True, average all channels to mono before plotting.
    """
    audio, sample_rate = load_audio(path)
    if max_seconds is not None:
        frame_cap = int(max_seconds * sample_rate)
        audio = audio[:, :frame_cap]

    if single_channel and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)

    num_channels, num_frames = audio.shape
    time_axis = np.arange(num_frames) / sample_rate

    if num_channels == 1:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(time_axis, audio[0])
        ax.set_ylabel("Amplitude")
        axes = (ax,)
    else:
        fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3.0 * num_channels), sharex=True)
        axes = np.atleast_1d(axes)
        for idx, channel in enumerate(audio):
            axes[idx].plot(time_axis, channel)
            axes[idx].set_ylabel(f"Ch {idx + 1}")

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title or f"Waveform: {Path(path).name}")
    fig.tight_layout()
    plt.show()
    return fig, axes


def load_manifest(manifest_csv: str | Path) -> pd.DataFrame:
    """
    Load a manifest CSV with consistent label handling.

    Normalises bit-encoded `label_multi` columns and adds a decoded label column.
    """
    df = pd.read_csv(
        manifest_csv,
        dtype={"label_multi": "string"},
        keep_default_na=False,
    )

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()

    if "label_multi" in df.columns:
        df["label_multi"] = (
            df["label_multi"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"[^01]", "", regex=True)
            .str.pad(len(IRMAS_CLASSES), side="right", fillchar="0")
            .str[: len(IRMAS_CLASSES)]
        )
        df["label"] = df["label_multi"].apply(lambda bits: decode_label_bits(bits, IRMAS_CLASSES))

    return df


def choose_one_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """Return a reproducible sample with one row per class label."""
    picked = []
    for label, group in sorted(df.groupby("label"), key=lambda pair: pair[0]):
        picked.append(group.sample(n=1, random_state=1337).iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def plot_first_n_mels(
    manifest_csv: str | Path,
    *,
    n: int = 5,
    tile: str = "v",
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> None:
    """Display the first `n` mel spectrograms listed in a manifest."""
    df = load_manifest(manifest_csv)

    shown = 0
    for _, row in df.iterrows():
        path = row["filepath"]
        if not os.path.exists(path):
            continue
        plot_mel_npy(
            path,
            tile=tile,
            db_min=db_min,
            db_max=db_max,
            title=f"{row.get('irmas_filename', Path(path).name)} — {row.get('label', 'unknown')}",
        )
        shown += 1
        if shown >= n:
            break

    if shown == 0:
        print("No mel files found for the provided manifest.")


def plot_sample_grid(
    samples_df: pd.DataFrame,
    *,
    tile_mode: str = "v",
    cols: int = 4,
) -> None:
    """Plot a grid of mel previews for the provided manifest rows."""
    if samples_df.empty:
        print("No samples to display.")
        return

    n_items = len(samples_df)
    rows = math.ceil(n_items / cols)
    fig_h_per_row = 3.0 if tile_mode.lower().startswith("v") else 2.2

    fig, axes = plt.subplots(rows, cols, figsize=(16, fig_h_per_row * rows))
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= n_items:
            continue

        row = samples_df.iloc[idx]
        mel = load_npy(row["filepath"])
        img = make_display_image(mel, tile=tile_mode)
        ax.imshow(img[::-1, :], aspect="auto", cmap="gray")
        ax.set_title(str(row.get("label", "")), fontsize=11, pad=4)

    plt.tight_layout()
    plt.show()
