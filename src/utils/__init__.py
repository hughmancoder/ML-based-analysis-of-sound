"""Utility helpers for data prep and analysis."""

from .mel_utils import (
    _hash_path,
    load_audio_stereo,
    ensure_duration,
    calc_fft_hop,
    mel_stereo2_from_stereo,
    _load_segment_stereo,
    _stereo_to_mel,
    _safe_relpath,
    MelDataset,
)

from .utils import (
    CLASSES,
    LABEL_TO_IDX_IRMAS,
    IDX_TO_LABEL_IRMAS,
    LABEL_TO_IDX_CN,
    IDX_TO_LABEL_CN,
    load_npy,
    decode_label_bits,
    load_manifest,
)

__all__ = [
    "_hash_path",
    "load_audio_stereo",
    "ensure_duration",
    "calc_fft_hop",
    "mel_stereo2_from_stereo",
    "_load_segment_stereo",
    "_stereo_to_mel",
    "_safe_relpath",
    "MelDataset",
    "CLASSES",
    "LABEL_TO_IDX_IRMAS",
    "IDX_TO_LABEL_IRMAS",
    "LABEL_TO_IDX_CN",
    "IDX_TO_LABEL_CN",
    "load_npy",
    "decode_label_bits",
    "load_manifest",
]
