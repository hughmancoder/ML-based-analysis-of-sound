from __future__ import annotations
from typing import Tuple, Optional
import hashlib
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf, librosa
from src.classes import IRMAS_CLASSES


def _hash_path(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()[:10]

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()
    
def load_audio_stereo(path: Path, target_sr: int) -> np.ndarray:
    x, sr_in = sf.read(str(path), always_2d=True)  # (T,C)
    x = x.astype(np.float32, copy=False)
    chans = []
    for c in range(x.shape[1]):
        y = librosa.resample(x[:, c], orig_sr=sr_in, target_sr=target_sr) if sr_in != target_sr else x[:, c]
        chans.append(y)
    T = max(len(ch) for ch in chans)
    chans = [np.pad(ch, (0, T-len(ch))) for ch in chans]
    stereo = np.stack(chans, axis=0)  # (C,T)
    if stereo.shape[0] == 1:
        stereo = np.vstack([stereo, stereo])
    elif stereo.shape[0] > 2:
        stereo = stereo[:2]
    return stereo

def precache_one(wav_path: Path, label: str, cache_root: Path,
                 sr: int, dur: float, n_mels: int, win_ms: float, hop_ms: float,
                 fmin: float, fmax: Optional[float]) -> Path:
    n_fft, hop, win_length = calc_fft_hop(sr, win_ms, hop_ms)
    stereo = load_audio_stereo(wav_path, target_sr=sr)
    stereo = ensure_duration(stereo, sr, dur)
    mel = mel_stereo2_from_stereo(
        stereo, sr,
        n_fft=n_fft, hop=hop, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )  # (2, n_mels, T)

    stem = wav_path.stem
    tag  = f"sr{sr}_dur{dur}_m{n_mels}_w{int(win_ms)}_h{int(hop_ms)}"
    fn   = f"{stem}__{_hash_path(str(wav_path))}__{tag}.npy"
    out_path = cache_root / label / fn
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mel.astype(np.float32))
    return out_path

def ensure_duration(stereo: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    C, T = stereo.shape; target = int(round(sr * duration_s))
    if T >= target: return stereo[:, :target]
    return np.pad(stereo, ((0,0),(0, target-T)))

def calc_fft_hop(sr: int, win_ms: float, hop_ms: float) -> Tuple[int,int,int]:
    win_length = int(round(sr * (win_ms/1000.0)))
    hop = int(round(sr * (hop_ms/1000.0)))
    n_fft = _next_pow2(win_length)
    return n_fft, hop, win_length

def mel_stereo2_from_stereo(stereo: np.ndarray, sr: int, n_fft: int, hop: int, win_length: int,
                            n_mels: int, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or (sr/2)
    feats = []
    for ch in (0, 1):
        S = librosa.feature.melspectrogram(y=stereo[ch], sr=sr, n_fft=n_fft,
                                           hop_length=hop, win_length=win_length, window="hann",
                                           n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, center=True)
        S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        feats.append(S_db)
    return np.stack(feats, axis=0)  # (2, n_mels, T)

def _load_segment_stereo(path: Path, target_sr: int, start_s: float, dur_s: float) -> np.ndarray:
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (N,C)
    y = y.T  # (C,N)
    if file_sr != target_sr:
        import librosa
        y = np.vstack([librosa.resample(y[ch], orig_sr=file_sr, target_sr=target_sr) for ch in range(y.shape[0])])
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]

    start = max(0, int(round(start_s * target_sr)))
    L = int(round(dur_s * target_sr))
    if start >= y.shape[1]:
        return np.zeros((2, L), dtype=np.float32)
    seg = y[:, start:start+L]
    if seg.shape[1] < L:
        seg = np.pad(seg, ((0,0),(0, L - seg.shape[1])), mode="constant")
    return seg




def _compute_starts(clip_len_s: float, win_s: float, stride_s: float) -> List[float]:
    starts: List[float] = []
    s, eps = 0.0, 1e-6
    while s + win_s <= clip_len_s + eps:
        starts.append(round(s, 3))
        s += stride_s
    if not starts:
        return [0.0]
    # tail coverage (if last window didn't reach end)
    tail = max(clip_len_s - win_s, 0.0)
    if starts[-1] + win_s < clip_len_s - eps and abs(tail - starts[-1]) > eps:
        starts.append(round(tail, 3))
    return sorted(set(starts))

def _stereo_to_mel(stereo: np.ndarray, sr: int, n_mels: int, win_ms: float, hop_ms: float,
                   fmin: float, fmax: Optional[float]) -> np.ndarray:
    n_fft, hop, win_length = calc_fft_hop(sr, win_ms, hop_ms)
    mel = mel_stereo2_from_stereo(stereo, sr, n_fft=n_fft, hop=hop, win_length=win_length,
                                  n_mels=n_mels, fmin=fmin, fmax=fmax)  # (2, n_mels, T)
    return mel.astype(np.float32)

def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return p.resolve().as_posix()
    
class MelDataset(Dataset):
    """
    Reads RAW-audio manifest (filepath,label). If cache_root is set, saves/loads
    (2, n_mels, T) .npy 
      <cache_root>/<label>/<stem>__<hash>__sr{sr}_dur{dur}_m{M}_w{W}_h{H}.npy
    """
    def __init__(self, manifest_csv: str, label_to_idx: dict,
                 cache_root: str | None = None,
                 sr=44100, duration_s=3.0, n_mels=128, win_ms=30.0, hop_ms=10.0,
                 fmin=20.0, fmax=None):
        self.df = pd.read_csv(manifest_csv)
        self.label_to_idx = label_to_idx
        self.cache_root = Path(cache_root) if cache_root else None

        self.sr = sr; self.duration_s = duration_s
        self.n_mels = n_mels; self.win_ms = win_ms; self.hop_ms = hop_ms
        self.fmin = fmin; self.fmax = fmax if fmax is not None else sr/2
        self.n_fft, self.hop, self.win_length = calc_fft_hop(sr, win_ms, hop_ms)

    def __len__(self): return len(self.df)

    def _cache_path(self, wav_path: Path, label: str):
        if not self.cache_root: return None
        stem = wav_path.stem
        tag  = f"sr{self.sr}_dur{self.duration_s}_m{self.n_mels}_w{int(self.win_ms)}_h{int(self.hop_ms)}"
        fn   = f"{stem}__{_hash_path(str(wav_path))}__{tag}.npy"
        return self.cache_root / label / fn

    def _compute_mel(self, wav_path: Path):
        stereo = load_audio_stereo(wav_path, self.sr)
        stereo = ensure_duration(stereo, self.sr, self.duration_s)
        mel = mel_stereo2_from_stereo(stereo, self.sr,
                                      n_fft=self.n_fft, hop=self.hop, win_length=self.win_length,
                                      n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        return mel.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = Path(row["filepath"])
        label = row["label"].strip().lower()
        y = self.label_to_idx[label]

        cpath = self._cache_path(wav, label)
        if cpath and cpath.exists():
            mel = np.load(cpath)
        else:
            mel = self._compute_mel(wav)
            if cpath:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(cpath, mel)
        return torch.from_numpy(mel), torch.tensor(y, dtype=torch.long)

    
