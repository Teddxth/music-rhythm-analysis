"""Estimation du BPM par traitement du signal (sans apprentissage).

Méthodes : enveloppe passe-bas (grosse caisse), passe-haut (charleston/caisse claire),
HPSS (partie percussive) avec nettoyage des intervalles + autocorrélation,
et une logique hybride avec repli.
"""
import numpy as np
import librosa
from scipy import signal

BPM_MIN, BPM_MAX = 70, 140


def _fold_bpm(bpm, lo=BPM_MIN, hi=BPM_MAX):
    """Ramène un tempo dans [lo, hi] (corrige les erreurs de demi/double tempo)."""
    if bpm is None or bpm <= 0:
        return None
    while bpm > hi:
        bpm /= 2
    while bpm < lo:
        bpm *= 2
    return float(bpm)


def _envelope(y, sr, fc, btype):
    b, a = signal.butter(4, 2 * fc / sr, btype=btype)
    return signal.filtfilt(b, a, np.abs(y))


def _bpm_from_peaks(env, sr, height_factor=1.2, min_gap_s=0.2):
    peaks, _ = signal.find_peaks(env, height=np.mean(env) * height_factor,
                                 distance=int(sr * min_gap_s))
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / sr
    med = np.median(intervals)
    good = intervals[np.abs(intervals - med) < 0.2 * med]   # rejet des intervalles aberrants
    if len(good) >= 2:
        med = np.median(good)
    return 60.0 / med


def _bpm_from_autocorr(env, sr):
    env = env - env.mean()
    # autocorrélation par FFT (O(n log n) au lieu de O(n²))
    n = 1 << (2 * len(env) - 1).bit_length()
    spec = np.fft.rfft(env, n)
    corr = np.fft.irfft(spec * np.conj(spec), n)[: len(env)]
    lag_min, lag_max = int(sr * 60 / BPM_MAX / 2), int(sr * 60 / BPM_MIN * 2)
    lag = np.argmax(corr[lag_min:lag_max]) + lag_min
    return 60.0 * sr / lag


def bpm_lowpass(y, sr, fc=150):
    return _fold_bpm(_bpm_from_peaks(_envelope(y, sr, fc, "low"), sr))


def bpm_highpass(y, sr, fc=3000):
    return _fold_bpm(_bpm_from_peaks(_envelope(y, sr, fc, "high"), sr))


def bpm_hpss(y, sr, fc=180):
    """HPSS -> partie percussive -> enveloppe lissée -> pics + autocorrélation."""
    _, y_perc = librosa.effects.hpss(y)
    env = _envelope(y_perc, sr, fc, "low")
    raw = _bpm_from_peaks(env, sr)
    corr = _bpm_from_autocorr(env, sr)
    if raw is None:
        return _fold_bpm(corr)
    return _fold_bpm(0.7 * _fold_bpm(raw) + 0.3 * _fold_bpm(corr))


def bpm_hybrid(y, sr, tol=0.10):
    """Logique hybride du projet : priorité au passe-bas ; si passe-bas et passe-haut
    divergent de plus de `tol`, on tranche avec la méthode HPSS ; sinon on moyenne."""
    low, high = bpm_lowpass(y, sr), bpm_highpass(y, sr)
    if low is None and high is None:
        return bpm_hpss(y, sr)
    if low is None or high is None:
        return low or high
    if abs(low - high) / low <= tol:
        return (low + high) / 2
    ref = bpm_hpss(y, sr)
    return min((low, high), key=lambda b: abs(b - ref)) if ref else low


def estimate_bpm(path, method="hybrid", duration=30.0):
    y, sr = librosa.load(path, sr=22050, mono=True, duration=duration)
    fn = {"lowpass": bpm_lowpass, "highpass": bpm_highpass,
          "hpss": bpm_hpss, "hybrid": bpm_hybrid}[method]
    return fn(y, sr)
