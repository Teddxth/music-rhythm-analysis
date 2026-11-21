"""Détection des temps et alternance pairs / impairs.

On impose à librosa le BPM déjà connu (réseau ou DSP) : librosa ne choisit plus
le tempo global (source d'erreurs de demi-tempo) et sert uniquement à caler
précisément chaque battement sur les pics d'énergie.
"""
import numpy as np
import librosa


def track_beats(y, sr, bpm=None, tightness=400):
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    kwargs = {"start_bpm": bpm, "tightness": tightness} if bpm else {}
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, **kwargs)
    times = librosa.frames_to_time(frames, sr=sr)
    return float(np.atleast_1d(tempo)[0]), times, onset[frames]


def label_even_odd(y, sr, beat_times):
    """Attribue la parité des temps. En tempo binaire, la grosse caisse (graves)
    marque un temps sur deux : on choisit la phase dont les temps ont le plus
    d'énergie basse fréquence comme temps 'forts'."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 150].sum(axis=0)
    idx = np.clip(librosa.time_to_frames(beat_times, sr=sr, hop_length=512), 0, len(low) - 1)
    e = low[idx]
    phase = 0 if e[0::2].mean() >= e[1::2].mean() else 1
    return ["fort" if (i - phase) % 2 == 0 else "faible" for i in range(len(beat_times))]


def analyze_beats(path, bpm=None):
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, times, _ = track_beats(y, sr, bpm)
    return {"tempo": tempo, "beat_times": times, "labels": label_even_odd(y, sr, times)}
