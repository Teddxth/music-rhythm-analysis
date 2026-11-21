"""Segmentation en phrases musicales par Matrice d'Auto-Similarité (SSM).

Méthode non supervisée (aucun entraînement) :
1. fusion chroma (harmonie) + MFCC (timbre), synchronisés sur les temps ;
2. SSM par similarité cosinus ;
3. convolution d'un noyau en damier gaussien le long de la diagonale -> courbe de nouveauté ;
4. détection de pics avec prominence et distance minimale (15-20 s) -> frontières.
"""
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import median_filter


def features(y, sr, hop=512):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    norm = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    F = np.vstack([norm(chroma), norm(mfcc)])
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    Fs = librosa.util.sync(F, beats, aggregate=np.median)
    times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_max=F.shape[1]), sr=sr, hop_length=hop)
    return Fs, times[: Fs.shape[1]]


def self_similarity(F):
    X = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
    S = X.T @ X
    return median_filter(S, size=(1, 7))   # lisse le long des diagonales


def checkerboard_kernel(L):
    g = signal.windows.gaussian(2 * L, std=L / 2)
    G = np.outer(g, g)
    sign = np.ones((2 * L, 2 * L)); sign[:L, L:] = sign[L:, :L] = -1
    return G * sign


def novelty_curve(S, L=16):
    K = checkerboard_kernel(L)
    P = np.pad(S, L, mode="constant")
    nov = np.array([(P[i:i + 2 * L, i:i + 2 * L] * K).sum() for i in range(S.shape[0])])
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def segment(path, min_gap_s=15.0, prominence=0.1, kernel=16):
    y, sr = librosa.load(path, sr=22050, mono=True)
    F, times = features(y, sr)
    S = self_similarity(F)
    nov = novelty_curve(S, kernel)
    beat_period = np.median(np.diff(times)) if len(times) > 1 else 0.5
    peaks, _ = signal.find_peaks(nov, prominence=prominence,
                                 distance=max(1, int(min_gap_s / beat_period)))
    return {"boundaries_s": times[peaks], "novelty": nov, "times": times, "ssm": S}
