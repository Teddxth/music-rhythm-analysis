"""Équilibrage du dataset par time-stretching.

Transforme une distribution gaussienne des BPM en distribution uniforme :
plage 70-120 BPM, quota de 10 morceaux par BPM, variation limitée à ±15 BPM
(au-delà, l'étirement dégrade l'attaque des percussions). Chaque fichier généré
est tracé dans un journal CSV.
"""
import csv
import os
from collections import Counter

import librosa
import soundfile as sf


def balance_dataset(tracks, out_dir, bpm_range=(70, 120), quota=10, max_shift=15,
                    duration=30.0, sr=22050):
    """tracks : liste de (chemin_audio, bpm_original, titre)."""
    os.makedirs(out_dir, exist_ok=True)
    counts = Counter(int(round(b)) for _, b, _ in tracks)
    lo, hi = bpm_range
    log_path = os.path.join(out_dir, "augmentation_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["fichier", "source", "bpm_origine", "bpm_cible", "variation"])
        for path, bpm, title in tracks:
            y = None
            for shift in range(-max_shift, max_shift + 1):
                target = int(round(bpm)) + shift
                if shift == 0 or not lo <= target <= hi or counts[target] >= quota:
                    continue
                if y is None:
                    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
                y_st = librosa.effects.time_stretch(y, rate=target / bpm)
                name = f"{target}-{int(round(bpm))}-{title}.wav"
                sf.write(os.path.join(out_dir, name), y_st, sr)
                log.writerow([name, os.path.basename(path), bpm, target, shift])
                counts[target] += 1
    return counts, log_path
