# Music Rhythm Analysis: BPM, beats & musical phrases

Automatic analysis of a song's rhythmic structure, combining **audio signal processing** and **deep learning**:

1. **Tempo (BPM)**: hybrid DSP estimator (low-pass / high-pass / HPSS) and a CNN regressor on mel-spectrograms
2. **Strong / weak beats**: librosa beat tracking *constrained* by the predicted BPM
3. **Musical phrases** (intro, verse, chorus…): unsupervised segmentation with a Self-Similarity Matrix

> Projet d'école d'ingénieurs (Arts et Métiers, 2025-2026) : outil pédagogique d'analyse de la construction rythmique d'une musique, pour aider les danseurs d'improvisation à repérer les changements de phrases musicales et les variations de tempo. Description en français plus bas.

## Context

The project started as a **teaching tool for improvisation dancers**. To improvise well, a dancer has to feel where a musical phrase starts and ends and when the tempo changes, which is hard to learn because music is difficult to structure: the same song mixes changing tempos, breaks and phrases of different lengths.

Classical signal-processing algorithms were not enough to anticipate and analyse these structures reliably across genres, so we moved to **AI-based approaches** (a CNN for the tempo, combined with beat tracking and self-similarity analysis for the phrases). The goal: during their training sessions, improvisation dancers can see and learn the **phrase changes** and **tempo variations** of the music they dance to.

---

## Pipeline

```
audio ──► BPM (DSP hybride ou CNN) ──► beat tracking contraint ──► temps forts / faibles
   └────► chroma + MFCC ──► matrice d'auto-similarité ──► courbe de nouveauté ──► frontières de phrases
```

| Module | Contenu |
|---|---|
| `rhythm/dsp_bpm.py` | Enveloppes passe-bas / passe-haut / HPSS, rejet des intervalles aberrants, autocorrélation FFT, logique hybride avec repli |
| `rhythm/augmentation.py` | Équilibrage du dataset par **time-stretching** (70-120 BPM, quota 10/BPM, ±15 BPM max, journal CSV de traçabilité) |
| `rhythm/cnn_bpm.py` | CNN Keras sur mel-spectrogrammes, fenêtres glissantes, Dropout, EarlyStopping, évaluation vs baseline |
| `rhythm/beats.py` | `beat_track` recalé sur le BPM connu + attribution fort/faible par énergie basse fréquence |
| `rhythm/segmentation.py` | SSM cosinus, noyau en damier gaussien, détection de pics (prominence + distance min.) |
| `notebooks/` | Notebooks d'exploration originaux (Colab) |

## Quick start

```bash
pip install -r requirements.txt
python analyze.py my_song.wav --plot              # BPM par DSP hybride
python analyze.py my_song.wav --model best_bpm_model.keras
```

## Results

| Approach | Data | Result |
|---|---|---|
| Hybrid DSP (low-pass + high-pass, fallback) | reference tracks (70-140 BPM, mostly pop/electro) | **±5 BPM** on ~95 % of tracks* |
| CNN v1 (mel-spectrogram regression) | [GiantSteps Tempo](https://github.com/GiantSteps/giantsteps-tempo-dataset) | MAE **20.5 BPM** (first baseline) |
| Phrase segmentation (SSM) | e.g. *Let It Be* | boundaries within **1-10 s** of the real changes |

\* During the prototyping phase, the fallback rule was tuned on tracks whose BPM was known in advance, so the score is a proof of concept, not a blind benchmark.

**Comparison with existing tools.** We benchmarked our BPM estimates against online BPM detectors (Tunebat, Soundplate BPM Finder, MusicExtend BPM Finder): our results were nearly identical, with models trained on far less data.

**What I learned from the CNN v1:** the dataset was heavily concentrated around 120-140 BPM, so the network learned to predict values near the mean. This led to the time-stretching balancing step (Gaussian → uniform distribution) and to the v2 training code (sliding windows, dropout, early stopping, per-track evaluation against a "predict the mean" baseline).

## Key ideas

- **Hybrid tempo + beat tracking.** The network is good at the global tempo but not at locating each beat; librosa is the opposite. Forcing `beat_track` to start from the predicted BPM gives a beat grid that lands on every kick.
- **Data first.** Balancing the BPM distribution mattered more than tweaking layers.
- **Knowing when not to use deep learning.** Phrase segmentation has no reliable ground truth and huge inter-genre variability → an unsupervised SSM method is lighter, needs no training and runs on a smartphone.

## Limitations & next steps

Binary-meter music only (70-140 BPM) · phrase boundaries ±10 s · next: pitch-shift / noise augmentation, CRNN for longer context, real-time interface.

---

### En bref

Projet réalisé en équipe de 4 élèves-ingénieurs (Arts et Métiers, encadré par R. Pile, Q. Consigny et S. Thiery). J'ai développé les algorithmes de détection du BPM par traitement du signal, le réseau de neurones et le pipeline de données.

Comparés aux détecteurs de BPM en ligne (Tunebat, Soundplate, MusicExtend), nos résultats étaient quasi identiques, avec un entraînement sur beaucoup moins de données.

Ce dépôt est une version restructurée et complétée du code du projet.

## Author

**Teddy Thamrong**, Élève-ingénieur Arts et Métiers (expertise robotique & IA) · recherche un stage de fin d'études en IA à partir de février ou de septembre 2027.

License: MIT
