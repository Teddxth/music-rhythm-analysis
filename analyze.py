"""Analyse complète d'un morceau : BPM, temps forts/faibles, phrases musicales.

Usage : python analyze.py chemin/vers/musique.wav [--model best_bpm_model.keras] [--plot]
"""
import argparse

from rhythm.beats import analyze_beats
from rhythm.dsp_bpm import estimate_bpm
from rhythm.segmentation import segment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio")
    p.add_argument("--model", help="modèle CNN Keras (sinon estimation DSP hybride)")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()

    if a.model:
        from tensorflow import keras
        from rhythm.cnn_bpm import predict_bpm
        bpm = predict_bpm(keras.models.load_model(a.model), a.audio)
    else:
        bpm = estimate_bpm(a.audio, "hybrid")
    print(f"BPM estimé : {bpm:.1f}")

    beats = analyze_beats(a.audio, bpm)
    print(f"{len(beats['beat_times'])} temps détectés (tempo recalé : {beats['tempo']:.1f})")
    for t, lab in list(zip(beats["beat_times"], beats["labels"]))[:8]:
        print(f"  {t:6.2f} s  {lab}")

    seg = segment(a.audio)
    print("Changements de phrase (s) :", ", ".join(f"{t:.1f}" for t in seg["boundaries_s"]))

    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].imshow(seg["ssm"], origin="lower", cmap="magma"); ax[0].set_title("Matrice d'auto-similarité")
        ax[1].plot(seg["times"], seg["novelty"])
        for b in seg["boundaries_s"]:
            ax[1].axvline(b, color="r", ls="--")
        ax[1].set(title="Courbe de nouveauté", xlabel="temps (s)")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
