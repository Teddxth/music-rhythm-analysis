"""Régression du BPM par CNN sur mel-spectrogrammes (Keras).

Améliorations par rapport à la première version : fenêtres glissantes avec
recouvrement, Dropout, EarlyStopping / ReduceLROnPlateau, sauvegarde du meilleur
modèle, et évaluation comparée à une baseline naïve (prédire la moyenne).
"""
import numpy as np
import librosa
from tensorflow import keras
from tensorflow.keras import layers

SR, N_MELS, HOP = 22050, 128, 512
WIN_S, STEP_S = 6.0, 3.0          # fenêtre de 6 s, pas de 3 s (50 % de recouvrement)


def mel_windows(path, duration=30.0):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=duration)
    win, step = int(WIN_S * SR), int(STEP_S * SR)
    out = []
    for start in range(0, max(1, len(y) - win + 1), step):
        seg = y[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        m = librosa.power_to_db(librosa.feature.melspectrogram(
            y=seg, sr=SR, n_mels=N_MELS, n_fft=2048, hop_length=HOP), ref=np.max)
        m = (m - m.min()) / (m.max() - m.min() + 1e-9)
        out.append(m[..., None])
    return np.stack(out)


def build_dataset(items):
    """items : liste de (chemin, bpm). Renvoie X, y et l'id du morceau (pour un split sans fuite)."""
    X, y, groups = [], [], []
    for i, (path, bpm) in enumerate(items):
        w = mel_windows(path)
        X.append(w); y += [bpm] * len(w); groups += [i] * len(w)
    return np.concatenate(X), np.array(y, dtype="float32"), np.array(groups)


def build_model(input_shape):
    m = keras.Sequential([keras.Input(shape=input_shape)])
    for f in (16, 32, 64, 128):
        m.add(layers.Conv2D(f, 3, padding="same", activation="relu"))
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D(2))
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dropout(0.3))
    m.add(layers.Dense(64, activation="relu"))
    m.add(layers.Dropout(0.3))
    m.add(layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mae", metrics=["mse"])
    return m


def train(X_tr, y_tr, X_val, y_val, out="best_bpm_model.keras", epochs=100):
    model = build_model(X_tr.shape[1:])
    cbs = [keras.callbacks.ModelCheckpoint(out, save_best_only=True),
           keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)]
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=16, callbacks=cbs)
    return model, hist


def accuracy_tolerance(y_true, y_pred, tol=0.05):
    """Part des prédictions à ±tol (relatif) du BPM réel."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_pred - y_true) <= tol * y_true))


def evaluate(model, X, y, groups):
    """Agrège les fenêtres par morceau (médiane) et compare à la baseline 'moyenne'."""
    pw = model.predict(X, verbose=0).ravel()
    ids = np.unique(groups)
    pred = np.array([np.median(pw[groups == g]) for g in ids])
    true = np.array([y[groups == g][0] for g in ids])
    base = np.full_like(true, true.mean())
    return {"mae": float(np.mean(np.abs(pred - true))),
            "acc_5pct": accuracy_tolerance(true, pred, 0.05),
            "acc_10pct": accuracy_tolerance(true, pred, 0.10),
            "baseline_mae": float(np.mean(np.abs(base - true)))}


def predict_bpm(model, path):
    return float(np.median(model.predict(mel_windows(path), verbose=0)))
