# app.py
# =========================================================
# WaveSketch: Multi-Color Drawing from Sound Waves
# - Audio upload → librosa analysis → generative drawings
# - Multi-color stroke mapping based on amplitude
# - m4a/mp3/wav ALL supported via temp-file + audioread fallback
# - Random Word API → Theme Influence (Option A)
# =========================================================

import io
import random
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import audioread
import tempfile

# ---------------------------------------------------------
# Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="WaveSketch - Multi-Color Sound Drawings",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 WaveSketch: Multi-Color Sound Drawings")
st.write(
    "Upload a sound clip. The waveform becomes a **multi-colored drawing**, "
    "where quiet moments turn blue and loud moments turn red."
)

# ---------------------------------------------------------
# RANDOM WORD API → DRAWING THEME
# ---------------------------------------------------------
def get_random_theme():
    try:
        return requests.get("https://random-word-api.herokuapp.com/word").json()[0].lower()
    except:
        return "abstract"


# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------
def normalize(value, min_val, max_val):
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-8), 0, 1))

def render_figure_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------
# UNIVERSAL AUDIO LOADER (m4a 지원)
# ---------------------------------------------------------
def load_audio_any_format(uploaded_file, target_sr=None):
    """
    Streamlit UploadedFile → temp file → safe load for all audio formats.
    """
    # Store to temporary file because audioread requires a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Try librosa (soundfile backend)
    try:
        y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        return y, sr
    except Exception:
        pass

    # Fallback to audioread (FFmpeg)
    try:
        with audioread.audio_open(tmp_path) as f:
            sr = f.samplerate
            data = []
            for buf in f:
                data.append(np.frombuffer(buf, dtype=np.int16))

            y = np.hstack(data).astype(np.float32) / 32768.0

            if target_sr and target_sr != sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            return y, sr

    except Exception as e:
        raise RuntimeError(f"Audio loading failed: {e}")


# ---------------------------------------------------------
# AUDIO ANALYSIS
# ---------------------------------------------------------
def analyze_audio(uploaded_file, target_points=1200):
    uploaded_file.seek(0)  # Important
    y, sr = load_audio_any_format(uploaded_file, target_sr=None)

    # limit to 10s
    if len(y) > 10 * sr:
        y = y[:10 * sr]

    n = len(y)
    idx = np.linspace(0, n - 1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0, 1, len(y_ds))

    # features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    try:
        pitch_vals = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        pitch_mean = float(np.nanmean(pitch_vals))
    except:
        pitch_mean = 0.0

    features = {
        "sr": sr,
        "rms_mean": float(np.mean(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "tempo": float(tempo),
        "pitch_mean": pitch_mean,
    }

    return t, y_ds, features


# ---------------------------------------------------------
# MULTI-COLOR MAPPING (quiet→blue, loud→red)
# ---------------------------------------------------------
def get_color_from_amplitude(value):
    x = float(np.clip(value, 0, 1))

    if x < 0.25:
        return (0, x * 4, 1)  # blue → cyan
    elif x < 0.5:
        return (0, 1, 1 - (x - 0.25) * 4)  # cyan → green
    elif x < 0.75:
        return ((x - 0.5) * 4, 1, 0)  # green → yellow
    else:
        return (1, 1 - (x - 0.75) * 4, 0)  # yellow → red


# ---------------------------------------------------------
# DRAWINGS
# ---------------------------------------------------------
def draw_line_art(t, y, features, complexity, thickness, seed, theme_inf):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    jitter_scale = (0.01 + energy_n * 0.03) * theme_inf["jitter_boost"]

    n_layers = 1 + complexity

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i in range(n_layers):
        offset = (i - (n_layers - 1) / 2) * 0.03
        jitter = np.random.normal(scale=jitter_scale, size=len(base_y))
        y_line = base_y + offset + jitter * (i / max(1, n_layers - 1))

        alpha = 0.25 + 0.35 * (1 - i / max(1, n_layers - 1))

        for j in range(len(t) - 1):
            amp_norm = abs(amp[j])
            seg_color = get_color_from_amplitude(amp_norm)

            ax.plot(
                t[j:j+2],
                y_line[j:j+2],
                color=seg_color,
                linewidth=thickness,
                alpha=alpha,
            )

    return render_figure_to_bytes(fig)


def draw_scribble_art(t, y, features, complexity, thickness, seed, theme_inf):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    zcr_n = normalize(features["zcr_mean"], 0, 0.3)

    jitter_base = (0.02 + energy_n * 0.04) * theme_inf["jitter_boost"]
    jagged_factor = (0.01 + (1 - zcr_n) * 0.03) * theme_inf["curve_strength"]

    n_paths = 5 + complexity * 3

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for _ in range(n_paths):
        jitter = np.random.normal(scale=jitter_base, size=len(base_y))
        jagged = np.random.normal(scale=jagged_factor, size=len(base_y))

        y_line = base_y + jitter + jagged * np.sign(np.gradient(base_y))

        alpha = 0.04 + 0.12 * np.random.rand()
        width = thickness * (0.5 + np.random.rand())

        for j in range(len(t) - 1):
            amp_norm = abs(amp[j])
            s
