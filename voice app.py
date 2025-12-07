# app.py
# =========================================================
# WaveSketch: Multi-Color + Emotion Reactive Sound Drawings
# - WAV / MP3 입력
# - amplitude / pitch / energy / ZCR 기반 색상 변조
# - emotion intensity 기반 alpha 조절
# - Drawing Styles:
#   Line Art / Scribble Art / Contour Wave / Particle Drift / Spiral Bloom
# =========================================================

import io
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import colorsys

# ---------------------------------------------------------
# Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="WaveSketch - Multi-Color Sound Drawings",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 WaveSketch: Multi-Color Emotion Reactive Sound Drawings")
st.write(
    "Upload a short **WAV or MP3** file.\n"
    "Your voice becomes a multi-color drawing based on **amplitude, pitch, energy, ZCR**, "
    "and **emotion intensity controls transparency**."
)
st.caption("⚠️ m4a는 환경 문제로 지원되지 않습니다. WAV 또는 MP3를 사용하세요.")

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def render_figure_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------------------------------------------------
# AUDIO ANALYSIS
# ---------------------------------------------------------
def analyze_audio(uploaded_file, target_points=1200):
    uploaded_file.seek(0)
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)

    if len(y) > 10 * sr:
        y = y[:10 * sr]

    idx = np.linspace(0, len(y) - 1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0, 1, len(y_ds))

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    try:
        pitches = librosa.yin(y, fmin=80, fmax=1200)
        pitch_mean = float(np.nanmean(pitches))
    except:
        pitch_mean = 0.0

    features = {
        "sr": sr,
        "rms": float(np.mean(rms)),
        "zcr": float(np.mean(zcr)),
        "centroid": float(np.mean(centroid)),
        "tempo": float(tempo),
        "pitch": pitch_mean
    }
    return t, y_ds, features

# ---------------------------------------------------------
# 🌈 COLOR ENGINE (Amplitude / Pitch / RMS / ZCR 기반)
# ---------------------------------------------------------
def get_dynamic_color(amplitude, pitch, energy, zcr):
    amp = np.clip(abs(amplitude), 0, 1)
    v = np.clip(0.2 + amp * 0.8, 0, 1)

    if pitch <= 0:
        pitch_norm = 0.0
    else:
        pitch_norm = np.clip((pitch - 80) / 800, 0, 1)

    h = pitch_norm * 0.9

    energy_norm = np.clip(energy * 45, 0, 1)
    s = np.clip(0.25 + energy_norm * 0.75, 0, 1)

    zcr_norm = np.clip(zcr * 8, 0, 1)
    h = (h + (random.random() - 0.5) * 0.25 * zcr_norm) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))

# ---------------------------------------------------------
# 🎭 Emotion → Transparency (alpha) 조절
# ---------------------------------------------------------
def emotion_intensity_alpha(emotion: str, confidence: float) -> float:
    emotion = (emotion or "neutral").lower()

    ranges = {
        "joy": (0.8, 1.0),
        "sadness": (0.2, 0.45),
        "anger": (0.75, 1.0),
        "fear": (0.35, 0.65),
        "surprise": (0.55, 0.85),
        "neutral": (0.55, 0.75),
    }

    low, high = ranges.get(emotion, ranges["neutral"])
    confidence = np.clip(confidence, 0, 1)
    return float(low + (high - low) * confidence)

# ---------------------------------------------------------
# DRAWING STYLES
# ---------------------------------------------------------
def draw_line_art(t, y, feats, complexity, thickness, seed, emotion_label, emotion_conf):
    random.seed(seed); np.random.seed(seed)
    emotion_alpha = emotion_intensity_alpha(emotion_label, emotion_conf)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35
    n_layers = 1 + complexity

    energy, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for layer in range(n_layers):
        offset = (layer - (n_layers-1)/2) * 0.03
        y_line = base_y + offset

        base_alpha = 0.35 - layer * 0.03
        alpha = base_alpha * emotion_alpha

        for i in range(len(t)-1):
            c = get_dynamic_color(amp[i], pitch, energy, zcr)
            ax.plot(t[i:i+2], y_line[i:i+2], color=c, linewidth=thickness, alpha=alpha)

    return render_figure_to_bytes(fig)


def draw_scribble_art(t, y, feats, complexity, thickness, seed, emotion_label, emotion_conf):
    random.seed(seed); np.random.seed(seed)
    emotion_alpha = emotion_intensity_alpha(emotion_label, emotion_conf)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25
    n_paths = 5 + complexity * 3

    energy, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for _ in range(n_paths):
        jitter = np.random.normal(scale=0.02 + energy * 0.05, size=len(base_y))
        y_line = base_y + jitter

        base_alpha = 0.07 + random.random() * 0.08
        alpha = base_alpha * emotion_alpha

        width = thickness * (0.5 + random.random())

        for i in range(len(t)-1):
            c = get_dynamic_color(amp[i], pitch, energy, zcr)
            ax.plot(t[i:i+2], y_line[i:i+2], color=c, linewidth=width, alpha=alpha)

    return render_figure_to_bytes(fig)


def draw_contour_wave(t, y, feats, complexity, thickness, seed, emotion_label, emotion_conf):
    random.seed(seed); np.random.seed(seed)
    emotion_alpha = emotion_intensity_alpha(emotion_label, emotion_conf)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    energy, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

    base_r = 0.3 + energy * 0.5
    angles = np.linspace(0, 2*np.pi, len(amp))

    for layer in range(1, complexity+3):
        offset = layer * 0.03
        r = base_r + amp * 0.25 + offset
        r += np.random.normal(scale=0.01 + zcr * 0.2, size=len(r))

        x = r * np.cos(angles)
        y2 = r * np.sin(angles)

        for i in range(len(x)-1):
            c = get_dynamic_color(amp[i], pitch, energy, zcr)
            ax.plot(x[i:i+2], y2[i:i+2], color=c,
                    linewidth=thickness*0.8,
                    alpha=0.7 * emotion_alpha)

    return render_figure_to_bytes(fig)


def draw_particle_drift(t, y, feats, complexity, thickness, seed, emotion_label, emotion_conf):
    random.seed(seed); np.random.seed(seed)
    emotion_alpha = emotion_intensity_alpha(emotion_label, emotion_conf)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    energy, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n_particles = 150 * complexity

    for _ in range(n_particles):
        i = random.randint(0, len(amp)-1)
        x = t[i]; y_pos = 0.5 + amp[i] * 0.3

