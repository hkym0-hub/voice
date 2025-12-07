# =========================================================
# WaveSketch: Multi-Color Drawing from Sound Waves
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

st.title("🎧 WaveSketch: Multi-Color Sound Drawings")
st.write(
    "Upload a short **WAV or MP3** file. "
    "Your voice becomes a multi-color drawing based on **amplitude, pitch, energy, and rhythm (ZCR)**."
)
st.caption("⚠️ m4a는 서버 환경 문제로 지원되지 않습니다. WAV 또는 MP3를 사용하세요.")


# ---------------------------------------------------------
# Utility
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
        pitches = librosa.yin(y, fmin=80, fmax=1000)
        pitch_mean = float(np.nanmean(pitches))
    except Exception:
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
# COLOR ENGINE
# ---------------------------------------------------------
def get_dynamic_color(amplitude, pitch, energy, zcr):
    amp = np.clip(abs(amplitude), 0, 1)
    v = np.clip(0.2 + amp * 0.8, 0, 1)

    if pitch <= 0:
        pitch_norm = 0.0
    else:
        pitch_norm = np.clip((pitch - 80) / 800, 0, 1)
    h = pitch_norm * 0.9

    energy_norm = np.clip(energy * 40, 0, 1)
    s = np.clip(0.25 + energy_norm * 0.75, 0, 1)

    zcr_norm = np.clip(zcr * 8, 0, 1)
    h = (h + (random.random() - 0.5) * 0.25 * zcr_norm) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))


# ---------------------------------------------------------
# DRAWING STYLES
# ---------------------------------------------------------
def draw_line_art(t, y, feats, complexity, thickness, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35
    n_layers = 1 + complexity

    energy = feats["rms"]
    pitch = feats["pitch"]
    zcr = feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for layer in range(n_layers):
        offset = (layer - (n_layers - 1) / 2) * 0.03
        y_line = base_y + offset
        alpha = 0.35 - layer * 0.03

        for i in range(len(t) - 1):
            color = get_dynamic_color(amp[i], pitch, energy, zcr)
            ax.plot(t[i:i+2], y_line[i:i+2], color=color,
                    linewidth=thickness, alpha=alpha)

    return render_figure_to_bytes(fig)


def draw_scribble_art(t, y, feats, complexity, thickness, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25

    energy = feats["rms"]
    pitch = feats["pitch"]
    zcr = feats["zcr"]

    n_paths = 5 + complexity * 3

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for _ in range(n_paths):
        jitter = np.random.normal(scale=0.02 + energy * 0.05, size=len(base_y))
        y_line = base_y + jitter

        alpha = 0.05 + random.random() * 0.10
        width = thickness * (0.5 + random.random())

        for i in range(len(t) - 1):
            color = get_dynamic_color(amp[i], pitch, energy, zcr)
            ax.plot(t[i:i+2], y_line[i:i+2], color=color,
                    linewidth=width, alpha=alpha)

    return render_figure_to_bytes(fig)


def draw_contour_wave(t, y, feats, complexity, thickness, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    energy = feats["rms"]
    pitch = feats["pitch"]
    zcr = feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    base_r = 0.3 + energy * 0.5
    angles = np.linspace(0, 2 * np.pi, len(amp))

    for layer in range(1, complexity + 3):
        offset = layer * 0.03
        r_line = base_r + amp * 0.25 + offset
        jitter = np.random.normal(scale=0.01 + zcr * 0.2, size=len(r_line))
        r_line = r_line + jitter

        x = r_line * np.cos(angles)
        y2 = r_line * np.sin(angles)

        for i in range(len(x) - 1):
            color = get_dynamic_color(amp[i], pitch, energy, zcr)
            ax.plot(x[i:i+2], y2[i:i+2], color=color,
                    linewidth=thickness * 0.7, alpha=0.7)

    return render_figure_to_bytes(fig)


def draw_particle_drift(t, y, feats, complexity, thickness, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    energy = feats["rms"]
    pitch = feats["pitch"]
    zcr = feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n_particles = 150 * complexity

    for _ in range(n_particles):
        i = random.randint(0, len(amp) - 1)
        x = t[i]
        y_pos = 0.5 + amp[i] * 0.3

        drift_x = x + np.random.normal(scale=0.02 + zcr * 0.1)
        drift_y = y_pos + np.random.normal(scale=0.02 + energy * 0.1)

        size = thickness * np.random.uniform(0.3, 1.2)
        color = get_dynamic_color(amp[i], pitch, energy, zcr)

        ax.scatter(drift_x, drift_y, color=color, s=size * 8, alpha=0.7)

    return render_figure_to_bytes(fig)


def draw_spiral_bloom(t, y, feats, complexity, thickness, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    energy = feats["rms"]
    pitch = feats["pitch"]
    zcr = feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    turns = 3 + complexity * 0.7
    angles = np.linspace(0, turns * 2 * np.pi, len(amp))
    radius = 0.1 + amp * 0.5

    jitter = np.random.normal(scale=0.02 + zcr * 0.1, size=len(radius))
    radius = radius + jitter

    x = radius * np.cos(angles)
    y2 = radius * np.sin(angles)

    for i in range(len(x) - 1):
        color = get_dynamic_color(amp[i], pitch, energy, zcr)
        ax.plot(x[i:i+2], y2[i:i+2], color=color,
                linewidth=thickness * 0.9, alpha=0.8)

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR UI (Emotion Controls 삭제된 버전)
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

drawing_style = st.sidebar.selectbox(
    "Drawing Style",
    ["Line Art", "Scribble Art", "Contour Wave", "Particle Drift", "Spiral Bloom"]
)

complexity = st.sidebar.slider("Complexity", 1, 10, 5)
thickness = st.sidebar.slider("Line / Stroke Thickness", 1, 6, 2)
seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

# --- API Key ---
st.sidebar.header("API Settings (optional)")
api_key = st.sidebar.text_input(
    "AssemblyAI API Key",
    placeholder="Enter your AssemblyAI API key...",
    type="password"
)

if api_key:
    st.sidebar.success("API Key registered ✔")
else:
    st.sidebar.info("API Key not set (emotion auto-detection disabled)")


# ---------------------------------------------------------
# Emotion Transparency Guide (UI 없이 설명만 유지)
# ---------------------------------------------------------
st.markdown("## 🫧 Emotion-Based Transparency Guide")
st.markdown("""
Each emotion influences the **transparency (alpha value)** of the strokes.

Brighter emotions generate stronger and more vivid lines,  
while heavier emotions create softer and more delicate impressions.

### Emotion → Alpha Range Mapping  
- **joy**: *0.8–1.0* — bright and clearly defined  
- **sadness**: *0.2–0.4* — faint, blurred, and fading  
- **anger**: *0.75–1.0* — bold, intense, and dominant  
- **fear**: *0.3–0.6* — trembling, unstable presence  
- **surprise**: *0.5–0.8* — sharp and striking  
- **neutral**: *0.55–0.75* — balanced transparency  

This transparency modulation adds emotional nuance to the artwork,  
making each generated visual feel more expressive and alive.
""")


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")

uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Analyzing audio…"):
        try:
            t, y_ds, feats = analyze_audio(uploaded_file)
        except Exception as e:
            st.error("Audio loading failed. Use WAV or MP3.")
            st.code(str(e))
            st.stop()

    st.subheader("2️⃣ Extracted Audio Features")
    st.json(feats)

    st.subheader("3️⃣ Generated Drawing")

    if drawing_style == "Line Art":
        img_buf = draw_line_art(t, y_ds, feats, complexity, thickness, seed)
    elif drawing_style == "Scribble Art":
        img_buf = draw_scribble_art(t, y_ds, feats, complexity, thickness, seed)
    elif drawing_style == "Contour Wave":
        img_buf = draw_contour_wave(t, y_ds, feats, complexity, thickness, seed)
    elif drawing_style == "Particle Drift":
        img_buf = draw_particle_drift(t, y_ds, feats, complexity, thickness, seed)
    else:
        img_buf = draw_spiral_bloom(t, y_ds, feats, complexity, thickness, seed)

    st.image(
        img_buf,
        caption=f"{drawing_style} – audio-driven multi-color drawing",
        use_container_width=True
    )

    st.download_button(
        "📥 Download Image",
        img_buf,
        file_name="wavesketch.png",
        mime="image/png"
    )

else:
    st.info("Please upload a WAV or MP3 file 🎵")


# ---------------------------------------------------------
# Color Guide
# ---------------------------------------------------------
st.markdown("## 🎨 Color Interpretation Guide")
st.markdown("""
### 🌗 Dark vs Bright Colors
- Quiet parts → darker  
- Loud parts → brighter  

### 🌈 Hue (Cool → Warm)
- Low pitch → blue/green  
- High pitch → orange/pink  

### 🎯 Saturation
- High RMS → vivid colors  
- Low RMS → soft colors  

### 🌀 ZCR
- More consonant/noisy speech → more hue flicker  
""")
