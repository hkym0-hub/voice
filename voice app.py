# app.py
# =========================================================
# WaveSketch: Multi-Color Drawing from Sound Waves
# - WAV / MP3 입력
# - Color Theme + amplitude/pitch/energy/ZCR 기반 색상 변조
# - Line Art / Scribble Art
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
    "Your voice becomes a generative multi-color drawing based on amplitude, pitch, energy, and rhythm."
)
st.caption("⚠️ m4a는 서버 환경 문제로 지원되지 않습니다. WAV 또는 MP3를 사용해 주세요.")

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
# Color Theme Base Palettes
# ---------------------------------------------------------
THEME_BASE = {
    "Pastel":      (0.55, 0.45, 0.90),
    "Neon":        (0.10, 0.95, 0.85),
    "Ink":         (0.05, 0.05, 0.10),
    "Fire":        (0.95, 0.40, 0.10),
    "Ocean":       (0.10, 0.40, 0.80),
}

def hsv_to_rgb_tuple(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))

# ---------------------------------------------------------
# AUDIO ANALYSIS
# ---------------------------------------------------------
def analyze_audio(uploaded_file, target_points=1200):

    uploaded_file.seek(0)
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)

    # 10초 제한
    if len(y) > 10 * sr:
        y = y[:10 * sr]

    # Downsample waveform for drawing
    idx = np.linspace(0, len(y)-1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0, 1, len(y_ds))

    # Features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    try:
        pitches = librosa.yin(y, fmin=80, fmax=1000)
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
# FULL COLOR ENGINE (예시 2 완전 적용)
# ---------------------------------------------------------
def get_dynamic_color(amplitude, pitch, energy, zcr, theme_name):
    
    # 기본 테마 색 (RGB)
    r0, g0, b0 = THEME_BASE[theme_name]
    
    # RGB → HSV 변환
    h, s, v = colorsys.rgb_to_hsv(r0, g0, b0)

    # amplitude → lightness 변화
    amp_norm = float(np.clip(abs(amplitude), 0, 1))
    v = np.clip(v * (0.6 + amp_norm * 0.8), 0, 1)

    # pitch → hue shift
    pitch_norm = np.clip(pitch / 700, 0, 1)
    h = (h + pitch_norm * 0.25) % 1.0

    # energy → saturation 조절
    energy_norm = np.clip(energy * 12, 0, 1)
    s = np.clip(s * (0.5 + energy_norm * 0.7), 0, 1)

    # ZCR → jitter (노이즈 색 변화)
    jitter = (random.random() - 0.5) * (zcr * 0.3)
    h = (h + jitter) % 1.0

    return hsv_to_rgb_tuple(h, s, v)

# ---------------------------------------------------------
# DRAWINGS
# ---------------------------------------------------------
def draw_line_art(t, y, feats, complexity, thickness, seed, theme_name):
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
        offset = (layer - (n_layers - 1)/2) * 0.03
        y_line = base_y + offset

        alpha = 0.35 - layer * 0.03

        for i in range(len(t)-1):
            color = get_dynamic_color(
                amplitude=amp[i],
                pitch=pitch,
                energy=energy,
                zcr=zcr,
                theme_name=theme_name
            )

            ax.plot(
                t[i:i+2],
                y_line[i:i+2],
                color=color,
                linewidth=thickness,
                alpha=alpha,
            )

    return render_figure_to_bytes(fig)


def draw_scribble_art(t, y, feats, complexity, thickness, seed, theme_name):
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

        for i in range(len(t)-1):
            color = get_dynamic_color(
                amplitude=amp[i],
                pitch=pitch,
                energy=energy,
                zcr=zcr,
                theme_name=theme_name
            )

            ax.plot(
                t[i:i+2],
                y_line[i:i+2],
                color=color,
                linewidth=width,
                alpha=alpha,
            )

    return render_figure_to_bytes(fig)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

drawing_style = st.sidebar.selectbox("Drawing Style", ["Line Art", "Scribble Art"])

theme_name = st.sidebar.selectbox(
    "Color Theme",
    ["Pastel", "Neon", "Ink", "Fire", "Ocean"]
)

complexity = st.sidebar.slider("Complexity", 1, 10, 5)
thickness = st.sidebar.slider("Line Thickness", 1, 6, 2)
seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")

uploaded_file = st.file_uploader(
    "Upload WAV or MP3",
    type=["wav", "mp3"]
)

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Analyzing audio features…"):
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
        img_buf = draw_line_art(t, y_ds, feats, complexity, thickness, seed, theme_name)
    else:
        img_buf = draw_scribble_art(t, y_ds, feats, complexity, thickness, seed, theme_name)

    st.image(img_buf, caption=f"{drawing_style} with {theme_name} theme", use_container_width=True)

    st.download_button(
        "📥 Download Image",
        img_buf,
        file_name="wavesketch.png",
        mime="image/png"
    )
else:
    st.info("Please upload a WAV or MP3 file 🎵")
