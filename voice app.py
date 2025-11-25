# app.py
# =========================================================
# WaveSketch: Drawing with Sound Waves
# - Streamlit app
# - Audio upload → librosa analysis → generative drawings
# - Styles: line-art / scribble / contour / charcoal
# - Added: Random Word API + Automatic BPM→Color mapping
# =========================================================

import io
import random
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa

# ---------------------------------------------------------
# Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="WaveSketch - Sound Drawings",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 WaveSketch: Drawing with Sound Waves")
st.write(
    "Upload a short voice or sound clip. "
    "The waveform becomes **drawing-style art**, and color changes automatically "
    "based on the **tempo (BPM)** of the audio."
)

# ---------------------------------------------------------
# RANDOM WORD API → DRAWING THEME
# ---------------------------------------------------------
def get_random_theme():
    url = "https://random-word-api.herokuapp.com/word"
    try:
        return requests.get(url).json()[0].lower()
    except:
        return "abstract"


# ---------------------------------------------------------
# UTIL FUNCTIONS
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
# BPM → COLOR
# ---------------------------------------------------------
def get_color_from_bpm(features):
    """
    Converts BPM into an RGB color.
    Slow → purple/blue
    Medium → teal
    Fast → yellow/green
    """
    x = normalize(features["tempo"], 40.0, 180.0)

    r = 0.4 + 0.6 * x
    g = 0.2 + 0.7 * x
    b = 1.0 - 0.7 * x

    return (
        float(np.clip(r, 0, 1)),
        float(np.clip(g, 0, 1)),
        float(np.clip(b, 0, 1)),
    )


# ---------------------------------------------------------
# AUDIO ANALYSIS
# ---------------------------------------------------------
def analyze_audio(file, target_points=1200):
    y, sr = librosa.load(file, sr=None, mono=True)

    if len(y) > 10 * sr:
        y = y[:10 * sr]

    n_samples = len(y)
    idx = np.linspace(0, n_samples - 1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0.0, 1.0, len(y_ds))

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    try:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        pitch_mean = float(np.nanmean(f0))
    except:
        pitch_mean = 0.0

    features = {
        "sr": sr,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "tempo": float(tempo),
        "pitch_mean": pitch_mean,
    }

    return t, y_ds, features


# ---------------------------------------------------------
# DRAWING STYLE FUNCTIONS (+ THEME + COLOR)
# ---------------------------------------------------------
def draw_line_art(t, y, features, complexity, thickness, seed, theme_influence, draw_color):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    jitter_scale = (0.01 + energy_n * 0.03) * theme_influence["jitter_boost"]

    n_layers = 1 + complexity

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i in range(n_layers):
        offset = (i - (n_layers - 1) / 2) * 0.03
        jitter = np.random.normal(scale=jitter_scale, size=len(base_y))
        y_line = base_y + offset + jitter * (i / max(1, n_layers - 1))

        alpha = 0.3 + 0.4 * (1 - i / max(1, n_layers - 1))
        ax.plot(t, y_line, color=draw_color, linewidth=thickness, alpha=alpha)

    return render_figure_to_bytes(fig)


def draw_scribble_art(t, y, features, complexity, thickness, seed, theme_influence, draw_color):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    zcr_n = normalize(features["zcr_mean"], 0, 0.3)

    jitter_base = (0.02 + energy_n * 0.04) * theme_influence["jitter_boost"]
    jagged_factor = (0.01 + (1 - zcr_n) * 0.03) * theme_influence["curve_strength"]

    n_paths = 5 + complexity * 3

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for _ in range(n_paths):
        jitter = np.random.normal(scale=jitter_base, size=len(base_y))
        jagged = np.random.normal(scale=jagged_factor, size=len(base_y))

        y_line = base_y + jitter + jagged * np.sign(np.gradient(base_y))

        ax.plot(
            t,
            y_line,
            linewidth=thickness * (0.5 + np.random.rand()),
            alpha=0.04 + 0.15 * np.random.rand(),
            color=draw_color,
        )

    return render_figure_to_bytes(fig)


def draw_contour_drawing(t, y, features, complexity, thickness, seed, theme_influence, draw_color):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    tempo_n = normalize(features["tempo"], 40, 180)

    base_radius = 0.3 + energy_n * 0.2
    radius = (base_radius + amp * 0.25) * theme_influence["curve_strength"]

    angles = 2 * np.pi * (t * (1.5 + tempo_n * 2.0))
    n_contours = 2 + complexity

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i in range(n_contours):
        offset_r = (i - (n_contours - 1) / 2) * 0.015
        jitter = np.random.normal(scale=0.006, size=len(radius))

        r_line = radius + offset_r + jitter
        x = 0.5 + r_line * np.cos(angles)
        y_line = 0.5 + r_line * np.sin(angles)

        alpha = 0.15 + 0.25 * (1 - i / max(1, n_contours - 1))
        ax.plot(x, y_line, linewidth=thickness, color=draw_color, alpha=alpha)

    return render_figure_to_bytes(fig)


def draw_charcoal_style(t, y, features, complexity, thickness, seed, theme_influence, draw_color):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.2

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    n_strokes = int((80 + complexity * 40) * theme_influence["density_mul"])

    stroke_min_len = int(len(t) * 0.03)
    stroke_max_len = int(len(t) * 0.12)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for _ in range(n_strokes):
        start = random.randint(0, len(t) - stroke_min_len - 1)
        length = random.randint(stroke_min_len, stroke_max_len)
        end = min(len(t) - 1, start + length)

        x_seg = t[start:end]
        y_seg = base_y[start:end] + np.random.normal(
            scale=0.01 + energy_n * 0.03, size=end - start
        )

        ax.plot(
            x_seg,
            y_seg,
            linewidth=thickness * np.random.uniform(0.7, 1.5),
            alpha=0.04 + 0.1 * np.random.rand(),
            color=draw_color,
        )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

drawing_style = st.sidebar.selectbox(
    "Drawing Style",
    ["Line Art", "Scribble Art", "Contour Drawing", "Charcoal / Ink"],
)

complexity = st.sidebar.slider("Complexity", 1, 10, 5)
thickness = st.sidebar.slider("Base Line Thickness", 1, 6, 2)
seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

# ---------------------------------------------------------
# THEME FROM API
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Drawing Theme (via API)")

if st.sidebar.button("Get Random Theme"):
    st.session_state["theme_word"] = get_random_theme()

theme = st.session_state.get("theme_word", None)

if theme:
    st.sidebar.success(f"Theme: **{theme}**")
else:
    st.sidebar.info("No theme yet.")

theme_influence = {"jitter_boost": 1.0, "curve_strength": 1.0, "density_mul": 1.0}

if theme:
    if theme in ["wave", "flow", "ocean", "water"]:
        theme_influence["curve_strength"] = 1.4
    elif theme in ["wind", "air", "breeze", "storm"]:
        theme_influence["jitter_boost"] = 1.6
    elif theme in ["mountain", "ridge", "rock"]:
        theme_influence["curve_strength"] = 0.7
    elif theme in ["shadow", "dark", "night"]:
        theme_influence["density_mul"] = 1.6
    elif theme in ["portrait", "face", "figure"]:
        theme_influence["density_mul"] = 0.8
        theme_influence["curve_strength"] = 1.1


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")

uploaded_file = st.file_uploader(
    "Upload a short sound / voice file (WAV, MP3, M4A)",
    type=["wav", "mp3", "m4a"],
)

if uploaded_file:
    st.audio(uploaded_file)
    uploaded_file.seek(0)

    with st.spinner("Analyzing sound..."):
        t, y_ds, feats = analyze_audio(uploaded_file)

    # COLOR BASED ON BPM
    draw_color = get_color_from_bpm(feats)

    st.subheader("2️⃣ Audio Features")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Sample Rate: `{feats['sr']}`")
        st.write(f"RMS Energy: `{feats['rms_mean']:.5f}`")
        st.write(f"ZCR: `{feats['zcr_mean']:.4f}`")
    with col2:
        st.write(f"Spectral Centroid: `{feats['centroid_mean']:.1f}`")
        st.write(f"Tempo (BPM): `{feats['tempo']:.1f}`")
        st.write(f"Pitch Mean: `{feats['pitch_mean']:.1f}`")

    st.subheader("3️⃣ Generated Drawing")

    if drawing_style == "Line Art":
        img_buf = draw_line_art(t, y_ds, feats, complexity, thickness, seed, theme_influence, draw_color)
    elif drawing_style == "Scribble Art":
        img_buf = draw_scribble_art(t, y_ds, feats, complexity, thickness, seed, theme_influence, draw_color)
    elif drawing_style == "Contour Drawing":
        img_buf = draw_contour_drawing(t, y_ds, feats, complexity, thickness, seed, theme_influence, draw_color)
    else:
        img_buf = draw_charcoal_style(t, y_ds, feats, complexity, thickness, seed, theme_influence, draw_color)

    st.image(img_buf, caption=f"{drawing_style} (Theme: {theme})", use_container_width=True)

    st.subheader("4️⃣ Download")
    st.download_button(
        "📥 Download Drawing",
        img_buf,
        file_name="wavesketch_drawing.png",
        mime="image/png",
    )

else:
    st.info("Upload an audio file to begin 🎨")
