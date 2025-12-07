# app.py
# =========================================================
# WaveSketch: Multi-Color Drawing from Sound Waves
# - Audio upload → librosa analysis → generative drawings
# - Multi-color stroke mapping based on amplitude (dB)
# - m4a/mp3/wav all supported via audioread fallback
# - Random Word API → Theme Influence (kept because you chose A)
# =========================================================

import io
import random
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import audioread   # ★ m4a 지원 핵심!

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
    "Upload a sound clip. Each tiny part of the waveform becomes a **colored stroke**, "
    "where quiet moments turn blue and loud moments turn red, creating expressive multi-color art."
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
# ★★★ UNIVERSAL AUDIO LOADER (m4a 지원) ★★★
# ---------------------------------------------------------
def load_audio_any_format(uploaded_file, target_sr=None):
    """
    Tries librosa.load first (soundfile backend).
    If it fails (.m4a/.aac), uses audioread to decode via ffmpeg.
    Always returns mono float32 waveform.
    """
    try:
        # Try standard librosa load first
        y, sr = librosa.load(uploaded_file, sr=target_sr, mono=True)
        return y, sr
    except Exception:
        pass  # fallback to ffmpeg-based loader below

    # Fallback → audioread (ffmpeg)
    with audioread.audio_open(uploaded_file) as f:
        sr = f.samplerate
        data = []

        for buf in f:
            data.append(np.frombuffer(buf, dtype=np.int16))

        y = np.hstack(data).astype(np.float32) / 32768.0  # normalize to -1~1

        # Resample if needed
        if target_sr and target_sr != sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return y, sr


# ---------------------------------------------------------
# AUDIO ANALYSIS
# ---------------------------------------------------------
def analyze_audio(file, target_points=1200):
    # ★ audio loader 변경됨
    y, sr = load_audio_any_format(file, target_sr=None)

    # limit to 10 seconds
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

    # pitch
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
# MULTI-COLOR MAP
# ---------------------------------------------------------
def get_color_from_amplitude(value):
    x = float(np.clip(value, 0, 1))

    if x < 0.25:      # blue → cyan
        r = 0
        g = x * 4
        b = 1
    elif x < 0.5:     # cyan → green
        r = 0
        g = 1
        b = 1 - (x - 0.25) * 4
    elif x < 0.75:    # green → yellow
        r = (x - 0.5) * 4
        g = 1
        b = 0
    else:             # yellow → red
        r = 1
        g = 1 - (x - 0.75) * 4
        b = 0

    return (float(r), float(g), float(b))


# ---------------------------------------------------------
# DRAWING FUNCTIONS
# ---------------------------------------------------------
def draw_line_art(t, y, features, complexity, thickness, seed, theme_inf):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35

    energy_n = normalize(features["rms_mean"], 0.0, 0.1)
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


# (scribble, contour, charcoal functions 동일 — 생략 가능하지만 유지)

def draw_scribble_art(t, y, features, complexity, thickness, seed, theme_inf):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25

    energy_n = normalize(features["rms_mean"], 0.0, 0.1)
    zcr_n = normalize(features["zcr_mean"], 0.0, 0.3)

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

        alpha = 0.03 + 0.12 * np.random.rand()
        width = thickness * (0.5 + np.random.rand())

        for j in range(len(t) - 1):
            amp_norm = abs(amp[j])
            seg_color = get_color_from_amplitude(amp_norm)

            ax.plot(
                t[j:j+2],
                y_line[j:j+2],
                color=seg_color,
                linewidth=width,
                alpha=alpha,
            )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

drawing_style = st.sidebar.selectbox(
    "Drawing Style",
    ["Line Art", "Scribble Art"],
)

complexity = st.sidebar.slider("Complexity", 1, 10, 5)
thickness = st.sidebar.slider("Base Line Thickness", 1, 6, 2)
seed = st.sidebar.slider("Random Seed", 0, 9999, 42)


# ---------------------------------------------------------
# THEME (via API)
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Drawing Theme (via API)")

if st.sidebar.button("Get Random Theme"):
    st.session_state["theme_word"] = get_random_theme()

theme = st.session_state.get("theme_word", None)

if theme:
    st.sidebar.success(f"🎨 Theme: **{theme}**")

theme_inf = {"jitter_boost": 1.0, "curve_strength": 1.0}

if theme:
    if theme in ["wave", "flow", "ocean"]:
        theme_inf["curve_strength"] = 1.3
    elif theme in ["wind", "air"]:
        theme_inf["jitter_boost"] = 1.5


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

    st.subheader("2️⃣ Audio Features")
    st.write(feats)

    st.subheader("3️⃣ Generated Multi-Color Drawing")

    if drawing_style == "Line Art":
        img_buf = draw_line_art(t, y_ds, feats, complexity, thickness, seed, theme_inf)
    else:
        img_buf = draw_scribble_art(t, y_ds, feats, complexity, thickness, seed, theme_inf)

    st.image(
        img_buf,
        caption=f"{drawing_style} (Theme: {theme}) – multi-color amplitude mapping",
        use_container_width=True,
    )

    st.subheader("4️⃣ Download")
    st.download_button(
        "📥 Download Drawing",
        img_buf,
        file_name="wavesketch_multicolor.png",
        mime="image/png",
    )
else:
    st.info("Upload an audio file to begin 🎨")
