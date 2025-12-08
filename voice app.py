# =========================================================
# WaveSketch (B-Version): Emotion = Thickness / Audio = Color
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
    page_title="WaveSketch - Multi Drawing Styles",
    page_icon="🎧",
    layout="wide"
)

# ----------------------- (1) 안내 텍스트 -----------------------
st.title("🎧 WaveSketch: Emotion-Driven Line Thickness + Audio-Driven Colors")
st.write(
    "Upload a short **WAV or MP3** file.\n"
    "**Emotion controls line thickness**, and **audio features control the colors**.\n\n"
    "⚠️ **This app requires an AssemblyAI API Key to start.**"
)
st.caption("m4a는 서버환경 문제로 지원되지 않습니다. WAV 또는 MP3를 사용하세요.")

# ---------------------------------------------------------
# Emotion → Line Thickness
# ---------------------------------------------------------
def get_emotion_thickness_multiplier(emotion):
    table = {
        "joy": 2.0,
        "anger": 2.4,
        "surprise": 1.6,
        "neutral": 1.0,
        "fear": 0.7,
        "sadness": 0.5
    }
    return table.get(emotion, 1.0)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def render_figure_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------
# AUDIO ANALYSIS
# ---------------------------------------------------------
def analyze_audio(uploaded_file, target_points=1400):
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

    try:
        pitch = librosa.yin(y, fmin=80, fmax=800)
        pitch_mean = float(np.nanmean(pitch))
    except:
        pitch_mean = 150.0

    features = {
        "rms": float(np.mean(rms)),
        "zcr": float(np.mean(zcr)),
        "centroid": float(np.mean(centroid)),
        "pitch": pitch_mean,
    }

    return t, y_ds, features


# ---------------------------------------------------------
# COLOR ENGINE → Audio controls color
# ---------------------------------------------------------
def get_audio_color(amplitude, pitch, rms, zcr):
    amp = np.clip(abs(amplitude), 0, 1)

    v = 0.3 + amp * 0.7
    pitch_norm = np.clip((pitch - 80) / 500, 0, 1)
    h = (0.65 - 0.65 * pitch_norm) % 1.0

    s = np.clip(rms * 12, 0.25, 1.0)
    h = (h + (random.random() - 0.5) * zcr * 0.2) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))


# ---------------------------------------------------------
# (A) Line Art Style (Original)
# ---------------------------------------------------------
def draw_line_style(t, y, feats, seed, emotion_mul):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)

    base_y = 0.5 + amp * 0.35
    rms, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i in range(len(t) - 1):
        color = get_audio_color(amp[i], pitch, rms, zcr)

        ax.plot(
            t[i:i+2],
            base_y[i:i+2],
            color=color,
            linewidth=1.4 * emotion_mul,
            alpha=0.9
        )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# (B) Ribbon Flow Style
# ---------------------------------------------------------
def draw_ribbon_style(t, y, feats, seed, emotion_mul):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25

    rms, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")

    for offset in np.linspace(-0.04, 0.04, 8):
        for i in range(len(t) - 1):
            color = get_audio_color(amp[i], pitch, rms, zcr)
            ax.plot(
                t[i:i+2],
                (base_y[i:i+2] + offset),
                color=color,
                linewidth=0.8 * emotion_mul,
                alpha=0.6
            )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# (C) Scatter Glow Style — 업그레이드 점 크기 반응
# ---------------------------------------------------------
def draw_scatter_glow_style(t, y, feats, seed, emotion_mul):
    random.seed(seed)
    np.random.seed(seed)

    amp = np.abs(y) / (np.max(np.abs(y)) + 1e-8)
    rms, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")

    glow_size = (amp * 80 * emotion_mul) + (rms * 300) + 5

    for i in range(len(t)):
        color = get_audio_color(amp[i], pitch, rms, zcr)

        ax.scatter(
            t[i],
            0.5 + amp[i] * 0.3,
            s=glow_size[i],
            color=color,
            alpha=0.75
        )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# (D) Wave Gradient Style
# ---------------------------------------------------------
def draw_wave_gradient_style(t, y, feats, seed, emotion_mul):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    rms, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")

    for layer in np.linspace(0.2, 1, 18):
        alpha = 0.05 + 0.03 * layer
        offset = (layer - 0.6) * 0.15

        for i in range(len(t) - 1):
            color = get_audio_color(amp[i], pitch, rms, zcr)
            ax.plot(
                t[i:i+2],
                0.5 + amp[i:i+2] * layer * 0.2 + offset,
                color=color,
                linewidth=emotion_mul * 0.3,
                alpha=alpha
            )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

draw_style = st.sidebar.selectbox(
    "Drawing Style",
    ["Line Art", "Ribbon Flow", "Scatter Glow", "Wave Gradient"]
)

emotion_label = st.sidebar.selectbox(
    "Emotion (Affects Thickness)",
    ["neutral", "joy", "sadness", "anger", "fear", "surprise"]
)
emotion_mul = get_emotion_thickness_multiplier(emotion_label)

seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

st.sidebar.header("AssemblyAI API Key (Required)")
api_key = st.sidebar.text_input(
    "Enter API Key…",
    placeholder="Enter your API key...",
    type="password"
)

# ---------------------------------------------------------
# ❗ API Key 없으면 앱 실행 중단
# ---------------------------------------------------------
if not api_key:
    st.error("🚫 AssemblyAI API Key is required to run WaveSketch.")
    st.stop()

# ---------------------------------------------------------
# Upload Audio
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")
uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

if not uploaded_file:
    st.stop()

st.audio(uploaded_file)

with st.spinner("Analyzing audio…"):
    t, y_ds, feats = analyze_audio(uploaded_file)

# ---------------------------------------------------------
# Extracted Features
# ---------------------------------------------------------
st.subheader("2️⃣ Extracted Audio Features")
st.json(feats)

# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------
st.subheader("3️⃣ Generated Drawing")

if draw_style == "Line Art":
    img_buf = draw_line_style(t, y_ds, feats, seed, emotion_mul)
elif draw_style == "Ribbon Flow":
    img_buf = draw_ribbon_style(t, y_ds, feats, seed, emotion_mul)
elif draw_style == "Scatter Glow":
    img_buf = draw_scatter_glow_style(t, y_ds, feats, seed, emotion_mul)
elif draw_style == "Wave Gradient":
    img_buf = draw_wave_gradient_style(t, y_ds, feats, seed, emotion_mul)

st.image(
    img_buf,
    caption=f"Style: {draw_style} — Emotion: {emotion_label}",
    use_container_width=True
)

# ⭐⭐ DOWNLOAD BUTTON ⭐⭐
st.download_button(
    label="⬇️ Download Image",
    data=img_buf,
    file_name="WaveSketch.png",
    mime="image/png"
)


# ---------------------------------------------------------
# (5) 🧵 Emotion-Based Line Thickness Guide
# ---------------------------------------------------------
st.markdown("## 🧵 Emotion-Based Line Thickness Guide")
st.markdown("""
Each emotion influences the **thickness of the lines** in the artwork.

### Emotion → Thickness Mapping  
- **joy** → much thicker, lively lines (~1.8×)  
- **anger** → the strongest and thickest strokes (~2.3×)  
- **surprise** → slightly thicker and sharper lines (~1.4×)  
- **neutral** → standard thickness (1.0×)  
- **fear** → thinner, more fragile lines (~0.6×)  
- **sadness** → the thinnest and most delicate strokes (~0.4×)  

On top of this, **louder moments** in your voice make lines locally thicker,
while quieter parts stay almost thread-like.
""")


# ---------------------------------------------------------
# (6) 🎨 Color Interpretation Guide
# ---------------------------------------------------------
st.markdown("## 🎨 Color Interpretation Guide")
st.markdown("""
### 🌗 Dark vs Bright Colors
Quiet parts → darker  
Loud parts → brighter  

### 🌈 Hue (Cool → Warm)
Low pitch → blue  
Mid pitch → green/yellow  
High pitch → orange/red  

### 🎯 Saturation
High RMS → vivid colors  
Low RMS → soft pastel  

### 🌀 ZCR
Noisy sections → color jitter  
""")
