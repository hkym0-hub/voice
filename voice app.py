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
    page_title="WaveSketch - Emotion Thickness + Audio Colors",
    page_icon="🎧",
    layout="wide"
)

# ----------------------- (1) 안내 텍스트 -----------------------
st.title("🎧 WaveSketch: Emotion-Driven Line Thickness + Audio-Driven Colors")
st.write(
    "Upload a short **WAV or MP3** file.\n"
    "**Emotion controls the line thickness**, and **audio features control the colors**."
)
st.caption("⚠️ m4a는 서버환경 문제로 지원되지 않습니다. WAV 또는 MP3를 사용하세요.")

# ---------------------------------------------------------
# Emotion → Line Thickness (B Version)
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

    # 10초 이상이면 자르기
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

    # Brightness = amplitude
    v = 0.3 + amp * 0.7

    # Hue = pitch (low→blue, high→red)
    pitch_norm = np.clip((pitch - 80) / 500, 0, 1)
    h = (0.65 - 0.65 * pitch_norm) % 1.0  # blue→green→yellow→red

    # Saturation = RMS
    s = np.clip(rms * 12, 0.25, 1.0)

    # Noise jitter = ZCR
    h = (h + (random.random() - 0.5) * zcr * 0.2) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))


# ---------------------------------------------------------
# Draw Style – Only Line Art (Stable)
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
# SIDEBAR UI
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

emotion_label = st.sidebar.selectbox(
    "Emotion (Affects Thickness)",
    ["neutral", "joy", "sadness", "anger", "fear", "surprise"]
)
emotion_mul = get_emotion_thickness_multiplier(emotion_label)

seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

# API key (optional)
st.sidebar.header("AssemblyAI API (Optional)")
api_key = st.sidebar.text_input(
    "Enter API Key…",
    placeholder="Enter your API key...",
    type="password"
)

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

img_buf = draw_line_style(t, y_ds, feats, seed, emotion_mul)

st.image(
    img_buf,
    caption=f"Emotion: {emotion_label} / Audio-Based Colors",
    use_container_width=True
)

# ---------------------------------------------------------
# Guides
# ---------------------------------------------------------
st.markdown("## 🧵 Emotion → Line Thickness Guide")
st.markdown("""
Emotion controls **line thickness**:
- Joy → very thick  
- Anger → heaviest  
- Surprise → medium-thick  
- Neutral → standard thickness  
- Fear → thin  
- Sadness → thinnest  
""")

st.markdown("## 🎨 Audio Feature → Color Guide")
st.markdown("""
### Color = Audio  
- **Brightness** → amplitude  
- **Hue (blue→red)** → pitch  
- **Saturation** → RMS (energy)  
- **Color jitter** → ZCR (noise level)  
""")
