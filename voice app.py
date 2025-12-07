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

# ----------------------- (1) 안내 텍스트 -----------------------
st.title("🎧 WaveSketch: Multi-Color Sound Drawings")
st.write(
    "Upload a short **WAV or MP3** file. "
    "Your voice becomes a multi-color drawing based on **amplitude, pitch, energy, and rhythm (ZCR)**."
)
st.caption("⚠️ m4a는 서버환경 문제로 지원되지 않습니다. WAV 또는 MP3를 사용하세요.")


# ---------------------------------------------------------
# Emotion → Line Thickness Mapping
# ---------------------------------------------------------
def get_emotion_thickness_multiplier(emotion):
    # 감정별 차이를 더 극적으로 키운 버전
    table = {
        "joy": 1.8,      # 밝고 두꺼움
        "anger": 2.3,    # 가장 강하고 두꺼움
        "surprise": 1.4, # 살짝 두꺼움
        "neutral": 1.0,  # 기준
        "fear": 0.6,     # 얇고 약함
        "sadness": 0.4   # 가장 얇고 여림
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
    v = 0.2 + amp * 0.8  # brightness

    pitch_norm = np.clip((pitch - 80) / 270, 0, 1)

    if pitch_norm < 0.5:
        h = 0.6 - pitch_norm * 0.6
    else:
        h = 0.3 - (pitch_norm - 0.5) * 0.3

    h = h % 1.0

    energy_norm = np.clip(energy * 15, 0, 1)
    s = 0.2 + energy_norm * 0.8

    zcr_norm = np.clip(zcr * 50, 0, 1)
    h = (h + (random.random() - 0.5) * 0.2 * zcr_norm) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))


# ---------------------------------------------------------
# LINE STYLE (emotion + amplitude 반영)
# ---------------------------------------------------------
def draw_line_art(t, y, feats, complexity, seed, emotion_mul):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)  # -1~1 → -1~1
    base_y = 0.5 + amp * 0.35
    n_layers = 1 + complexity

    energy, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 감정 + 음량에 따라 선 두께가 크게 달라지도록 설계
    base_width = 1.2  # neutral, quiet일 때 최소 두께 기준

    for layer in range(n_layers):
        offset = (layer - (n_layers - 1) / 2) * 0.03
        y_line = base_y + offset
        alpha = max(0.05, 0.35 - layer * 0.03)

        for i in range(len(t) - 1):
            color = get_dynamic_color(amp[i], pitch, energy, zcr)

            # amplitude(0~1) → 1 ~ 4 배
            local_amp = float(np.clip(abs(amp[i]), 0, 1))
            amp_factor = 1.0 + local_amp * 3.0

            # 최종 선 두께 = 기본 * 감정 * 음량
            linewidth = base_width * emotion_mul * amp_factor

            ax.plot(
                t[i:i+2], y_line[i:i+2],
                color=color,
                linewidth=linewidth,
                alpha=alpha
            )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

# 이제 스타일 선택은 없고 복잡도/시드만 조절
complexity = st.sidebar.slider("Complexity (Layer Count)", 1, 10, 5)
seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

# ⭐ 감정 선택 UI
emotion_label = st.sidebar.selectbox(
    "Emotion",
    ["neutral", "joy", "sadness", "anger", "fear", "surprise"]
)
emotion_mul = get_emotion_thickness_multiplier(emotion_label)

# ⭐ API KEY UI
st.sidebar.header("AssemblyAI API")
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
# (2) Upload Audio
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")

uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

if not uploaded_file:
    st.stop()

st.audio(uploaded_file)

with st.spinner("Analyzing audio…"):
    t, y_ds, feats = analyze_audio(uploaded_file)


# ---------------------------------------------------------
# (3) Extracted Audio Features
# ---------------------------------------------------------
st.subheader("2️⃣ Extracted Audio Features")
st.json(feats)


# ---------------------------------------------------------
# (4) Generated Drawing
# ---------------------------------------------------------
st.subheader("3️⃣ Generated Drawing")

img_buf = draw_line_art(t, y_ds, feats, complexity, seed, emotion_mul)

st.image(
    img_buf,
    caption=f"Line Style – audio-driven multi-color drawing ({emotion_label})",
    use_container_width=True
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
