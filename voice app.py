# =========================================================
# WaveSketch: Emotion Colors + Audio-driven Thickness
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
    page_title="WaveSketch - Emotion Colors + Audio Thickness",
    page_icon="🎧",
    layout="wide"
)

# ----------------------- (1) 안내 텍스트 -----------------------
st.title("🎧 WaveSketch: Emotion Colors + Audio-Driven Line Thickness")
st.write(
    "Upload a short **WAV or MP3** file. "
    "Your voice generates a drawing where **emotion controls the colors** "
    "and **sound dynamics control the line thickness**."
)
st.caption("⚠️ m4a는 서버환경 문제로 지원되지 않습니다. WAV 또는 MP3 사용을 권장합니다.")


# ---------------------------------------------------------
# Emotion → Color Palette (Hue ranges)
# ---------------------------------------------------------
def get_emotion_hue_range(emotion):
    """
    감정마다 고유한 색조(hue) 범위를 반환.
    그 범위 안에서 랜덤하게 색을 생성함.
    """
    table = {
        "joy":      (0.10, 0.20),   # yellow → orange
        "sadness":  (0.55, 0.65),   # blue → deep blue
        "anger":    (0.95, 1.00),   # red
        "fear":     (0.68, 0.75),   # purple
        "surprise": (0.30, 0.40),   # green → mint
        "neutral":  (0.00, 1.00),   # full spectrum
    }
    return table.get(emotion, (0.00, 1.00))


# ---------------------------------------------------------
# Utility: Render Matplotlib → streamlit image
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

    if len(y) > 10 * sr:  # 최대 10초까지만 분석
        y = y[:10 * sr]

    idx = np.linspace(0, len(y) - 1, target_points, dtype=int)
    y_ds = y[idx]  # downsample
    t = np.linspace(0, 1, len(y_ds))

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    # 음정(pitch)은 색상 변화에 사용하지 않지만 feature로 출력
    try:
        pitches = librosa.yin(y, fmin=80, fmax=1000)
        pitch_mean = float(np.nanmean(pitches))
    except Exception:
        pitch_mean = 0.0

    features = {
        "sr": sr,
        "rms": float(np.mean(rms)),
        "zcr": float(np.mean(zcr)),
        "pitch": pitch_mean,
    }

    return t, y_ds, features


# ---------------------------------------------------------
# COLOR ENGINE (Emotion → Hue range)
# ---------------------------------------------------------
def get_emotion_color(emotion):
    hue_min, hue_max = get_emotion_hue_range(emotion)
    h = random.uniform(hue_min, hue_max)

    s = random.uniform(0.6, 1.0)  # vivid saturation
    v = random.uniform(0.7, 1.0)  # bright value

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))


# ---------------------------------------------------------
# THICKNESS ENGINE (Audio-driven)
# ---------------------------------------------------------
def compute_line_thickness(amplitude, rms, zcr):
    """
    소리 세기(amplitude, rms, zcr)에 따라 선 굵기 변화.
    감정은 굵기에 영향을 주지 않음.
    """
    amp_factor = abs(amplitude) * 4
    rms_factor = rms * 30
    zcr_factor = zcr * 8

    thickness = 1.0 + amp_factor + rms_factor + zcr_factor
    return max(0.5, thickness)  # 최소 굵기 보장


# ---------------------------------------------------------
# Drawing: Line Style Only
# ---------------------------------------------------------
def draw_line_style(t, y, feats, emotion, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)

    rms = feats["rms"]
    zcr = feats["zcr"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    base_y = 0.5 + amp * 0.35  # 파형 변환
    n_layers = 8  # 고정된 레이어 수

    for layer in range(n_layers):
        offset = (layer - (n_layers - 1) / 2) * 0.03
        y_line = base_y + offset

        for i in range(len(t) - 1):
            color = get_emotion_color(emotion)

            lw = compute_line_thickness(
                amplitude=amp[i],
                rms=rms,
                zcr=zcr
            )

            ax.plot(
                t[i:i+2], y_line[i:i+2],
                color=color,
                linewidth=lw,
                alpha=0.7,
            )

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

emotion_label = st.sidebar.selectbox(
    "Emotion (Affects Colors)",
    ["neutral", "joy", "sadness", "anger", "fear", "surprise"]
)

seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

# AssemblyAI API 미사용이지만 입력창 유지
st.sidebar.header("AssemblyAI API (Optional)")
api_key = st.sidebar.text_input(
    "AssemblyAI API Key",
    placeholder="Enter your API key...",
    type="password"
)


# ---------------------------------------------------------
# 1️⃣ Upload Audio
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")
uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

if not uploaded_file:
    st.stop()

st.audio(uploaded_file)

t, y_ds, feats = analyze_audio(uploaded_file)


# ---------------------------------------------------------
# 2️⃣ Extracted Audio Features
# ---------------------------------------------------------
st.subheader("2️⃣ Extracted Audio Features")
st.json(feats)


# ---------------------------------------------------------
# 3️⃣ Generated Drawing
# ---------------------------------------------------------
st.subheader("3️⃣ Generated Drawing")

img_buf = draw_line_style(t, y_ds, feats, emotion_label, seed)

st.image(
    img_buf,
    caption=f"Line Style – Emotion Colors + Audio Thickness",
    use_container_width=True
)


# ---------------------------------------------------------
# 4️⃣ Emotion → Color Guide
# ---------------------------------------------------------
st.markdown("## 🎨 Emotion-Based Color Guide")
st.markdown("""
Each emotion generates colors from a **unique hue range**, giving each drawing a distinct emotional tone.

### Emotion → Color Mapping  
- **joy** → Yellow / Orange spectrum  
- **sadness** → Blue / Deep blue  
- **anger** → Red / Crimson  
- **fear** → Purple / Dark violet  
- **surprise** → Green / Mint  
- **neutral** → All colors (full spectrum, softer saturation)

Emotion affects **only color**, not thickness.
""")


# ---------------------------------------------------------
# 5️⃣ Audio → Line Thickness Guide
# ---------------------------------------------------------
st.markdown("## 🧵 Audio-Based Line Thickness Guide")
st.markdown("""
The **thickness** of each line segment is determined by audio dynamics:

### Thickness Factors  
- **Amplitude** (momentary volume) → stronger → thicker  
- **RMS Energy** (overall loudness) → higher → thicker  
- **ZCR** (noisiness/consonants) → higher → slightly thicker  

So the artwork visually reflects how **intense or calm** your voice was.
""")
