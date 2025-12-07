# app.py
# =========================================================
# WaveSketch: Multi-Color Drawing from Sound Waves
# - WAV / MP3 입력
# - amplitude / pitch / energy / ZCR 기반 색상 변조
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
    """
    - WAV/MP3 로드
    - 최대 10초까지만 사용
    - 드로잉용으로 waveform 다운샘플링
    - RMS, ZCR, Spectral Centroid, Tempo, Pitch 추출
    """
    uploaded_file.seek(0)
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)

    # 10초 제한
    if len(y) > 10 * sr:
        y = y[:10 * sr]

    # Downsample waveform for drawing
    idx = np.linspace(0, len(y) - 1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0, 1, len(y_ds))

    # Features (전체 신호 기준)
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
# 🌈 FULL COLOR ENGINE (Theme 제거 버전)
# amplitude / pitch / energy / ZCR 만으로 색 결정
# ---------------------------------------------------------
def get_dynamic_color(amplitude, pitch, energy, zcr):
    """
    amplitude → Value(밝기)
    pitch → Hue(색상)
    energy(RMS) → Saturation(채도)
    ZCR → Hue jitter(색 흔들림, 노이즈)
    """

    # amplitude → 밝기 (V)
    amp = np.clip(abs(amplitude), 0, 1)
    v = np.clip(0.2 + amp * 0.8, 0, 1)  # 조용할수록 어두운 톤, 클수록 밝아짐

    # pitch → hue (대략 저음: 차가운색, 고음: 따뜻한색/보라)
    if pitch <= 0:
        pitch_norm = 0.0
    else:
        pitch_norm = np.clip((pitch - 80) / 800, 0, 1)
    h = pitch_norm * 0.9  # 0~0.9 범위로 전체 스펙트럼 거의 다 사용

    # energy → saturation (E가 클수록 채도↑)
    energy_norm = np.clip(energy * 40, 0, 1)
    s = np.clip(0.25 + energy_norm * 0.75, 0, 1)

    # ZCR → hue jitter (색상 흔들림: 자음/노이즈 많을수록 더 흔들림)
    zcr_norm = np.clip(zcr * 8, 0, 1)
    h = (h + (random.random() - 0.5) * 0.25 * zcr_norm) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))

# ---------------------------------------------------------
# DRAWING STYLES
# ---------------------------------------------------------
def draw_line_art(t, y, feats, complexity, thickness, seed):
    """
    시간축을 따라 흐르는 여러 겹의 선.
    """
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
            ax.plot(
                t[i:i+2],
                y_line[i:i+2],
                color=color,
                linewidth=thickness,
                alpha=alpha,
            )

    return render_figure_to_bytes(fig)


def draw_scribble_art(t, y, feats, complexity, thickness, seed):
    """
    여러 겹의 낙서(scribble) 레이어를 겹쳐 그린 스타일.
    """
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
            ax.plot(
                t[i:i+2],
                y_line[i:i+2],
                color=color,
                linewidth=width,
                alpha=alpha,
            )

    return render_figure_to_bytes(fig)


def draw_contour_wave(t, y, feats, complexity, thickness, seed):
    """
    파형을 polar 좌표에 매핑해서 동심원/윤곽선처럼 그리는 스타일.
    """
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
            ax.plot(
                x[i:i+2],
                y2[i:i+2],
                color=color,
                linewidth=thickness * 0.7,
                alpha=0.7,
            )

    return render_figure_to_bytes(fig)


def draw_particle_drift(t, y, feats, complexity, thickness, seed):
    """
    각 샘플을 입자(점)로 생각해서, 소리의 흐름에 따라 흩날리는 점들을 찍는 스타일.
    """
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
    """
    나선형으로 퍼져 나가는 꽃/은하 같은 이미지.
    """
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
    radius = (0.1 + amp * 0.5)

    jitter = np.random.normal(scale=0.02 + zcr * 0.1, size=len(radius))
    radius = radius + jitter

    x = radius * np.cos(angles)
    y2 = radius * np.sin(angles)

    for i in range(len(x) - 1):
        color = get_dynamic_color(amp[i], pitch, energy, zcr)
        ax.plot(
            x[i:i+2],
            y2[i:i+2],
            color=color,
            linewidth=thickness * 0.9,
            alpha=0.8,
        )

    return render_figure_to_bytes(fig)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

drawing_style = st.sidebar.selectbox(
    "Drawing Style",
    ["Line Art", "Scribble Art", "Contour Wave", "Particle Drift", "Spiral Bloom"],
)

complexity = st.sidebar.slider("Complexity", 1, 10, 5)
thickness = st.sidebar.slider("Line / Stroke Thickness", 1, 6, 2)
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
    else:  # Spiral Bloom
        img_buf = draw_spiral_bloom(t, y_ds, feats, complexity, thickness, seed)

    st.image(
        img_buf,
        caption=f"{drawing_style} – audio-driven multi-color drawing",
        use_container_width=True,
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
# 🎨 Color Interpretation Guide (새 컬러 엔진용 설명)
# ---------------------------------------------------------
st.markdown("## 🎨 Color Interpretation Guide")

st.markdown("""
### 🌗 어두운 색 vs 밝은 색 (Value)
- **어두운 색** → 작은 amplitude (조용한 목소리, 속삭임, 긴장 낮은 구간)  
- **밝은 색** → 큰 amplitude (크게 말할 때, 감정이 올라간 구간)  

---

### 🌈 차가운 색 vs 따뜻한 색 (Hue)
- **차가운 색 (파랑·청록 계열)** → 상대적으로 **낮은 pitch**  
- **따뜻한 색 (노랑·주황·빨강·보라)** → **높은 pitch**, 고음·하이톤  

---

### 🎯 선명한 색 vs 흐린 색 (Saturation)
- **선명하고 쨍한 색** → **Energy(RMS)가 높은 구간**  
  - 강한 발성, 힘이 실린 말, 감정이 격한 부분  
- **흐릿하고 부드러운 색** → **Energy가 낮은 구간**  
  - 힘을 빼고 말하는 부분, 차분한 톤  

---

### 🌀 색이 자꾸 흔들리는 구간 (Jitter)
- **색이 빠르게 바뀌거나 무지개처럼 튀는 구간** → **ZCR(Zero Crossing Rate)이 높은 부분**  
  - 자음이 많이 섞인 발음, 숨소리, 잡음, 거친 소리들이 많을수록  
  - 선의 색이 더 불규칙하게 떨리며 표현됩니다.  

이렇게 한 장의 그림은  
**“얼마나 크게 말했는지(amplitude)”**,  
**“얼마나 높은 톤이었는지(pitch)”**,  
**“얼마나 힘이 실렸는지(energy)”**,  
**“얼마나 거칠게 발음했는지(ZCR)”**  
를 동시에 색으로 시각화하고 있습니다.
""")
