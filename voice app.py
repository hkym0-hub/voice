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
    "Your voice becomes a multi-color drawing based on amplitude, pitch, energy, and rhythm."
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
# Color Theme Base Palettes
# ---------------------------------------------------------
THEME_BASE = {
    "Pastel": (0.55, 0.45, 0.90),
    "Neon":   (0.10, 0.95, 0.85),
    "Ink":    (0.05, 0.05, 0.10),
    "Fire":   (0.95, 0.40, 0.10),
    "Ocean":  (0.10, 0.40, 0.80),
}

# ---------------------------------------------------------
# AUDIO ANALYSIS
# ---------------------------------------------------------
def analyze_audio(uploaded_file, target_points=1200):
    uploaded_file.seek(0)
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)

    if len(y) > 10 * sr:
        y = y[:10 * sr]

    idx = np.linspace(0, len(y)-1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0, 1, len(y_ds))

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
# 🌈 FULL COLOR ENGINE
# ---------------------------------------------------------
def get_dynamic_color(amplitude, pitch, energy, zcr, theme_name):

    # 1) Theme Base RGB → HSV
    r0, g0, b0 = THEME_BASE[theme_name]
    h, s, v = colorsys.rgb_to_hsv(r0, g0, b0)

    # 2) amplitude → 밝기
    amp = np.clip(abs(amplitude), 0, 1)
    v = np.clip(0.35 + amp * 0.65, 0, 1)

    # 3) pitch → hue shift
    pitch_norm = np.clip((pitch - 80) / 600, 0, 1)
    h = (h + pitch_norm * 0.30) % 1.0

    # 4) energy → saturation
    energy_norm = np.clip(energy * 35, 0, 1)
    s = np.clip(0.25 + energy_norm * 0.75, 0, 1)

    # 5) ZCR → jitter
    jitter = (random.random() - 0.5) * (zcr * 1.8)
    h = (h + jitter) % 1.0

    theme_rgb = colorsys.hsv_to_rgb(h, s, v)

    # 6) amplitude-gradient (Blue→Red)
    if amp < 0.25:
        grad = (0, amp * 4, 1)
    elif amp < 0.5:
        grad = (0, 1, 1 - (amp - 0.25) * 4)
    elif amp < 0.75:
        grad = ((amp - 0.5) * 4, 1, 0)
    else:
        grad = (1, 1 - (amp - 0.75) * 4, 0)

    # Blend Theme + Gradient
    final_r = theme_rgb[0] * 0.55 + grad[0] * 0.45
    final_g = theme_rgb[1] * 0.55 + grad[1] * 0.45
    final_b = theme_rgb[2] * 0.55 + grad[2] * 0.45

    return (final_r, final_g, final_b)


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
            color = get_dynamic_color(amp[i], pitch, energy, zcr, theme_name)
            ax.plot(t[i:i+2], y_line[i:i+2], color=color,
                    linewidth=thickness, alpha=alpha)

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
            color = get_dynamic_color(amp[i], pitch, energy, zcr, theme_name)
            ax.plot(t[i:i+2], y_line[i:i+2], color=color,
                    linewidth=width, alpha=alpha)

    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

drawing_style = st.sidebar.selectbox("Drawing Style", ["Line Art", "Scribble Art"])
theme_name = st.sidebar.selectbox("Color Theme",
                                  ["Pastel", "Neon", "Ink", "Fire", "Ocean"])
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
        img_buf = draw_line_art(t, y_ds, feats, complexity, thickness, seed, theme_name)
    else:
        img_buf = draw_scribble_art(t, y_ds, feats, complexity, thickness, seed, theme_name)

    st.image(img_buf, caption=f"{drawing_style} with {theme_name} theme",
             use_container_width=True)

    st.download_button(
        "📥 Download Image",
        img_buf,
        file_name="wavesketch.png",
        mime="image/png"
    )

else:
    st.info("Please upload a WAV or MP3 file 🎵")

# ---------------------------------------------------------
# 🎨 Art Style
# ---------------------------------------------------------
def draw_contour_wave(t, y, feats, complexity, thickness, seed, theme_name)
def draw_particle_drift(t, y, feats, complexity, thickness, seed, theme_name)
def draw_spiral_bloom(t, y, feats, complexity, thickness, seed, theme_name)

def draw_contour_wave(t, y, feats, complexity, thickness, seed, theme_name):
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

        for i in range(len(x)-1):
            color = get_dynamic_color(amp[i], pitch, energy, zcr, theme_name)
            ax.plot(x[i:i+2], y2[i:i+2], color=color,
                    linewidth=thickness * 0.7, alpha=0.7)

    return render_figure_to_bytes(fig)

def draw_particle_drift(t, y, feats, complexity, thickness, seed, theme_name):
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

        color = get_dynamic_color(amp[i], pitch, energy, zcr, theme_name)

        ax.scatter(drift_x, drift_y, color=color, s=size * 8, alpha=0.7)

    return render_figure_to_bytes(fig)

def draw_spiral_bloom(t, y, feats, complexity, thickness, seed, theme_name):
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

    for i in range(len(x)-1):
        color = get_dynamic_color(amp[i], pitch, energy, zcr, theme_name)
        ax.plot(x[i:i+2], y2[i:i+2], color=color,
                linewidth=thickness * 0.9, alpha=0.8)

    return render_figure_to_bytes(fig)



# ---------------------------------------------------------
# 🎨 Color Interpretation Guide (추가된 부분)
# ---------------------------------------------------------
st.markdown("## 🎨 Color Interpretation Guide")

st.markdown("""
### 🔵 차가운·어두운 색 (Blue / Cyan)
**→ 낮은 Amplitude (작은 음량, 조용한 발성)**  
속삭임, 조용한 파형, 안정적인 톤에서 나타납니다.  

---

### 🟢 초록 계열
**→ 중간 Amplitude + 안정된 Pitch**  
일반적인 말하기 톤, 감정 변화가 적은 구간입니다.  

---

### 🟡🟠🔴 밝고 따뜻한 색 (Yellow / Orange / Red)
**→ 높은 Amplitude + 강한 Energy(RMS)**  
크게 말하는 구간, 감정이 실린 톤, 고음/강세를 의미합니다.  

---

### 💜 보라 / 💗 핑크 계열
**→ Pitch(음높이)가 높아질 때 Hue가 이동**  
더 높은 음역대에서 색조가 화려해지고 따뜻해집니다.  

---

### 🌀 색 흔들림(Jitter)
**→ ZCR(Zero Crossing Rate, 소리의 거칠기)**  
거친 자음, 숨소리, 노이즈가 많을수록 색상이 흔들립니다.  
""")
