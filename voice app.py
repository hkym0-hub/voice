# app.py
# =========================================================
# WaveSketch: Multi-Color Drawing from Sound Waves
# - Audio upload → librosa analysis → generative drawings
# - Multi-color stroke mapping based on amplitude
# - 안정성 위해 WAV / MP3만 지원 (m4a는 환경상 지원 불가)
# =========================================================

import io
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa

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
    "The waveform becomes a **multi-colored drawing**, "
    "where quiet moments turn blue and loud moments turn red."
)
st.caption("⚠️ m4a는 서버 코덱 제한 때문에 지원되지 않습니다. 녹음 파일을 WAV/MP3로 변환해서 사용해 주세요.")

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
# AUDIO ANALYSIS (librosa.load 만 사용)
# ---------------------------------------------------------
def analyze_audio(uploaded_file, target_points=1200):
    """
    WAV / MP3 전용 분석.
    """
    uploaded_file.seek(0)
    # librosa가 UploadedFile 객체도 직접 읽을 수 있음
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)

    # 최대 10초까지만 사용
    if len(y) > 10 * sr:
        y = y[:10 * sr]

    n = len(y)
    idx = np.linspace(0, n - 1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0, 1, len(y_ds))

    # 특징 추출
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    try:
        pitch_vals = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        pitch_mean = float(np.nanmean(pitch_vals))
    except Exception:
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
# MULTI-COLOR MAPPING (quiet→blue → … → red)
# ---------------------------------------------------------
def get_color_from_amplitude(value):
    x = float(np.clip(value, 0, 1))

    if x < 0.25:
        return (0, x * 4, 1)  # blue→cyan
    elif x < 0.5:
        return (0, 1, 1 - (x - 0.25) * 4)  # cyan→green
    elif x < 0.75:
        return ((x - 0.5) * 4, 1, 0)  # green→yellow
    else:
        return (1, 1 - (x - 0.75) * 4, 0)  # yellow→red

# ---------------------------------------------------------
# DRAWINGS
# ---------------------------------------------------------
def draw_line_art(t, y, features, complexity, thickness, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    jitter_scale = 0.01 + energy_n * 0.03

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
            color = get_color_from_amplitude(amp_norm)

            ax.plot(
                t[j:j+2],
                y_line[j:j+2],
                color=color,
                linewidth=thickness,
                alpha=alpha,
            )

    return render_figure_to_bytes(fig)


def draw_scribble_art(t, y, features, complexity, thickness, seed):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25

    energy_n = normalize(features["rms_mean"], 0, 0.1)
    zcr_n = normalize(features["zcr_mean"], 0, 0.3)

    jitter_base = 0.02 + energy_n * 0.04
    jagged_factor = 0.01 + (1 - zcr_n) * 0.03

    n_paths = 5 + complexity * 3

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for _ in range(n_paths):
        jitter = np.random.normal(scale=jitter_base, size=len(base_y))
        jagged = np.random.normal(scale=jagged_factor, size=len(base_y))

        y_line = base_y + jitter + jagged * np.sign(np.gradient(base_y))

        alpha = 0.04 + 0.12 * np.random.rand()
        width = thickness * (0.5 + np.random.rand())

        for j in range(len(t) - 1):
            amp_norm = abs(amp[j])
            color = get_color_from_amplitude(amp_norm)

            ax.plot(
                t[j:j+2],
                y_line[j:j+2],
                color=color,
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
# MAIN
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")

uploaded_file = st.file_uploader(
    "Upload a short sound / voice file (WAV, MP3 only)",
    type=["wav", "mp3"],   # ❗ 여기서 m4a 제거
)

if uploaded_file:
    st.audio(uploaded_file)

    try:
        with st.spinner("Analyzing sound..."):
            t, y_ds, feats = analyze_audio(uploaded_file)
    except Exception as e:
        st.error("오디오를 불러오는 중 오류가 발생했습니다. 파일을 WAV/MP3로 변환해서 다시 시도해 주세요.")
        st.code(str(e))
    else:
        st.subheader("2️⃣ Audio Features")
        st.json(feats)

        st.subheader("3️⃣ Generated Multi-Color Drawing")

        if drawing_style == "Line Art":
            img_buf = draw_line_art(t, y_ds, feats, complexity, thickness, seed)
        else:
            img_buf = draw_scribble_art(t, y_ds, feats, complexity, thickness, seed)

        st.image(
            img_buf,
            caption=f"{drawing_style} – Multi-color amplitude mapping",
            use_container_width=True,
        )

        st.download_button(
            "📥 Download Drawing",
            img_buf,
            file_name="wavesketch_multicolor.png",
            mime="image/png",
        )
else:
    st.info("Upload a WAV or MP3 file to begin 🎨")
