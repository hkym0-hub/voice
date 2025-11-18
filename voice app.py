# app.py
# =========================================================
# WaveSketch: Drawing with Sound Waves
# - Streamlit app
# - Audio upload → librosa analysis → generative drawings
# - Styles: line-art / scribble / contour / charcoal
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
    page_title="WaveSketch - Sound Drawings",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 WaveSketch: Drawing with Sound Waves")
st.write(
    "Upload a short voice or sound clip. "
    "This app analyzes the waveform and generates **drawing-style art** "
    "based on sound intensity and movement."
)


# ---------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------
def normalize(value, min_val, max_val):
    """간단한 0~1 정규화"""
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-8), 0.0, 1.0))


def render_figure_to_bytes(fig):
    """matplotlib Figure → PNG bytes"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------
# 오디오 분석 함수
# ---------------------------------------------------------
def analyze_audio(file, target_points=1200):
    """
    오디오 파일을 로드하고:
    - downsample된 waveform
    - 시간축 t (0~1)
    - 특징값들(dict)
    을 반환
    """
    y, sr = librosa.load(file, sr=None, mono=True)

    # 너무 길면 10초까지만 사용
    max_duration = 10.0
    if len(y) > max_duration * sr:
        y = y[: int(max_duration * sr)]

    # 다운샘플링 (드로잉용 포인트 수 줄이기)
    n_samples = len(y)
    idx = np.linspace(0, n_samples - 1, target_points, dtype=int)
    y_ds = y[idx]
    t = np.linspace(0.0, 1.0, len(y_ds))

    # 특징 추출
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # pitch 추정 (yin)
    try:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        pitch_mean = float(np.nanmean(f0))
    except Exception:
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
# 드로잉 스타일 1: Line Art
# ---------------------------------------------------------
def draw_line_art(t, y, features, complexity, thickness, seed):
    """
    파형을 여러 겹의 선으로 표현하는 라인 드로잉.
    - amplitude: 중앙에서 위/아래로 휘어진 정도
    - complexity: 레이어 수
    - thickness: 선 두께
    """
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35

    # 특징값 기반 약간의 왜곡
    energy_n = normalize(features["rms_mean"], 0.0, 0.1)
    jitter_scale = 0.01 + energy_n * 0.03

    n_layers = 1 + complexity  # 최소 2~11 정도

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i in range(n_layers):
        offset = (i - (n_layers - 1) / 2) * 0.03  # 위아래로 약간씩 평행 이동
        jitter = np.random.normal(scale=jitter_scale, size=len(base_y))
        y_line = base_y + offset + jitter * (i / max(1, n_layers - 1))

        alpha = 0.25 + 0.5 * (1 - i / max(1, n_layers - 1))
        ax.plot(
            t,
            y_line,
            color="black",
            linewidth=thickness,
            alpha=alpha,
        )

    fig.tight_layout()
    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# 드로잉 스타일 2: Scribble Art
# ---------------------------------------------------------
def draw_scribble_art(t, y, features, complexity, thickness, seed):
    """
    에너지와 리듬에 따라 흐트러진 낙서(scribble) 스타일.
    - complexity: 낙서 레이어 수
    - thickness: 기본 선 두께
    """
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.25

    energy_n = normalize(features["rms_mean"], 0.0, 0.1)
    zcr_n = normalize(features["zcr_mean"], 0.0, 0.3)

    # 리듬이 일정할수록 더 규칙적인 scribble
    jitter_base = 0.02 + energy_n * 0.04
    jagged_factor = 0.01 + (1 - zcr_n) * 0.03

    n_paths = 5 + complexity * 3  # 여러 겹의 낙서

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i in range(n_paths):
        jitter = np.random.normal(scale=jitter_base, size=len(base_y))
        jagged = np.random.normal(scale=jagged_factor, size=len(base_y))

        # 불규칙한 spline 느낌
        y_line = base_y + jitter + jagged * np.sign(np.gradient(base_y))

        alpha = 0.03 + 0.15 * np.random.rand()
        width = thickness * (0.5 + np.random.rand())

        ax.plot(
            t,
            y_line,
            color="black",
            linewidth=width,
            alpha=alpha,
        )

    fig.tight_layout()
    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# 드로잉 스타일 3: Contour Drawing
# ---------------------------------------------------------
def draw_contour_drawing(t, y, features, complexity, thickness, seed):
    """
    파형을 polar 좌표계에 매핑해서 윤곽(contour) 드로잉처럼 표현.
    - 하나의 큰 윤곽 + 여러 보조 윤곽선
    """
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    energy_n = normalize(features["rms_mean"], 0.0, 0.1)
    tempo_n = normalize(features["tempo"], 40.0, 180.0)

    base_radius = 0.3 + energy_n * 0.2
    radius = base_radius + amp * 0.25

    # angle은 시간 + tempo에 따른 회전
    angles = 2 * np.pi * (t * (1.5 + tempo_n * 2.0))

    x_base = 0.5 + radius * np.cos(angles)
    y_base = 0.5 + radius * np.sin(angles)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_contours = 2 + complexity  # 여러 겹의 윤곽

    for i in range(n_contours):
        offset_r = (i - (n_contours - 1) / 2) * 0.015
        jitter = np.random.normal(scale=0.005 + 0.01 * energy_n, size=len(radius))

        r_line = radius + offset_r + jitter
        x = 0.5 + r_line * np.cos(angles)
        y_line = 0.5 + r_line * np.sin(angles)

        alpha = 0.15 + 0.25 * (1 - i / max(1, n_contours - 1))
        ax.plot(
            x,
            y_line,
            color="black",
            linewidth=thickness,
            alpha=alpha,
        )

    fig.tight_layout()
    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# 드로잉 스타일 4: Charcoal / Ink Style
# ---------------------------------------------------------
def draw_charcoal_style(t, y, features, complexity, thickness, seed):
    """
    짧은 스트로크들을 여러 번 겹쳐서 목탄/잉크 느낌을 내는 스타일.
    - complexity: 스트로크 개수
    - thickness: 기본 선 두께
    """
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.2

    energy_n = normalize(features["rms_mean"], 0.0, 0.1)
    centroid_n = normalize(features["centroid_mean"], 500.0, 5000.0)

    n_strokes = 80 + complexity * 40
    stroke_min_len = int(len(t) * 0.03)
    stroke_max_len = int(len(t) * 0.12)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_facecolor("#f5f5f0")  # 약간 노란 종이 느낌
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for _ in range(n_strokes):
        # 랜덤 구간 선택
        start = random.randint(0, len(t) - stroke_min_len - 1)
        length = random.randint(stroke_min_len, stroke_max_len)
        end = min(len(t) - 1, start + length)

        x_seg = t[start:end]
        y_seg = base_y[start:end]

        # 약간 굽이치는 jitter
        jitter = np.random.normal(
            scale=0.01 + energy_n * 0.03,
            size=len(y_seg),
        )
        y_seg = y_seg + jitter

        # 목탄느낌: 반투명 짙은 회색
        alpha = 0.02 + 0.12 * np.random.rand()
        gray = 0.1 + 0.4 * (1 - centroid_n)  # brighter sound → lighter stroke

        ax.plot(
            x_seg,
            y_seg,
            color=(gray, gray, gray),
            linewidth=thickness * np.random.uniform(0.7, 1.5),
            alpha=alpha,
        )

    fig.tight_layout()
    return render_figure_to_bytes(fig)


# ---------------------------------------------------------
# SIDEBAR 설정
# ---------------------------------------------------------
st.sidebar.header("Controls")

drawing_style = st.sidebar.selectbox(
    "Drawing Style",
    ["Line Art", "Scribble Art", "Contour Drawing", "Charcoal / Ink"],
)

complexity = st.sidebar.slider(
    "Complexity (more lines / strokes)",
    min_value=1,
    max_value=10,
    value=5,
)

thickness = st.sidebar.slider(
    "Base Line Thickness",
    min_value=1,
    max_value=6,
    value=2,
)

seed = st.sidebar.slider(
    "Random Seed",
    min_value=0,
    max_value=9999,
    value=42,
)


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")

uploaded_file = st.file_uploader(
    "Upload a short voice/sound clip (WAV, MP3, M4A)",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    uploaded_file.seek(0)
    with st.spinner("Analyzing sound..."):
        t, y_ds, feats = analyze_audio(uploaded_file)

    st.subheader("2️⃣ Audio Features (for drawing decisions)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Sample Rate: `{feats['sr']:.0f} Hz`")
        st.write(f"Mean RMS Energy: `{feats['rms_mean']:.5f}`")
        st.write(f"Zero-Crossing Rate: `{feats['zcr_mean']:.4f}`")
    with col2:
        st.write(f"Spectral Centroid (mean): `{feats['centroid_mean']:.1f}`")
        st.write(f"Tempo (BPM): `{feats['tempo']:.1f}`")
        st.write(f"Pitch (mean f0): `{feats['pitch_mean']:.1f} Hz`")

    st.subheader("3️⃣ Generated Drawing")

    if drawing_style == "Line Art":
        img_buf = draw_line_art(t, y_ds, feats, complexity, thickness, seed)
    elif drawing_style == "Scribble Art":
        img_buf = draw_scribble_art(t, y_ds, feats, complexity, thickness, seed)
    elif drawing_style == "Contour Drawing":
        img_buf = draw_contour_drawing(t, y_ds, feats, complexity, thickness, seed)
    else:  # Charcoal / Ink
        img_buf = draw_charcoal_style(t, y_ds, feats, complexity, thickness, seed)

    st.image(img_buf, caption=f"{drawing_style} from your sound", use_container_width=True)

    st.subheader("4️⃣ Download")
    st.download_button(
        label="📥 Download Drawing as PNG",
        data=img_buf,
        file_name="wavesketch_drawing.png",
        mime="image/png",
    )

else:
    st.info("Please upload a sound file to start 🎨")
