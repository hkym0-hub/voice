from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

# app.py
# ------------------------------------------
# VoicePainter: Draw With Your Voice
# Streamlit + librosa + matplotlib
# ------------------------------------------

import io
import random
import colorsys

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import librosa

from matplotlib.patches import Circle, Ellipse, Polygon

# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(
    page_title="VoicePainter - Generative Poster from Voice",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 VoicePainter: Draw With Your Voice")
st.write(
    "Upload a short voice clip and this app will analyze its sound features "
    "and generate a **unique generative poster** based on your voice."
)


# -----------------------------
# 오디오 특징 추출 함수
# -----------------------------
def extract_audio_features(file) -> dict:
    """
    파일에서 오디오를 로드하고,
    pitch, energy, spectral centroid, rhythm 등을 추출하여 dict로 반환.
    """
    # librosa로 오디오 로딩
    # sr=None → 원래 샘플레이트 유지
    y, sr = librosa.load(file, sr=None, mono=True)

    # 너무 길면 앞부분 몇 초만 사용 (예: 10초)
    max_duration = 10.0
    if len(y) > max_duration * sr:
        y = y[: int(max_duration * sr)]

    # RMS 에너지
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # Zero-crossing rate (리듬/노이즈 느낌)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    # Spectral centroid (밝기/날카로움)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Tempo (BPM) 추정
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Pitch (기본 주파수) 추정
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
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zcr_mean": zcr_mean,
        "centroid_mean": centroid_mean,
        "tempo": float(tempo),
        "pitch_mean": pitch_mean,
    }
    return features


# -----------------------------
# 특징 → [0, 1] 정규화 도우미
# -----------------------------
def normalize(value, min_val, max_val):
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-8), 0.0, 1.0))


# -----------------------------
# 오디오 특징 → 색 팔레트 변환
# -----------------------------
def features_to_palette(features: dict, n_colors: int = 5):
    """
    pitch, rms, centroid 등을 이용해 HLS 공간에서 색상 팔레트 생성.
    """
    pitch = features["pitch_mean"]
    rms = features["rms_mean"]
    centroid = features["centroid_mean"]
    tempo = features["tempo"]
    zcr = features["zcr_mean"]

    # 대략적인 범위 가정 후 정규화
    pitch_n = normalize(pitch, 80.0, 800.0)          # Hz
    rms_n = normalize(rms, 0.0, 0.1)
    cent_n = normalize(centroid, 500.0, 5000.0)
    tempo_n = normalize(tempo, 40.0, 180.0)
    zcr_n = normalize(zcr, 0.0, 0.3)

    base_hue = pitch_n  # 0~1
    base_light = 0.3 + rms_n * 0.4  # 0.3~0.7
    base_sat = 0.4 + cent_n * 0.5   # 0.4~0.9

    palette = []
    for i in range(n_colors):
        # hue variation: tempo + index 기반으로 약간씩 회전
        hue_shift = (tempo_n * 0.3 + i * 0.12) % 1.0
        h = (base_hue + hue_shift) % 1.0

        # saturation/lightness에 약간의 변주
        s = np.clip(base_sat + (i - n_colors // 2) * 0.05, 0.25, 0.95)
        l = np.clip(base_light + (zcr_n - 0.5) * 0.2 + (i - n_colors // 2) * 0.03,
                    0.2, 0.85)

        r, g, b = colorsys.hls_to_rgb(h, l, s)
        palette.append((r, g, b))
    return palette


# -----------------------------
# Generative Poster 생성
# -----------------------------
def generate_poster(features: dict, palette, seed: int = 0):
    """
    오디오 특징과 팔레트를 바탕으로 추상 포스터를 생성하고
    PNG bytes를 반환.
    """
    random.seed(seed)
    np.random.seed(seed)

    rms = features["rms_mean"]
    tempo = features["tempo"]
    zcr = features["zcr_mean"]

    # 정규화된 값으로 shape 수, 크기, 거칠기 제어
    energy_n = normalize(rms, 0.0, 0.1)
    tempo_n = normalize(tempo, 40.0, 180.0)
    zcr_n = normalize(zcr, 0.0, 0.3)

    n_shapes = int(20 + energy_n * 50)       # 20 ~ 70 개
    max_radius = 0.1 + energy_n * 0.25      # 전체 크기
    noise_factor = 0.02 + zcr_n * 0.1       # 위치/형태 불규칙 정도

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_facecolor((0.02, 0.02, 0.04))  # 어두운 배경
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i in range(n_shapes):
        color = random.choice(palette)

        # 위치 (리듬이 일정할수록 중앙 집중, 불규칙하면 분산)
        cx = np.clip(
            0.5 + np.random.normal(0, 0.25 + noise_factor),
            0.0, 1.0
        )
        cy = np.clip(
            0.5 + np.random.normal(0, 0.25 + noise_factor),
            0.0, 1.0
        )

        # 모양 선택 (tempo에 따라 긴 타원/동그라미 비율 변화)
        shape_type_prob = tempo_n
        r = max_radius * (0.2 + np.random.rand())

        if np.random.rand() < 0.4 + 0.4 * shape_type_prob:
            # Ellipse
            width = r * (0.5 + np.random.rand())
            height = r * (0.5 + np.random.rand())
            angle = np.random.rand() * 360
            shape = Ellipse((cx, cy), width, height, angle=angle,
                            linewidth=0, color=color, alpha=0.8)
        elif np.random.rand() < 0.7:
            # Circle
            shape = Circle((cx, cy), r,
                           linewidth=0, color=color, alpha=0.8)
        else:
            # Polygon (삼각형/사각형 근처)
            k = random.choice([3, 4, 5])
            angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
            jitter = np.random.normal(0, noise_factor, size=k)
            xs = cx + (r + jitter) * np.cos(angles)
            ys = cy + (r + jitter) * np.sin(angles)
            points = np.stack([xs, ys], axis=1)
            shape = Polygon(points, closed=True,
                            linewidth=0, color=color, alpha=0.8)

        ax.add_patch(shape)

    # PNG로 저장해서 BytesIO 반환
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# -----------------------------
# Streamlit UI
# -----------------------------
st.subheader("1️⃣ Upload your voice")

uploaded_file = st.file_uploader(
    "Upload a short voice clip (WAV, MP3, OGG, M4A)",
    type=["wav", "mp3", "ogg", "m4a"]
)

seed = st.slider("Random Seed (poster variation)", 0, 9999, 42)

if uploaded_file is not None:
    # 오디오 플레이어
    st.audio(uploaded_file)

    # librosa에서 다시 읽을 수 있게 포인터 리셋
    uploaded_file.seek(0)

    with st.spinner("Analyzing your voice..."):
        features = extract_audio_features(uploaded_file)

    # 다시 포인터 리셋 (필요할 경우 대비)
    uploaded_file.seek(0)

    st.subheader("2️⃣ Extracted audio features")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Basic Stats**")
        st.write(f"- Sample rate: `{features['sr']:.0f} Hz`")
        st.write(f"- Pitch (mean f0): `{features['pitch_mean']:.1f} Hz`")
        st.write(f"- Tempo: `{features['tempo']:.1f} BPM`")

    with col2:
        st.write("**Energy & Texture**")
        st.write(f"- RMS energy (mean): `{features['rms_mean']:.5f}`")
        st.write(f"- Spectral centroid (mean): `{features['centroid_mean']:.1f}`")
        st.write(f"- Zero-crossing rate (mean): `{features['zcr_mean']:.4f}`")

    # 팔레트 생성
    palette = features_to_palette(features, n_colors=5)

    st.subheader("3️⃣ Generated color palette from your voice")
    # 팔레트를 matplotlib으로 시각화
    fig, ax = plt.subplots(figsize=(5, 1))
    for i, c in enumerate(palette):
        ax.add_patch(
            Rectangle = plt.Rectangle((i, 0), 1, 1, color=c)
        )
    ax.set_xlim(0, len(palette))
    ax.set_ylim(0, 1)
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("4️⃣ Generative poster")
    with st.spinner("Drawing your voice as an abstract poster..."):
        poster_buf = generate_poster(features, palette, seed=seed)
    st.image(poster_buf, caption="Your Voice Poster", use_container_width=True)

    st.download_button(
        label="📥 Download poster as PNG",
        data=poster_buf,
        file_name="voice_poster.png",
        mime="image/png",
    )

else:
    st.info("Please upload a short voice clip to start 🎧")
