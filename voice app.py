# =========================================================
# WaveSketch (Auto Emotion Version Only / Safe Error-Free)
# Emotion = Auto-Detected by AssemblyAI / Audio = Color
# =========================================================

import io
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import colorsys
import requests
import time

# ---------------------------------------------------------
# Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="WaveSketch - Auto Emotion Detection",
    page_icon="🎧",
    layout="wide"
)

# ----------------------- 안내 텍스트 -----------------------
st.title("🎧 WaveSketch: Auto Emotion + Audio Colors")
st.write("""
Upload a short **WAV or MP3** file.

- Emotion is **automatically detected** from the audio  
- Detected emotion controls **line thickness**  
- Audio features control the **colors**
""")
st.caption("⚠️ m4a 파일은 Streamlit Cloud에서 지원되지 않습니다.")

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
# AssemblyAI Emotion Detection
# ---------------------------------------------------------
def assemblyai_upload(api_key, file):
    upload_url = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": api_key}

    file.seek(0)
    response = requests.post(upload_url, headers=headers, data=file)
    return response.json().get("upload_url")

def assemblyai_request_sentiment(api_key, audio_url):
    url = "https://api.assemblyai.com/v2/transcript"
    headers = {"authorization": api_key}
    data = {
        "audio_url": audio_url,
        "sentiment_analysis": True
    }

    res = requests.post(url, json=data, headers=headers).json()
    return res.get("id")

def assemblyai_poll(api_key, transcript_id):
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {"authorization": api_key}

    while True:
        res = requests.get(url, headers=headers).json()
        if res["status"] == "completed":
            return res
        if res["status"] == "error":
            return None
        time.sleep(1)

# ---------------------------------------------------------
# SAFE: 감정 분석 결과가 없을 때도 절대 오류 안 나게!
# ---------------------------------------------------------
def extract_dominant_emotion(result_json):

    # API 자체 실패 → neutral
    if not result_json:
        return "neutral"

    # 음악 파일·효과음·비언어적 오디오에서 자주 발생
    results = result_json.get("sentiment_analysis_results")

    # 감정 분석 불가 → neutral
    if not results or len(results) == 0:
        return "neutral"

    counts = {}
    for item in results:
        emo = item["sentiment"].lower()
        counts[emo] = counts.get(emo, 0) + 1

    return max(counts, key=counts.get)

# ---------------------------------------------------------
# AUDIO ANALYSIS → Color Controls
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
# COLOR ENGINE
# ---------------------------------------------------------
def get_audio_color(amplitude, pitch, rms, zcr):
    amp = np.clip(abs(amplitude), 0, 1)
    v = 0.3 + amp * 0.7
    pitch_norm = np.clip((pitch - 80) / 500, 0, 1)
    h = (0.65 - pitch_norm * 0.65) % 1.0
    s = np.clip(rms * 12, 0.25, 1.0)
    h = (h + (random.random() - 0.5) * zcr * 0.2) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

# ---------------------------------------------------------
# DRAWING
# ---------------------------------------------------------
def draw_line_style(t, y, feats, seed, emotion_mul):
    random.seed(seed)
    np.random.seed(seed)

    amp = y / (np.max(np.abs(y)) + 1e-8)
    base_y = 0.5 + amp * 0.35

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rms, pitch, zcr = feats["rms"], feats["pitch"], feats["zcr"]

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
# SIDEBAR → Only API Key + Seed
# ---------------------------------------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "AssemblyAI API Key (Required)",
    placeholder="Enter your API key...",
    type="password"
)

seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

if not api_key:
    st.warning("⚠️ Enter your AssemblyAI API Key to auto-detect emotion.")
    st.stop()

# ---------------------------------------------------------
# Upload Audio
# ---------------------------------------------------------
st.subheader("1️⃣ Upload Audio")
uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

if not uploaded_file:
    st.stop()

st.audio(uploaded_file)

# ---------------------------------------------------------
# AUTO EMOTION DETECTION
# ---------------------------------------------------------
st.subheader("2️⃣ Detecting Emotion from Audio…")

with st.spinner("Uploading & analyzing…"):
    audio_url = assemblyai_upload(api_key, uploaded_file)
    transcript_id = assemblyai_request_sentiment(api_key, audio_url)
    result_json = assemblyai_poll(api_key, transcript_id)

detected_emotion = extract_dominant_emotion(result_json)
emotion_mul = get_emotion_thickness_multiplier(detected_emotion)

st.success(f"🎭 Detected Emotion: **{detected_emotion}**")
st.caption("Emotion automatically controls line thickness.")

# ---------------------------------------------------------
# Audio Feature Extraction
# ---------------------------------------------------------
st.subheader("3️⃣ Audio Features")

with st.spinner("Extracting features…"):
    t, y_ds, feats = analyze_audio(uploaded_file)

st.json(feats)

# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------
st.subheader("4️⃣ Generated Drawing")

img_buf = draw_line_style(t, y_ds, feats, seed, emotion_mul)

st.image(
    img_buf,
    caption=f"Emotion: {detected_emotion} / Auto Emotion Analysis",
    use_container_width=True
)

st.download_button(
    label="⬇️ Download Image",
    data=img_buf,
    file_name="WaveSketch.png",
    mime="image/png"
)
