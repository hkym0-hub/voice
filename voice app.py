# ---------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------
st.sidebar.header("Drawing Controls")

drawing_style = st.sidebar.selectbox(
    "Drawing Style",
    ["Line Art", "Scribble Art", "Contour Wave", "Particle Drift", "Spiral Bloom"]
)

complexity = st.sidebar.slider("Complexity", 1, 10, 5)
thickness = st.sidebar.slider("Line / Stroke Thickness", 1, 8, 3)
seed = st.sidebar.slider("Random Seed", 0, 9999, 42)

# ---------------------------------------------------------
# 🔑 API KEY INPUT (추가된 부분)
# ---------------------------------------------------------
st.sidebar.header("API Settings (optional)")
api_key = st.sidebar.text_input(
    "AssemblyAI API Key",
    type="password",
    placeholder="Enter your AssemblyAI API key..."
)

if api_key:
    st.sidebar.success("API Key registered ✔")
else:
    st.sidebar.info("API Key not set (emotion auto-detection disabled)")

# ---------------------------------------------------------
# Emotion Controls
# ---------------------------------------------------------
st.sidebar.header("Emotion Controls")
emotion_label = st.sidebar.selectbox(
    "Emotion",
    ["neutral", "joy", "sadness", "anger", "fear", "surprise"]
)
emotion_conf = st.sidebar.slider("Emotion Confidence", 0.0, 1.0, 0.7)
