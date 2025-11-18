# =========================================================
# 🔹 PAGE: MEMORY ATLAS
# =========================================================
elif page == "🌈 Memory Atlas":
    st.title("🌈 Memory Atlas: Weather → Color → Generative Art")

    city = st.text_input("City Name (English)", "Seoul")
    date = st.date_input("Select a date")
    seed = st.slider("Poster variation (seed)", 0, 9999, 22)

    st.markdown("""---""")

    # -------- LAT/LON lookup (optional: default Seoul)
    # You can replace this with a real geocoding API if needed
    city_coords = {
        "Seoul": (37.5665, 126.9780),
        "Tokyo": (35.6895, 139.6917),
        "New York": (40.7128, -74.0060),
        "Paris": (48.8566, 2.3522),
        "London": (51.5074, -0.1278)
    }

    lat, lon = city_coords.get(city, (37.5665, 126.9780))

    if st.button("Generate Memory Poster"):
        st.subheader("📡 Fetching Weather Data...")

        # -------------------------
        # API REQUEST
        url = (
            f"{api_base}?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relativehumidity_2m,cloudcover&timezone=auto"
        )

        try:
            data = requests.get(url).json()
            temp = np.mean(data["hourly"]["temperature_2m"])
            humidity = np.mean(data["hourly"]["relativehumidity_2m"])
            cloud = np.mean(data["hourly"]["cloudcover"])
        except:
            st.error("API request failed.")
            st.stop()

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temp (avg)", f"{temp:.1f} °C")
        col2.metric("💧 Humidity", f"{humidity:.1f} %")
        col3.metric("☁️ Cloud Cover", f"{cloud:.1f} %")

        st.markdown("### 1️⃣ Generated Memory Palette")

        # -------------------------
        # WEATHER → COLOR PALETTE
        h = normalize(temp, -5, 35)        # temp → hue
        s = normalize(humidity, 20, 100)   # humidity → saturation
        l = 0.3 + normalize(cloud, 0, 100) * 0.4   # cloud → lightness

        palette = []
        for i in range(5):
            h2 = (h + i * 0.12) % 1.0
            rgb = colorsys.hls_to_rgb(h2, l, s)
            palette.append(rgb)

        # palette preview
        fig, ax = plt.subplots(figsize=(4, 1))
        for i, c in enumerate(palette):
            ax.add_patch(Rectangle((i, 0), 1, 1, color=c))
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.axis("off")
        st.pyplot(fig)

        st.markdown("### 2️⃣ Weather-Based Generative Poster")

        # -------------------------
        # GENERATE POSTER
        random.seed(seed)
        np.random.seed(seed)

        fig2, ax2 = plt.subplots(figsize=(6, 9))
        ax2.set_facecolor("black")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        # poster shape amount based on humidity + cloud
        complexity = int(25 + (humidity / 100) * 30 + (cloud / 100) * 20)

        for _ in range(complexity):
            color = random.choice(palette)

            cx = np.random.rand()
            cy = np.random.rand()

            # size varies with temperature
            size = 0.03 + (normalize(temp, -5, 35) * 0.15)

            # choose shape
            shape_type = random.random()
            if shape_type < 0.45:
                # circle
                shape = Circle((cx, cy), size, color=color, alpha=0.8)
            else:
                # ellipse
                w = size * np.random.uniform(0.4, 1.5)
                h = size * np.random.uniform(0.4, 1.5)
                angle = random.random() * 360
                shape = Ellipse((cx, cy), w, h, angle=angle, color=color, alpha=0.8)

            ax2.add_patch(shape)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        st.image(buf, caption=f"Memory Poster for {city} on {date}", use_container_width=True)

        st.download_button(
            "📥 Download Memory Poster",
            buf,
            file_name=f"memory_atlas_{city}_{date}.png",
            mime="image/png"
        )
