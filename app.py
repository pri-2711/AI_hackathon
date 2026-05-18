import base64
import io
import os
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

MODEL_PATH = "vegetable_model.keras"
CLASS_LABELS = ["chili", "ivygourd", "ladyfinger", "peas", "pointedgourd"]
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "vegid")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "uploads")


def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        return None


@st.cache_resource
def get_database():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        return client[MONGO_DB]
    except ServerSelectionTimeoutError:
        return None
    except Exception:
        return None


def image_to_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def store_image_record(collection, image_bytes: bytes, filename: str, label: str, confidence: float):
    doc = {
        "filename": filename,
        "label": label,
        "confidence": float(confidence),
        "timestamp": datetime.utcnow(),
        "image_bytes": base64.b64encode(image_bytes).decode("utf-8"),
    }
    try:
        collection.insert_one(doc)
    except Exception as exc:
        st.warning(f"Could not save image to database: {exc}")


def load_recent_scans(collection, limit: int = 6, filter_label: str | None = None):
    query = {}
    if filter_label:
        query["label"] = filter_label
    return list(collection.find(query).sort("timestamp", -1).limit(limit))


def predict_image(model, image: Image.Image):
    try:
        img = image.convert("RGB").resize((128, 128))
        data = np.array(img, dtype=np.float32) / 255.0
        data = np.expand_dims(data, axis=0)
        preds = model.predict(data, verbose=0)[0]
        index = int(np.argmax(preds))
        return CLASS_LABELS[index], float(preds[index])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None, 0.0


def format_timestamp(ts):
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(ts)


def inject_style():
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

        /* ── Root palette ── */
        :root {
            --bg-base:        #070e09;
            --bg-surface:     #0c1810;
            --bg-raised:      #122016;
            --bg-hover:       #1a2e1e;
            --border:         #1f3826;
            --border-bright:  #2d5038;
            --accent:         #3a8c52;
            --accent-soft:    #2a6640;
            --accent-glow:    rgba(58, 140, 82, 0.18);
            --accent-light:   #6fc98a;
            --text-primary:   #e8f2ea;
            --text-secondary: #8aab90;
            --text-muted:     #4d6e54;
            --success:        #4caf72;
            --warning:        #c8a84b;
            --danger:         #c85a4c;
            --radius-sm:      6px;
            --radius-md:      12px;
            --radius-lg:      18px;
        }

        /* ── Global reset ── */
        html, body, [class*="css"] {
            font-family: 'DM Mono', monospace !important;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }

        /* ── App wrapper ── */
        .main .block-container {
            background-color: var(--bg-base) !important;
            padding: 2rem 2.5rem 4rem !important;
            max-width: 1200px !important;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-secondary) !important;
        }

        /* ── Headings ── */
        h1 {
            font-family: 'DM Serif Display', serif !important;
            font-size: 3.2rem !important;
            font-weight: 400 !important;
            letter-spacing: -0.5px !important;
            color: var(--accent-light) !important;
            line-height: 1.1 !important;
            margin-bottom: 0 !important;
        }
        h2, h3 {
            font-family: 'DM Serif Display', serif !important;
            font-weight: 400 !important;
            color: var(--text-primary) !important;
            letter-spacing: 0.2px !important;
        }
        h5 {
            font-family: 'DM Mono', monospace !important;
            font-size: 0.78rem !important;
            font-weight: 300 !important;
            letter-spacing: 2.5px !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
            margin-top: 0.1rem !important;
        }
        h4 {
            font-family: 'DM Serif Display', serif !important;
            font-weight: 400 !important;
            color: var(--accent-light) !important;
            font-size: 1.3rem !important;
        }

        /* ── Divider ── */
        hr {
            border: none !important;
            border-top: 1px solid var(--border) !important;
            margin: 1.8rem 0 !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            background-color: var(--bg-surface) !important;
            border: 1.5px dashed var(--border-bright) !important;
            border-radius: var(--radius-md) !important;
            padding: 1.2rem !important;
            transition: border-color 0.2s ease, background 0.2s ease !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--accent) !important;
            background-color: var(--bg-hover) !important;
        }
        [data-testid="stFileUploader"] * {
            color: var(--text-secondary) !important;
        }

        /* ── Camera input ── */
        [data-testid="stCameraInput"] {
            border: 1.5px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
            background: var(--bg-surface) !important;
        }
        [data-testid="stCameraInput"] button {
            background-color: var(--bg-raised) !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border-bright) !important;
        }

        /* ── Image display ── */
        [data-testid="stImage"] img {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
        }

        /* ── Primary button ── */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-soft) 0%, var(--accent) 100%) !important;
            color: #fff !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            letter-spacing: 1.8px !important;
            text-transform: uppercase !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.6rem 1.6rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 12px var(--accent-glow) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%) !important;
            box-shadow: 0 4px 20px rgba(58,140,82,0.35) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button:active {
            transform: translateY(0px) !important;
        }

        /* ── Success / info / warning / error banners ── */
        [data-testid="stAlert"] {
            border-radius: var(--radius-md) !important;
            border: none !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.85rem !important;
        }
        div[data-testid="stAlert"][data-baseweb="notification"] {
            background-color: rgba(76, 175, 114, 0.1) !important;
            border-left: 3px solid var(--success) !important;
            color: var(--accent-light) !important;
        }
        div.stWarning > div {
            background-color: rgba(200, 168, 75, 0.08) !important;
            border-left: 3px solid var(--warning) !important;
        }
        div.stError > div {
            background-color: rgba(200, 90, 76, 0.08) !important;
            border-left: 3px solid var(--danger) !important;
        }
        div.stInfo > div {
            background-color: rgba(58, 140, 82, 0.08) !important;
            border-left: 3px solid var(--accent) !important;
            color: var(--text-secondary) !important;
        }

        /* ── Radio buttons ── */
        [data-testid="stRadio"] label {
            font-family: 'DM Mono', monospace !important;
            font-size: 0.75rem !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
            color: var(--text-secondary) !important;
        }
        [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
            border-color: var(--accent) !important;
            background-color: transparent !important;
        }
        [data-testid="stRadio"] [aria-checked="true"] [data-baseweb="radio"] > div:first-child {
            background-color: var(--accent) !important;
        }

        /* ── Scan cards ── */
        .scan-card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1rem 1.1rem 0.8rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s ease, background 0.2s ease;
        }
        .scan-card:hover {
            border-color: var(--border-bright);
            background: var(--bg-raised);
        }
        .scan-label {
            font-family: 'DM Serif Display', serif;
            font-size: 1.05rem;
            color: var(--accent-light);
            display: inline-block;
            margin-bottom: 0.1rem;
        }
        .scan-meta {
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 0.6rem;
        }
        .confidence-bar-wrap {
            height: 3px;
            background: var(--border);
            border-radius: 99px;
            margin: 0.5rem 0 0.25rem;
            overflow: hidden;
        }
        .confidence-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-soft), var(--accent-light));
            border-radius: 99px;
        }
        .confidence-text {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 1px;
            color: var(--accent);
        }

        /* ── Subheader refinement ── */
        [data-testid="stMarkdownContainer"] p {
            font-family: 'DM Mono', monospace !important;
            font-size: 0.85rem !important;
            color: var(--text-secondary) !important;
            line-height: 1.7 !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 99px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-soft); }

        /* ── Subtle grid texture on body ── */
        .main::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(var(--border) 1px, transparent 1px),
                linear-gradient(90deg, var(--border) 1px, transparent 1px);
            background-size: 40px 40px;
            opacity: 0.18;
            pointer-events: none;
            z-index: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="VegID", layout="wide", page_icon="🥬")
    inject_style()

    st.markdown("# VegID")
    st.markdown("##### Local Vegetable Classifier")
    st.markdown("Upload or snap an image, verify it with our CNN model, and store the result for later review.")

    db = get_database()
    collection = db[MONGO_COLLECTION] if db is not None else None
    if db is None:
        st.warning(
            "MongoDB is not available. Images will still be processed locally, but upload history will not be saved. "
            "Set MONGO_URI to connect to your database."
        )

    model = load_model()
    if model is None:
        st.error("Model missing or could not be loaded. Ensure vegetable_model.keras exists in the project root.")

    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Identify a vegetable")
            uploaded_file = st.file_uploader(
                "Upload or drag an image",
                type=["jpg", "jpeg", "png", "heic"],
                help="Supports JPG, PNG, HEIC. Max 10 MB.",
            )
            camera_image = st.camera_input("Open Camera")
            image = None
            source_name = None

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                source_name = uploaded_file.name
            elif camera_image is not None:
                image = Image.open(camera_image)
                source_name = "camera_capture.png"

            if image is not None:
                st.image(image, caption="Uploaded image", use_column_width=True)
                if model is not None and st.button("Identify Vegetable"):
                    label, confidence = predict_image(model, image)
                    if label:
                        st.success(f"Detected: {label.title()} ({confidence * 100:.1f}% confidence)")
                        if collection is not None:
                            store_image_record(collection, image_to_bytes(image), source_name, label, confidence)
                        else:
                            st.info("Image was classified locally but not saved because MongoDB is unavailable.")

        with col2:
            st.subheader("Filter by vegetable")
            filter_label = st.radio(
                "",
                options=["All"] + [label.title() for label in CLASS_LABELS],
                horizontal=True,
            )
            query_label = None if filter_label == "All" else filter_label.lower()

            st.subheader("Recent scans")
            if collection is not None:
                recent = load_recent_scans(collection, limit=6, filter_label=query_label)
                if recent:
                    for scan in recent:
                        conf_pct = scan['confidence'] * 100
                        st.markdown(
                            f"""
                            <div class="scan-card">
                                <div class="scan-label">{scan['label'].title()}</div>
                                <div class="scan-meta">{format_timestamp(scan['timestamp'])}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        try:
                            image_data = base64.b64decode(scan["image_bytes"])
                            st.image(image_data, width=280)
                        except Exception:
                            st.write("Unable to load saved preview.")
                        st.markdown(
                            f"""
                                <div class="confidence-bar-wrap">
                                    <div class="confidence-bar-fill" style="width:{conf_pct:.1f}%"></div>
                                </div>
                                <div class="confidence-text">Confidence: {conf_pct:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("---")
                else:
                    st.info("No recent scans yet. Start by uploading a vegetable image!")
            else:
                st.info("Connect MongoDB to see recent uploads and verification history.")

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        "- Upload or take an image with your camera.\n"
        "- Click Identify Vegetable.\n"
        "- Result is shown instantly and stored in MongoDB for verification."
    )
    if db is not None:
        st.markdown(f"Database connected to `{MONGO_URI}` and collection `{MONGO_COLLECTION}`.")


if __name__ == "__main__":
    main()