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
        .main { background-color: #0d1117; color: #e5e7eb; }
        section[data-testid='stSidebar'] { background-color: #0b1220; }
        .css-1d391kg { color: #f8fafc; }
        .stButton>button { background-color: #1f6feb; color: white; }
        .st-b5 { background-color: rgba(96, 165, 250, 0.12); }
        .css-1v0mbdj { color: #e5e7eb; }
        .css-18ni7ap { color: #cbd5e1; }
        .css-1aumxhk { background-color: #111827; }
        .css-1cgyl8q { background-color: #111827; }
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
                        st.write(f"**{scan['label'].title()}** — {format_timestamp(scan['timestamp'])}")
                        try:
                            image_data = base64.b64decode(scan["image_bytes"])
                            st.image(image_data, width=280)
                        except Exception:
                            st.write("Unable to load saved preview.")
                        st.markdown(f"Confidence: **{scan['confidence'] * 100:.1f}%**")
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
